from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import cv2 as cv


# class PoseEstimation:
#     dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
#     parameters = cv.aruco.DetectorParameters_create()
#     parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR

#     @classmethod
#     def estimate_marker_rt(
#         cls,
#         img: np.ndarray,
#         camera_matrix: np.ndarray,
#         dist_coeffs: np.ndarray,
#         marker_length: float,
#         marker_id: int,
#     ) -> tuple[int, np.ndarray, np.ndarray]:
#         corners, ids, _ = cv.aruco.detectMarkers(
#             img, cls.dictionary, parameters=cls.parameters
#         )

#         num_markers = 0
#         if ids is not None:
#             rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
#                 corners, marker_length, camera_matrix, dist_coeffs
#             )

#             marker_idxs = np.where(ids == [marker_id])[0]
#             num_markers = len(marker_idxs)
#             if num_markers == 1:
#                 marker_idx = marker_idxs[0]

#                 rvec = rvecs[marker_idx, 0]
#                 tvec = tvecs[marker_idx, 0]

#         return num_markers, rvec, tvec

#     @classmethod
#     def get_all_markers_rt(
#         cls, cameras: list[Camera], marker_length: float, marker_id: int
#     ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
#         all_num_markers = []
#         rvecs = []
#         tvecs = []

#         num_cameras = len(cameras)
#         for i in range(num_cameras):
#             camera = cameras[i]
#             num_markers, rvec, tvec = None, None, None

#             if camera.is_intrinsic():
#                 frame = camera.read()
#                 camera_matrix = camera.camera_matrix
#                 dist_coeffs = camera.dist_coeffs

#                 num_markers, rvec, tvec = cls.estimate_marker_rt(
#                     frame, camera_matrix, dist_coeffs, marker_length, marker_id
#                 )

#             all_num_markers.append(num_markers)
#             rvecs.append(rvec)
#             tvecs.append(tvec)

#         return all_num_markers, rvecs, tvecs


class CoordXForm:
    @classmethod
    def cvt_img2world(
        cls,
        img_pt: np.ndarray,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        cam_pt = cls.cvt_img2cam(img_pt, camera_matrix)
        world_pt = cls.cvt_cam2world(cam_pt, rvec, tvec)
        # world_pt = np.linalg.inv(w) @ (np.linalg.inv(camera_matrix) @ img_pt)

        return world_pt

    @classmethod
    def cvt_world2img(
        cls,
        world_pt: np.ndarray,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        cam_pt = cls.cvt_world2cam(world_pt, rvec, tvec)
        img_pt = cls.cvt_cam2img(cam_pt, camera_matrix)

        return img_pt

    @classmethod
    def cvt_img2cam(cls, img_pt: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        img_pt = cls.hom(img_pt)

        cam_pt = np.linalg.inv(camera_matrix) @ img_pt

        return cam_pt

    @classmethod
    def cvt_cam2img(cls, cam_pt: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        img_pt = camera_matrix @ cam_pt

        return cls.hom_inv(img_pt)

    @classmethod
    def cvt_cam2world(
        cls, cam_pt: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> np.ndarray:
        w = cls.calc_w_matrix(rvec, tvec)

        world_pt = np.linalg.inv(w) @ cam_pt

        return cls.hom_inv(world_pt)

    @classmethod
    def cvt_world2cam(
        cls, world_pt: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> np.ndarray:
        world_pt = cls.hom(world_pt)
        w = cls.calc_w_matrix(rvec, tvec)

        cam_pt = w @ world_pt

        return cam_pt

    @staticmethod
    def calc_w_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        r, _ = cv.Rodrigues(rvec)
        t = tvec

        w = np.insert(r, 3, t, axis=1)
        w = np.delete(w, 2, axis=1)

        return w

    @staticmethod
    def hom(pt: np.ndarray) -> np.ndarray:
        pt: list = pt.tolist()
        pt.append(1)

        return np.array(pt)

    @staticmethod
    def hom_inv(pt: np.ndarray) -> np.ndarray:
        pt /= pt[-1]

        return pt[:-1]

    @staticmethod
    def cvt_cam2cam(cam_pt: np.ndarray, ww: np.ndarray) -> np.ndarray:
        cam_pt = ww @ cam_pt

        return cam_pt


class WorldMeasure:
    ref_camera: Camera = None

    x_offset = 0.0
    y_offset = 0.0
    t_offset = 0.0

    @classmethod
    def estimate_world_pt(
        cls, img_pt: tuple[int, int], camera: Camera
    ) -> tuple[float, float]:
        cam_pt = CoordXForm.cvt_img2cam(img_pt, camera.camera_matrix)
        if not camera.is_ref_camera():
            cam_pt = CoordXForm.cvt_cam2cam(cam_pt, camera.ww_matrix)

        world_pt = CoordXForm.cvt_cam2world(
            cam_pt, cls.ref_camera.rvec, cls.ref_camera.tvec
        )

        world_pt = cls._correct_pt(
            world_pt, -cls.x_offset, -cls.y_offset, -cls.t_offset
        )

        return tuple(world_pt)

    @classmethod
    def estimate_img_pt(
        cls, world_pt: tuple[float, float], camera: Camera
    ) -> tuple[int, int]:
        world_pt = cls._correct_pt(world_pt, cls.x_offset, cls.y_offset, cls.t_offset)

        cam_pt = CoordXForm.cvt_world2cam(
            world_pt, cls.ref_camera.rvec, cls.ref_camera.tvec
        )
        if not camera.is_ref_camera():
            cam_pt = CoordXForm.cvt_cam2cam(cam_pt, camera.ww_inv_matrix)

        img_pt = CoordXForm.cvt_cam2img(cam_pt, camera.camera_matrix)

        return tuple(map(int, img_pt))

    @classmethod
    def _correct_pt(cls, pt: np.ndarray, x: float, y: float, t: float) -> np.ndarray:
        mat = np.array(
            [[np.cos(t), -np.sin(t), x], [np.sin(t), np.cos(t), y], [0, 0, 1]]
        )

        pt = CoordXForm.hom(pt)
        pt = mat @ pt
        pt = CoordXForm.hom_inv(pt)

        return pt

    @classmethod
    def estimate_distance(
        cls,
        img_pt1: tuple[int, int],
        img_pt2: tuple[int, int],
        camera: Camera,
    ) -> float:
        world_pt1 = cls.estimate_world_pt(img_pt1, camera)
        world_pt2 = cls.estimate_world_pt(img_pt2, camera)

        return np.sqrt(np.sum((np.array(world_pt1) - np.array(world_pt2)) ** 2))

    @classmethod
    def reset_settings(cls, cameras: list[Camera], ref_camera_id: int) -> None:
        cls.cameras = cameras
        cls.ref_camera = cameras[ref_camera_id]


class Measure:
    offset = np.array([0.0, 0.0])

    @classmethod
    def estimate_world_pt(
        cls,
        img_pt: tuple[int, int],
        cameras: list[Camera],
        camera_id: int,
        ref_camera_id: int,
    ) -> tuple[float, float]:
        ref_camera = cameras[ref_camera_id]
        ref_camera_matrix = ref_camera.camera_matrix

        if camera_id != ref_camera_id:
            camera = cameras[camera_id]
            camera_matrix = camera.camera_matrix
            ww = camera.ww_matrix

            # ToDo: rivise
            img_pt = cls._convert_img2img(img_pt, camera_matrix, ref_camera_matrix, ww)

        ref_rvec = ref_camera.rvec
        ref_tvet = ref_camera.tvec

        world_pt = cls._convert_img2world(img_pt, ref_camera_matrix, ref_rvec, ref_tvet)

        return tuple(np.array(world_pt) - cls.offset)

    @classmethod
    def estimate_distance(
        cls,
        img_pt1: tuple[int, int],
        img_pt2: tuple[int, int],
        cameras: list[Camera],
        camera_id: int,
        ref_camera_id: int,
    ) -> float:
        world_pt1 = cls.estimate_world_pt(img_pt1, cameras, camera_id, ref_camera_id)
        world_pt2 = cls.estimate_world_pt(img_pt2, cameras, camera_id, ref_camera_id)

        return np.sqrt(np.sum((np.array(world_pt1) - np.array(world_pt2)) ** 2))

    @classmethod
    def estimate_img_pt(
        cls,
        world_pt: tuple[float, float],
        cameras: list[Camera],
        camera_id: int,
        ref_camera_id: int,
    ) -> tuple[int, int]:
        world_pt = np.array(world_pt) + cls.offset

        ref_camera = cameras[ref_camera_id]
        ref_camera_matrix = ref_camera.camera_matrix
        ref_rvec = ref_camera.rvec
        ref_tvec = ref_camera.tvec

        img_pt = cls._convert_world2img(world_pt, ref_camera_matrix, ref_rvec, ref_tvec)

        if camera_id != ref_camera_id:
            camera = cameras[camera_id]
            camera_matrix = camera.camera_matrix
            ww_inv = camera.ww_inv_matrix

            img_pt = cls._convert_img2img(
                img_pt, camera_matrix, ref_camera_matrix, ww_inv
            )

        return img_pt

    @classmethod
    def _convert_img2world(
        cls,
        img_pt: tuple[int, int],
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[float, float]:
        img_pt = list(img_pt)
        img_pt.append(1)
        p_i = np.array(img_pt)

        w = cls.calc_w(rvec, tvec)

        p_w = np.linalg.inv(camera_matrix @ w) @ p_i
        p_w /= p_w[-1]
        world_pt = p_w[:-1]

        return tuple(world_pt)

    @classmethod
    def _convert_world2img(
        cls,
        world_pt: tuple[float, float],
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> tuple[int, int]:
        world_pt = list(world_pt)
        world_pt.append(1)
        p_w = np.array(world_pt)

        w = cls.calc_w(rvec, tvec)

        p_i = camera_matrix @ w @ p_w
        p_i /= p_i[-1]
        img_pt = p_i[:-1]

        return tuple(map(int, img_pt))

    @staticmethod
    def calc_w(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        r, _ = cv.Rodrigues(rvec)
        t = tvec

        w = np.insert(r, 3, t, axis=1)
        w = np.delete(w, 2, axis=1)

        return w

    @staticmethod
    def _convert_img2img(
        img_pt: tuple[int, int],
        camera_matrix1: np.ndarray,
        camera_matrix2: np.ndarray,
        ww_matrix: np.ndarray,
    ) -> tuple[int, int]:
        img_pt = list(img_pt)
        img_pt.append(1)
        p1_i = np.array(img_pt)

        p2_i = camera_matrix2 @ ww_matrix @ np.linalg.inv(camera_matrix1) @ p1_i
        p2_i /= p2_i[-1]
        img_pt = p2_i[:-1]

        return tuple(map(int, img_pt))


class CameraRelation:
    @staticmethod
    def get_all_markers_rt(
        cameras: list[Camera], marker_length: float, marker_id: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        num_markers_list = []
        rvecs = []
        tvecs = []

        num_cameras = len(cameras)
        for i in range(num_cameras):
            camera = cameras[i]
            num_markers, rvec, tvec = None, None, None

            if camera.is_intrinsic():
                frame = camera.read()
                camera_matrix = camera.camera_matrix
                dist_coeffs = camera.dist_coeffs

                num_markers, rvec, tvec = PoseEstimation.estimate_marker_rt(
                    frame, camera_matrix, dist_coeffs, marker_length, marker_id
                )

            num_markers_list.append(num_markers)
            rvecs.append(rvec)
            tvecs.append(tvec)

        incidence = np.where(np.array(num_markers_list) == 1, 1, 0)

        return rvecs, tvecs, incidence

    @staticmethod
    def calc_ww(
        rvec1: np.ndarray, tvec1: np.ndarray, rvec2: np.ndarray, tvec2: np.ndarray
    ) -> np.ndarray:
        w1 = Measure.calc_w(rvec1, tvec1)
        w2 = Measure.calc_w(rvec2, tvec2)

        return w2 @ np.linalg.inv(w1)