import cv2 as cv
import numpy as np


class Camera:
    def __init__(self, camera_id=0) -> None:
        vcap = cv.VideoCapture(0)
        if not vcap.isOpened():
            raise Exception("VideoCapture error.")

        self.vcap: cv.VideoCapture = vcap
        self.camera_matrix: np.ndarray = None
        self.dist_coeffs: np.ndarray = None
        self.rvec: np.ndarray = None
        self.tvec: np.ndarray = None

    def read(self, undistort=False) -> np.ndarray:
        _, frame = self.vcap.read()
        print(frame)
        if undistort:
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv.undistort(frame, self.camera_matrix, self.dist_coeffs)

        return frame

    def release(self) -> None:
        self.vcap.release()

    @property
    def frame_width(self) -> int:
        return self.vcap.get(cv.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self) -> int:
        return self.vcap.get(cv.CAP_PROP_FRAME_HEIGHT)

    @property
    def camera_position(self) -> tuple[float, float, float]:
        if self.is_extrinsic():
            r, _ = cv.Rodrigues(self.rvec)
            t = self.tvec

            return tuple(-r.T @ t.T)

    @property
    def camera_position(self) -> tuple[float, float, float]:
        if self.rvec is not None and self.tvec is not None:
            r, _ = cv.Rodrigues(self.rvec)
            t = self.tvec

            return tuple(-r.T @ t.T)
