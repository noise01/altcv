import numpy as np


def median_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=-1)

    h, w, c = img.shape

    pad = ksize // 2
    dst = np.zeros((h + pad * 2, w + pad * 2, c), dtype=np.float32)
    dst[pad : pad + h, pad : pad + w] = img.copy().astype(np.float32)
    tmp = dst.copy()

    for y in range(h):
        for x in range(w):
            for i in range(c):
                dst[y + pad, x + pad, i] = np.median(
                    tmp[y : y + ksize, x : x + ksize, i]
                )

    dst = np.clip(dst, 0, 255)
    dst = dst[pad : pad + h, pad : pad + w].astype(np.uint8)

    if img.shape[2] == 1:
        dst = np.squeeze(dst, axis=-1)

    return dst


def mean_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=-1)

    h, w, c = img.shape

    pad = ksize // 2
    dst = np.zeros((h + pad * 2, w + pad * 2, c), dtype=np.float32)
    dst[pad : pad + h, pad : pad + w] = img.copy().astype(np.float32)
    tmp = dst.copy()

    for y in range(h):
        for x in range(w):
            for i in range(c):
                dst[y + pad, x + pad, i] = np.mean(tmp[y : y + ksize, x : x + ksize, i])

    dst = np.clip(dst, 0, 255)
    dst = dst[pad : pad + h, pad : pad + w].astype(np.uint8)

    if img.shape[2] == 1:
        dst = np.squeeze(dst, axis=-1)

    return dst


def _filter(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=-1)

    h, w, c = img.shape
    ksize = len(ker)

    pad = ksize // 2
    dst = np.zeros((h + pad * 2, w + pad * 2, c), dtype=np.float32)
    dst[pad : pad + h, pad : pad + w] = img.copy().astype(np.float32)
    tmp = dst.copy()

    for y in range(h):
        for x in range(w):
            for i in range(c):
                dst[y + pad, x + pad, i] = np.sum(
                    tmp[y : y + ksize, x : x + ksize, i] * ker
                )

    dst = np.clip(dst, 0, 255)
    dst = dst[pad : pad + h, pad : pad + w].astype(np.uint8)

    if img.shape[2] == 1:
        dst = np.squeeze(dst, axis=-1)

    return dst


def gaussian_filter(img: np.ndarray, ksize=3, sigma=1.3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    pad = ksize // 2

    ker = np.zeros((ksize, ksize), dtype=np.float32)
    for y in range(-pad, -pad + ksize):
        for x in range(-pad, -pad + ksize):
            ker[y + pad, x + pad] = np.exp(-(x**2 + y**2) / 2 / sigma**2)
    ker /= 2 * np.pi * sigma**2
    ker /= np.sum(ker)

    dst = _filter(img, ker)

    return dst


def diagonal_motion_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    ker = np.eye(ksize, dtype=np.float32)
    ker /= ksize

    dst = _filter(img, ker)

    return dst


def anti_diagonal_motion_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    ker = np.eye(ksize, dtype=np.float32)
    ker = np.fliplr(ker)
    ker /= ksize

    dst = _filter(img, ker)

    return dst


def horizontal_motion_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    pad = ksize // 2

    ker = np.zeros((ksize, ksize), dtype=np.float32)
    ker[pad, :] = 1.0
    ker /= ksize

    dst = _filter(img, ker)

    return dst


def vertical_motion_filter(img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    pad = ksize // 2

    ker = np.zeros((ksize, ksize), dtype=np.float32)
    ker[:, pad] = 1.0
    ker /= ksize

    dst = _filter(img, ker)

    return dst


def max_min_filter(gray_img: np.ndarray, ksize=3) -> np.ndarray:
    if ksize % 2 == 0:
        return

    if len(gray_img.shape) != 2:
        return

    h, w = gray_img.shape

    pad = ksize // 2
    dst = np.zeros((h + pad * 2, w + pad * 2), dtype=np.float32)
    dst[pad : pad + h, pad : pad + w] = gray_img.copy().astype(np.float32)
    tmp = dst.copy()

    for y in range(h):
        for x in range(w):
            dst[y + pad, x + pad] = np.max(tmp[y : y + ksize, x : x + ksize]) - np.min(
                tmp[y : y + ksize, x : x + ksize]
            )

    dst = np.clip(dst, 0, 255)
    dst = dst[pad : pad + h, pad : pad + w].astype(np.uint8)

    return dst


def horizontal_differential_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])

    dst = _filter(gray_img, ker)

    return dst


def vertical_differential_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    dst = _filter(gray_img, ker)

    return dst


def horizontal_sobel_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    dst = _filter(gray_img, ker)

    return dst


def vertical_sobel_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    dst = _filter(gray_img, ker)

    return dst


def horizontal_prewitt_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    dst = _filter(gray_img, ker)

    return dst


def vertical_prewitt_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    dst = _filter(gray_img, ker)

    return dst


def laplacian_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    dst = _filter(gray_img, ker)

    return dst


def emboss_filter(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    dst = _filter(gray_img, ker)

    return dst


def log_filter(gray_img: np.ndarray, ksize=5, sigma=3) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    pad = ksize // 2

    ker = np.zeros((ksize, ksize), dtype=np.float32)
    for y in range(-pad, -pad + ksize):
        for x in range(-pad, -pad + ksize):
            ker[y + pad, x + pad] = (x**2 + y**2 - 2 * sigma**2) * np.exp(
                -(x**2 + y**2) / 2 / sigma**2
            )
    ker /= 2 * np.pi * sigma**6
    ker /= np.sum(ker)

    dst = _filter(gray_img, ker)

    return dst
