import numpy as np


def cvt_bgr2rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        return

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    dst = np.stack([r, g, b], axis=2)

    return dst


def cvt_bgr2gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        return

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    dst = 0.2126 * r + 0.7152 * g + 0.0722 * b
    dst = dst.astype(np.uint8)

    return dst


def cvt_bgr2hsv(img: np.ndarray) -> np.ndarray:
    if len(img.shape) != 3:
        return

    bgr = img.copy() / 255.0

    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]

    max_val = np.max(img, axis=2).copy()
    min_val = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    hsv = np.zeros_like(img, dtype=np.float32)

    idx = np.where(max_val == min_val)
    hsv[:, :, 0][idx] = 0
    with np.errstate(all="ignore"):
        idx = np.where(min_arg == 0)
        hsv[:, :, 0][idx] = 60 * (g[idx] - r[idx]) / (max_val[idx] - min_val[idx]) + 60
        idx = np.where(min_arg == 1)
        hsv[:, :, 0][idx] = 60 * (r[idx] - b[idx]) / (max_val[idx] - min_val[idx]) + 300
        idx = np.where(min_arg == 2)
        hsv[:, :, 0][idx] = 60 * (b[idx] - g[idx]) / (max_val[idx] - min_val[idx]) + 180

    hsv[:, :, 1] = max_val.copy() - min_val.copy()
    hsv[:, :, 2] = max_val.copy()

    return hsv


def dicrease_color(img: np.ndarray) -> np.ndarray:
    dst = img.copy()

    dst = (dst // 64) * 64 + 32

    return dst


def gamma_correction(img: np.ndarray, gamma=2.2) -> np.ndarray:
    dst = img.copy()

    dst /= 255.0
    dst = dst ** (1 / gamma)
    dst *= 255
    dst = dst.astype(np.uint8)

    return dst
