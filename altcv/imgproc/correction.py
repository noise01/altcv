import numpy as np


def hist_equalization(gray_img: np.ndarray) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    h, w = gray_img.shape
    s = h * w

    i_max = gray_img.max()

    dst = gray_img.copy()

    hist_cnt = 0.0
    for i in range(255):
        idx = np.where(gray_img == i)
        hist_cnt += len(gray_img[idx])
        dst[idx] = i_max / s * hist_cnt

    dst = dst.astype(np.uint8)

    return dst
