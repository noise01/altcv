import numpy as np


def binarization(gray_img: np.ndarray, th: int) -> np.ndarray:
    if len(gray_img.shape) != 2:
        return

    dst = np.where(gray_img < th, 0, 255)
    dst = dst.astype(np.uint8)

    return dst


def otsu_binarization(gray_img: np.ndarray) -> np.ndarray:
    # ToDo: mu -> n

    max_sigma_b = 0
    max_th = 0

    for th in range(np.min(gray_img), np.max(gray_img) + 1):
        n0 = np.count_nonzero(gray_img < th)
        if n0 == 0:
            mu0 = 0
        else:
            mu0 = np.mean(gray_img[gray_img < th])

        n1 = np.count_nonzero(gray_img >= th)
        if n1 == 0:
            mu1 = 0
        else:
            mu1 = np.mean(gray_img[gray_img >= th])

        sigma_b = n0 * n1 * (mu0 - mu1) ** 2

        if sigma_b > max_sigma_b:
            max_sigma_b = sigma_b
            max_th = th

    dst = binarization(gray_img, max_th)

    return dst
