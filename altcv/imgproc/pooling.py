import numpy as np


def average_pooling(img: np.ndarray, pool_size) -> np.ndarray:
    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=-1)

    dst = img.copy()

    h, w, c = img.shape
    step = pool_size

    for y in range(0, h, step):
        for x in range(0, w, step):
            for i in range(c):
                dst[y : y + step, x : x + step, i] = np.mean(
                    img[y : y + step, x : x + step, i]
                )

    if img.shape[2] == 1:
        dst = np.squeeze(dst, axis=-1)

    return dst


def max_pooling(img: np.ndarray, pool_size) -> np.ndarray:
    if len(img.shape) != 3:
        img = np.expand_dims(img, axis=-1)

    dst = img.copy()

    h, w, c = img.shape
    step = pool_size

    for y in range(0, h, step):
        for x in range(0, w, step):
            for i in range(c):
                dst[y : y + step, x : x + step, i] = np.max(
                    img[y : y + step, x : x + step, i]
                )

    if img.shape[2] == 1:
        dst = np.squeeze(dst, axis=-1)

    return dst
