import numpy as np
import cv2 as cv
from PIL import Image


def pil2cv(image: np.ndarray) -> np.ndarray:
    dst_image = np.array(image, dtype=np.uint8)
    if dst_image.ndim == 2:
        pass
    elif dst_image.shape[2] == 3:
        dst_image = cv.cvtColor(dst_image, cv.COLOR_RGB2BGR)
    elif dst_image.shape[2] == 4:
        dst_image = cv.cvtColor(dst_image, cv.COLOR_RGBA2BGR)
    return dst_image


def cv2pil(image: np.ndarray) -> np.ndarray:
    dst_image = image.copy()
    if dst_image.ndim == 2:
        pass
    elif dst_image.shape[2] == 3:
        dst_image = cv.cvtColor(dst_image, cv.COLOR_BGR2RGB)
    elif dst_image.shape[2] == 4:
        dst_image = cv.cvtColor(dst_image, cv.COLOR_BGRA2RGBA)
    dst_image = Image.fromarray(dst_image)
    return dst_image
