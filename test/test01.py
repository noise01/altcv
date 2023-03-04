if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2 as cv

from altcv import Camera

camera = Camera()

while True:
    frame = camera.read()
    print(frame)
    cv.imshow("capture", frame)
    

camera.release()

