import cv2
import numpy as np


def get_in_cal(vid):
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)

    return cameraMatrix, distCoeffs