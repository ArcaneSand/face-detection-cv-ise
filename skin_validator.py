"""
Skin-color sanity check on a face ROI.
"""

import cv2
import numpy as np


def is_skin_region(face_roi_bgr):
    """
    Returns (passed, skin_ratio). Builds an HSV mask, runs open + close,
    and passes if 30% or more of the pixels survive.
    """
    hsv = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2HSV)

    # HSV range tuned for indoor webcam lighting.
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skin_ratio = cv2.countNonZero(mask) / mask.size
    return skin_ratio > 0.30, skin_ratio
