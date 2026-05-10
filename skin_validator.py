"""
skin_validator.py — Skin-color verification.
Owner: Person 2.

Lectures to apply when filling this in:
  L04  HSV color space
  L08  Thresholding (cv2.inRange) for binary segmentation
  L09  Morphological opening + closing, contour analysis
"""

import cv2
import numpy as np


def is_skin_region(face_roi_bgr):
    """
    Verify that a candidate face ROI is plausibly a face by skin color.

    Returns:
        passed (bool):      True if skin_ratio > threshold
        skin_ratio (float): fraction of pixels classified as skin (0.0 - 1.0)

    Implementation steps (Person 2):
        1. hsv = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2HSV)
        2. Build skin mask with cv2.inRange. Starter ranges:
               lower = np.array([0, 30, 60])
               upper = np.array([20, 150, 255])
           Tune on actual webcam frames from your team.
        3. Clean the mask with morphology (L09):
               kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
               mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
               mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        4. skin_ratio = cv2.countNonZero(mask) / mask.size
        5. return (skin_ratio > 0.30, skin_ratio)
    """
    # ===========================================================
    #                    WORK HERE (Person 2)
    # ===========================================================


    # ===========================================================

    # Stub: passes everything so the pipeline still runs end-to-end.
    return True, 1.0