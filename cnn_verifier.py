"""
cnn_verifier.py — Supplementary CNN-based face verifier.
Owner: Person 2.

Per the project brief, Deep Learning may ONLY be used to verify / enhance
results from traditional CV — NOT as the primary detector. This module
receives ROIs that the Haar Cascade already accepted, and gives a yes/no
second opinion.

Lectures: L12 (ML/DL fundamentals), L13 (CNNs).
"""

import os
import cv2

# Person 2: download these two files into the models/ folder:
#   opencv_face_detector_uint8.pb
#   opencv_face_detector.pbtxt
# Source:
#   https://github.com/opencv/opencv/tree/master/samples/dnn
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
PROTO_PATH   = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")

_NET = None  # lazy-loaded once


def _load_net():
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromTensorflow(WEIGHTS_PATH, PROTO_PATH)
    return _NET


def verify_face_cnn(face_roi_bgr, conf_threshold=0.5):
    """
    Returns (passed: bool, confidence: float).

    """

    blob = cv2.dnn.blobFromImage(face_roi_bgr, 1.0, (300, 300),
                                         [104, 117, 123], swapRB=False, crop=False)
    
    net = _load_net()
    net.setInput(blob)
    detections = net.forward()
    max_conf = 0.0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > max_conf:
            max_conf = conf
    
    return max_conf >= conf_threshold, float(max_conf)