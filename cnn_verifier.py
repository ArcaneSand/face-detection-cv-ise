"""
cnn_verifier.py: Supplementary CNN check on a face ROI. 

The Haar stage is the primary detector; this gives a yes/no second opinion using OpenCV's pretrained
SSD face detector.
"""

import os
import cv2

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
PROTO_PATH   = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")

_NET = None


def _load_net():
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromTensorflow(WEIGHTS_PATH, PROTO_PATH)
    return _NET


def verify_face_cnn(face_roi_bgr, conf_threshold=0.5):
    """
    Run the SSD detector over an ROI that passed the Haar stage. Returns
    (passed, max_confidence). If no proposal clears conf_threshold, the
    ROI is rejected.
    """
    # 300x300 input with Caffe BGR mean subtraction, per the SSD's training config.
    blob = cv2.dnn.blobFromImage(
        face_roi_bgr, 1.0, (300, 300),
        [104, 117, 123], swapRB=False, crop=False,
    )

    net = _load_net()
    net.setInput(blob)
    detections = net.forward()  # shape (1, 1, N, 7); col 2 is the confidence

    # Best score across all proposals.
    max_conf = 0.0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > max_conf:
            max_conf = conf

    return max_conf >= conf_threshold, float(max_conf)
