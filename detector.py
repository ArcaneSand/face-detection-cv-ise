"""
detector.py: preprocessing + Haar Cascade face/eye detection.
"""

import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Keep CLAHE from amplifying sensor noise.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(bgr_frame):
    """BGR frame -> grayscale, blurred, CLAHE-equalized."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return CLAHE.apply(blurred)


def detect_faces(gray_frame):
    """Run the Haar cascade and return a list of (x, y, w, h)."""
    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return list(faces)


def detect_eyes(face_roi_gray):
    """Detect eyes inside a cropped face ROI."""
    eyes = EYE_CASCADE.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
    )
    return list(eyes)