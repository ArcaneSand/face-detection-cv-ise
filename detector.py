"""
detector.py — Preprocessing + Haar Cascade face & eye detection.
Owner: Person 1.

Lecture coverage:
  L02  OpenCV basics
  L03  BGR -> Grayscale
  L04  CLAHE (adaptive histogram equalization)
  L05  Gaussian blur (linear filter, smoothing)
  L06  Linear filtering for noise reduction
  L07  CLAHE as image restoration
  L11  Haar Cascade face & eye detection (Viola-Jones)
"""

import cv2

# Pre-trained Haar Cascade XMLs ship with OpenCV.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# CLAHE object reused across frames — caps contrast amplification to avoid
# boosting noise when the lighting is already uniform.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(bgr_frame):
    """
    BGR webcam frame -> denoised, contrast-normalized grayscale.
      1) BGR -> Gray         (L03)
      2) Gaussian blur 5x5   (L05/L06) — reduces webcam sensor noise
      3) CLAHE               (L04/L07) — fixes uneven lighting locally
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = CLAHE.apply(blurred)
    return equalized


def detect_faces(gray_frame):
    """Run Haar Cascade. Returns list of (x, y, w, h) tuples."""
    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,    # multi-scale step
        minNeighbors=5,     # higher = fewer false positives
        minSize=(60, 60),   # ignore tiny detections
    )
    return list(faces)


def detect_eyes(face_roi_gray):
    """Detect eyes inside a cropped face ROI. Used as a sanity check."""
    eyes = EYE_CASCADE.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
    )
    return list(eyes)