# Real-Time Face Detection — CV Course Project

A real-time webcam face detector. Primary detection is a classical Haar
Cascade pipeline; a small CNN is used only as a supplementary verifier.

## Requirements
- Python 3.9+
- A working webcam

## Setup
    git clone <REPO-URL>
    cd face-detection-cv
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Download CNN model (required)
Place these two files into `models/`:
  - opencv_face_detector_uint8.pb
  - opencv_face_detector.pbtxt

Source: https://github.com/opencv/opencv/tree/master/samples/dnn

## Run
    python main.py
Press `q` to quit.

## Pipeline
[Person 3: describe each stage with the lecture it comes from.]

## Team & Contributions
[Person 3: list members and what each did.]