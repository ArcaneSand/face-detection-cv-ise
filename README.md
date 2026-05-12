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

## CNN model
The pretrained SSD weights live in `models/`:
  - `opencv_face_detector_uint8.pb`
  - `opencv_face_detector.pbtxt`

Source: https://github.com/opencv/opencv/tree/master/samples/dnn

## Run
    python main.py
Press `q` to quit.

## Evaluate

Run the pipeline on the first 300 images of the WIDER FACE validation
split and report Precision / Recall / F1 at IoU >= 0.5. The dataset is
pulled from the Hugging Face Hub (cached under `~/.cache/huggingface/`)
on first run; no manual download is required.

    python evaluate.py

Prints `TP / FP / FN / Precision / Recall / F1`. Five qualitative
example overlays (TP green / FP red / FN yellow) are written to
`wider_examples/`.

## Pipeline

Each webcam frame passes through six stages, listed below with the
lecture each one comes from.

1. **Capture** (L02 OpenCV basics)
   Read a BGR frame from the webcam with `cv2.VideoCapture`.

2. **Preprocess** (L03, L04, L05, L06, L07)
   Convert BGR to grayscale (L03), apply a 5x5 Gaussian blur to suppress
   sensor noise (L05, L06), and apply CLAHE to fix uneven lighting
   (L04, L07).

3. **Haar face detection** (L11 Viola-Jones)
   Run `haarcascade_frontalface_default.xml` over the preprocessed gray.
   Pass every surviving box into the per-face verification stages below.

4. **Eye sanity check** (L11)
   Run `haarcascade_eye.xml` inside each candidate ROI. Drop any box
   with zero detected eyes; without an eye match, the Haar hit is more
   likely background texture than a real face.

5. **Skin-color check** (L04 HSV, L08 thresholding, L09 morphology)
   Convert the ROI to HSV (L04), build a skin mask with `cv2.inRange`
   (L08), and clean it with open + close morphology (L09). The ROI
   passes when 30% or more of the pixels survive.

6. **CNN verifier** (L12 ML/DL fundamentals, L13 CNNs)
   Push the ROI through OpenCV's pretrained SSD face detector. The ROI
   passes when the top proposal clears 0.5 confidence.

Draw a bounding box only when the ROI clears every stage above.

## Team & Contributions
[Person 3: list members and what each did.]