"""
evaluate.py: run the pipeline on WIDER FACE (val split) and report P/R/F1.

use python evaluate.py
"""

import os
import sys
import cv2
import numpy as np
from datasets import load_dataset
import detector
import skin_validator
import cnn_verifier


EXAMPLES_DIR  = "wider_examples"
NUM_IMAGES    = 300        # small subset of the ~3,200-image val split
IOU_THRESHOLD = 0.5
NUM_EXAMPLES  = 5


def pil_to_bgr(pil_image):
    """PIL RGB image -> OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect(frame):
    """Full Haar + eye + skin + CNN pipeline; return list of (x, y, w, h)."""
    gray = detector.preprocess(frame)
    out = []
    for (x, y, w, h) in detector.detect_faces(gray):
        face_gray = gray[y:y + h, x:x + w]
        face_bgr  = frame[y:y + h, x:x + w]

        if len(detector.detect_eyes(face_gray)) < 1:
            continue

        skin_ok, _ = skin_validator.is_skin_region(face_bgr)
        if not skin_ok:
            continue

        cnn_ok, _ = cnn_verifier.verify_face_cnn(face_bgr)
        if not cnn_ok:
            continue

        out.append((x, y, w, h))
    return out


def match(preds, gts):
    """Greedy IoU match. Return (tp, fp, fn, pred_is_tp, gt_is_matched)."""
    gt_matched = [False] * len(gts)
    pred_tp    = [False] * len(preds)
    tp = 0
    for pi, p in enumerate(preds):
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if gt_matched[j]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= IOU_THRESHOLD:
            gt_matched[best_j] = True
            pred_tp[pi] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - sum(gt_matched)
    return tp, fp, fn, pred_tp, gt_matched


def save_example(path, frame, preds, gts, pred_tp, gt_matched):
    """Save the frame with TP=green, FP=red, FN=yellow boxes drawn on it."""
    img = frame.copy()
    for (x, y, w, h), is_tp in zip(preds, pred_tp):
        color = (0, 255, 0) if is_tp else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    for (x, y, w, h), is_matched in zip(gts, gt_matched):
        if not is_matched:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imwrite(path, img)


def load_wider_val():
    """Load the WIDER FACE validation split from Hugging Face."""
    print("Loading WIDER FACE validation split from Hugging Face ...")
    try:
        return load_dataset("wider_face", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: could not load 'wider_face' from Hugging Face: {e}")
        print("Tip: `pip install -U datasets` and check your network connection.")
        sys.exit(1)


def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    ds = load_wider_val()

    tp = fp = fn = 0
    n_images = n_faces = 0
    saved = 0

    for idx, sample in enumerate(ds):
        if idx >= NUM_IMAGES:
            break

        frame = pil_to_bgr(sample["image"])
        gts = [tuple(map(int, bb)) for bb in sample["faces"]["bbox"]]

        preds = detect(frame)
        i_tp, i_fp, i_fn, pred_tp, gt_matched = match(preds, gts)
        tp += i_tp; fp += i_fp; fn += i_fn
        n_images += 1
        n_faces  += len(gts)

        if saved < NUM_EXAMPLES and preds and gts:
            out_path = os.path.join(EXAMPLES_DIR, f"example_{saved + 1:02d}.png")
            save_example(out_path, frame, preds, gts, pred_tp, gt_matched)
            saved += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    print()
    print(f"WIDER FACE val (first {n_images} images, {n_faces} GT faces)")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print(f"Example overlays saved to {EXAMPLES_DIR}/")


if __name__ == "__main__":
    main()
