"""
Real-time face detection demo. Run `python main.py`, press q to quit.
"""

import time
import cv2

import detector
import skin_validator
import cnn_verifier


def draw_box(frame, x, y, w, h, color=(0, 255, 0), label="Face"):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame, label, (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = detector.preprocess(frame)
        candidate_faces = detector.detect_faces(gray)

        for (x, y, w, h) in candidate_faces:
            face_gray = gray[y:y + h, x:x + w]
            face_bgr  = frame[y:y + h, x:x + w]

            eyes = detector.detect_eyes(face_gray)
            if len(eyes) < 1:
                continue

            skin_ok, _ = skin_validator.is_skin_region(face_bgr)
            if not skin_ok:
                continue

            cnn_ok, _ = cnn_verifier.verify_face_cnn(face_bgr)
            if not cnn_ok:
                continue

            draw_box(frame, x, y, w, h)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        cv2.imshow("Face Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
