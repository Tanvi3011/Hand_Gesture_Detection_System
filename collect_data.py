import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_PATH   = "hand_landmarker.task"
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

OUTPUT_CSV = "dataset/landmarks.csv"
os.makedirs("dataset", exist_ok=True)

MY_LETTERS = ['W', 'X', 'Y', 'Z']
SAMPLES    = 50

file_exists = os.path.exists(OUTPUT_CSV)
cap = cv2.VideoCapture(0)

print("\nINSTRUCTIONS:")
print("  S     = save sample")
print("  SPACE = skip to next letter")
print("  Q     = quit\n")

with open(OUTPUT_CSV, "a", newline="") as f:

    for letter in MY_LETTERS:
        count = 0
        print(f"\nSign the letter: {letter}")
        print(f"Press S to save {SAMPLES} samples")

        while count < SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )
            result  = detector.detect(mp_image)
            hand_ok = False
            row     = []

            if result.hand_landmarks:
                lm  = result.hand_landmarks[0]
                wx  = lm[0].x
                wy  = lm[0].y
                hand_ok = True
                for pt in lm:
                    row.extend([
                        round(pt.x - wx, 6),
                        round(pt.y - wy, 6),
                        round(pt.z,      6)
                    ])
                points = [(int(p.x*w), int(p.y*h)) for p in lm]
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17)
                ]
                for start, end in connections:
                    cv2.line(frame, points[start],
                             points[end], (255,255,255), 2)
                for point in points:
                    cv2.circle(frame, point, 5, (0,200,255), -1)

            # Top bar
            cv2.rectangle(frame, (0,0), (w,90), (30,30,30), -1)

            # Letter display
            cv2.putText(frame, f"Sign: {letter}",
                        (20, 50),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.5, (0,255,180), 2)

            # Progress bar
            bar = int((count/SAMPLES)*400)
            cv2.rectangle(frame, (20,65),(420,78),(60,60,60),-1)
            cv2.rectangle(frame, (20,65),(20+bar,78),(0,255,100),-1)
            cv2.putText(frame, f"{count}/{SAMPLES}",
                        (430, 77),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200,200,200), 1)

            # Hand status
            hand_color = (0,255,100) if hand_ok else (0,0,255)
            hand_text  = "Hand detected" if hand_ok else "No hand detected"
            cv2.putText(frame, hand_text,
                        (w-250, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, hand_color, 2)

            # Bottom bar
            cv2.rectangle(frame, (0,h-50),(w,h),(20,20,50),-1)
            cv2.putText(frame,
                        "S=save sample   SPACE=skip letter   Q=quit",
                        (20, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (150,150,220), 1)

            cv2.imshow("Collecting Sign Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and hand_ok:
                with open(OUTPUT_CSV, "a", newline="") as f2:
                    writer2 = csv.writer(f2)
                    writer2.writerow(row + [letter])
                count += 1
                print(f"  Saved {count}/{SAMPLES} for {letter}")

            elif key == 32:
                print(f"  Skipped {letter} with {count} samples")
                break

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                print("\nCollection stopped.")
                exit()

        print(f"Done {letter}: {count} samples saved")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\nAll letters done!")
print("Now run: python train_model.py")