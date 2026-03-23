import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import pyttsx3
import pickle
import threading
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from collections import deque

# ── Load models ────────────────────────────────────────────────────────────
print("Loading models...")
letter_model = load_model("models/sign_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load gesture model if exists
gesture_model_exists = os.path.exists("models/gesture_model.h5")
if gesture_model_exists:
    gesture_model = load_model("models/gesture_model.h5")
    with open("models/gesture_encoder.pkl", "rb") as f:
        ge = pickle.load(f)
    print(f"Gesture model loaded! Gestures: {list(ge.classes_)}")

print(f"Letter model loaded! Classes: {list(le.classes_)}")

# ── MediaPipe setup ────────────────────────────────────────────────────────
MODEL_PATH   = "hand_landmarker.task"
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# ── TTS setup ─────────────────────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 145)
tts_engine.setProperty('volume', 1.0)

def speak(text):
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── State variables ────────────────────────────────────────────────────────
prediction_buffer  = deque(maxlen=20)
sentence_words     = []
current_word       = []
last_sign          = ""
sign_hold_count    = 0
HOLD_THRESHOLD     = 15
CONFIDENCE_MIN     = 0.65
last_accepted_time = 0
COOLDOWN_SECONDS   = 1.5
MODE               = "LETTER"   # LETTER or GESTURE
flash_timer        = 0
last_accepted_sign = ""

# ── Colors ─────────────────────────────────────────────────────────────────
NAVY   = (80,  40,  0)
TEAL   = (200, 200, 0)
GREEN  = (100, 220, 100)
WHITE  = (255, 255, 255)
GRAY   = (150, 150, 150)
DARK   = (30,  30,  30)
BLUE   = (200, 150, 50)
GOLD   = (0,   180, 220)
RED    = (80,  80,  220)

# ── Helper functions ───────────────────────────────────────────────────────
def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        wx = lm[0].x
        wy = lm[0].y
        row = []
        for pt in lm:
            row.extend([pt.x - wx, pt.y - wy, pt.z])
        return np.array(row, dtype=np.float32), result.hand_landmarks[0]
    return None, None

def get_letter_prediction(features):
    probs = letter_model.predict(features.reshape(1, -1), verbose=0)[0]
    idx   = np.argmax(probs)
    return le.classes_[idx], probs[idx]

def get_gesture_prediction(features):
    if not gesture_model_exists:
        return "No gesture model", 0.0
    probs = gesture_model.predict(features.reshape(1, -1), verbose=0)[0]
    idx   = np.argmax(probs)
    return ge.classes_[idx], probs[idx]

def draw_landmarks_manual(frame, landmarks, w, h):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 200, 255), -1)
        cv2.circle(frame, point, 5, (255, 255, 255), 1)

def draw_rounded_rect(frame, x1, y1, x2, y2, color, alpha=0.7, radius=12):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def build_display_sentence():
    word = "".join(current_word)
    sent = " ".join(sentence_words)
    full = (sent + " " + word).strip()
    return full if full else "..."

# ── Main loop ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "="*55)
print("  Hand Gesture Detection System")
print("  Developed by: Tanvi, Tisha, Kashish")
print("  Guided by: Prof. Nikhil Gondaliya")
print("  GHPCE&T — Computer Engineering")
print("="*55)
print("\nControls:")
print("  SPACE = finish word")
print("  ENTER = speak sentence")
print("  G     = toggle Gesture/Letter mode")
print("  C     = clear all")
print("  P     = screenshot")
print("  Q     = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame         = cv2.flip(frame, 1)
    h, w          = frame.shape[:2]
    current_sign  = ""
    confidence    = 0.0
    sign_accepted = False
    current_time  = time.time()

    features, landmarks = extract_landmarks(frame)

    if features is not None:
        draw_landmarks_manual(frame, landmarks, w, h)

        if MODE == "LETTER":
            current_sign, confidence = get_letter_prediction(features)
        else:
            current_sign, confidence = get_gesture_prediction(features)

        prediction_buffer.append(current_sign)
        stable_sign = max(set(prediction_buffer),
                          key=prediction_buffer.count)

        if confidence >= CONFIDENCE_MIN:
            if stable_sign == last_sign:
                sign_hold_count += 1
            else:
                sign_hold_count = 0
                last_sign       = stable_sign

            if sign_hold_count == HOLD_THRESHOLD:
                time_since = current_time - last_accepted_time
                if time_since >= COOLDOWN_SECONDS:
                    if MODE == "LETTER":
                        current_word.append(stable_sign)
                    else:
                        if current_word:
                            sentence_words.append("".join(current_word))
                            current_word = []
                        sentence_words.append(stable_sign)
                        speak(stable_sign)
                    sign_accepted      = True
                    last_accepted_time = current_time
                    last_accepted_sign = stable_sign
                    sign_hold_count    = 0
                    last_sign          = ""
                    flash_timer        = 8
                    print(f"Accepted: {stable_sign}")

    if flash_timer > 0:
        flash_timer -= 1

    # ── DRAW UI ────────────────────────────────────────────────────────────

    # Top header bar
    draw_rounded_rect(frame, 0, 0, w, 95, (25, 25, 25), alpha=0.85, radius=0)

    # Mode badge
    mode_color = (180, 100, 20) if MODE == "LETTER" else (20, 140, 80)
    mode_text  = "LETTER MODE" if MODE == "LETTER" else "GESTURE MODE"
    draw_rounded_rect(frame, 15, 10, 190, 42, mode_color, alpha=0.9, radius=8)
    cv2.putText(frame, mode_text, (25, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)

    # Current sign prediction
    sign_color = (100, 255, 180) if confidence >= CONFIDENCE_MIN else (120, 120, 120)
    sign_text  = f"{current_sign}" if current_sign else "---"
    cv2.putText(frame, sign_text, (210, 58),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, sign_color, 3)

    # Confidence percentage
    conf_text = f"{confidence*100:.0f}%"
    cv2.putText(frame, conf_text, (330, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 220, 100), 2)

    # Hold progress bar
    bar_x  = 480
    bar_w  = 380
    bar_h  = 18
    bar_y  = 38
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    if sign_hold_count > 0:
        fill = int((sign_hold_count / HOLD_THRESHOLD) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h),
                      (100, 220, 100), -1)
    cv2.putText(frame, "HOLD", (bar_x + bar_w + 10, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Cooldown bar
    time_since = current_time - last_accepted_time
    if time_since < COOLDOWN_SECONDS:
        cool_fill = int((time_since / COOLDOWN_SECONDS) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + cool_fill, bar_y + bar_h),
                      (50, 150, 220), -1)
        cv2.putText(frame, "WAIT", (bar_x + bar_w + 10, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 150, 220), 1)

    # Current word — top right
    word_now = "".join(current_word)
    draw_rounded_rect(frame, w-310, 8, w-10, 48,
                      (40, 60, 40), alpha=0.8, radius=8)
    cv2.putText(frame, f"Word: {word_now}",
                (w-300, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (100, 255, 100), 2)

    # Last accepted sign flash
    if flash_timer > 0:
        cv2.putText(frame, f"+ {last_accepted_sign}",
                    (w//2 - 40, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5,
                    (100, 255, 100), 4)

    # Green border flash on acceptance
    if sign_accepted:
        cv2.rectangle(frame, (0, 0), (w, h), (100, 255, 100), 6)

    # Bottom sentence panel
    draw_rounded_rect(frame, 0, h-145, w, h,
                      (15, 15, 40), alpha=0.9, radius=0)

    cv2.putText(frame, "Sentence:", (20, h-115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (150, 150, 220), 1)

    sentence_display = build_display_sentence()
    # Truncate if too long
    if len(sentence_display) > 45:
        sentence_display = "..." + sentence_display[-42:]
    cv2.putText(frame, sentence_display, (20, h-70),
                cv2.FONT_HERSHEY_DUPLEX, 1.4,
                (255, 255, 255), 2)

    # Controls bar
    cv2.putText(frame,
                "SPACE=word  ENTER=speak  G=mode  C=clear  P=screenshot  Q=quit",
                (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)

    # Project name watermark bottom right
    cv2.putText(frame, "Hand Gesture Detection System",
                (w-380, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (80, 80, 80), 1)

    cv2.imshow("Hand Gesture Detection System", frame)

    # ── Key controls ───────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == 32:        # SPACE
        if current_word:
            word = "".join(current_word)
            sentence_words.append(word)
            print(f"Word saved: {word}")
            current_word = []

    elif key == 13:        # ENTER
        final = " ".join(sentence_words)
        if current_word:
            final += " " + "".join(current_word)
        if final.strip():
            print(f"Speaking: {final.strip()}")
            speak(final.strip())

    elif key == ord('g'):  # G — toggle mode
        MODE = "GESTURE" if MODE == "LETTER" else "LETTER"
        prediction_buffer.clear()
        last_sign       = ""
        sign_hold_count = 0
        print(f"Mode switched to: {MODE}")

    elif key == ord('c'):  # CLEAR
        sentence_words.clear()
        current_word.clear()
        prediction_buffer.clear()
        last_sign       = ""
        sign_hold_count = 0
        print("Cleared!")

    elif key == ord('p'):  # SCREENSHOT
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("App closed.")