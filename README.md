# Hand Gesture Detection System

A real-time American Sign Language (ASL) hand gesture detection system that recognizes letters A-Z using a webcam, builds complete sentences, and speaks them aloud using Text-to-Speech.

---

## Project Demo
```
Sign H → E → L → L → O → SPACE
Sign J → A → I → SPACE
Press ENTER → Speaker says "HELLO JAI"
```

---

## Team

| Name | Enrollment No |
|---|---|
| Tanvi Panchal | 12302110501052 |
| Tisha Patel | 12302110501054 |
| Kashish Ka Patel | 12302110501071 |

Guided By — Prof. Nikhil Gondaliya
G H Patel College of Engineering & Technology
Department of Computer Engineering
Semester 6 — A.Y. 2025-26

---

## Features

- Real-time ASL letter recognition A-Z using webcam
- 96.71% test accuracy using personal collected dataset
- Sentence builder — letters form words, words form sentences
- Audio output — pyttsx3 speaks full sentence aloud
- 20-frame prediction buffer for smooth stable predictions
- Cooldown timer prevents double letter detection
- Hold progress bar with visual feedback on screen
- Screenshot save with P key
- Works completely offline — no internet required
- No special hardware — works on any laptop webcam

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13 | Core language |
| MediaPipe | 0.10.33 | Hand landmark detection |
| TensorFlow | CPU | Neural network |
| OpenCV | 4.x | Webcam and UI |
| pyttsx3 | Latest | Text to speech |
| scikit-learn | Latest | Data processing |
| NumPy | Latest | Array operations |

---

## Project Structure
```
sign_language_system/
├── dataset/
│   └── landmarks.csv          ← collected gesture data
├── models/
│   ├── sign_model.h5          ← trained model
│   └── label_encoder.pkl      ← label encoder
├── hand_landmarker.task        ← MediaPipe model
├── collect_data.py             ← data collection script
├── train_model.py              ← model training script
├── main_app.py                 ← real-time application
└── README.md
```

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/hand-gesture-detection.git
cd hand-gesture-detection
```

### Step 2 — Install dependencies
```bash
pip install opencv-python mediapipe tensorflow-cpu numpy pyttsx3 scikit-learn tqdm
```

### Step 3 — Download MediaPipe model
The `hand_landmarker.task` file will be downloaded automatically when you run `collect_data.py` for the first time.

---

## How to Run

### Step 1 — Collect your own data
```bash
python collect_data.py
```
- Webcam opens showing live feed
- Make each ASL sign A-Z
- Press S to save each sample
- Press SPACE to skip a letter
- Press Q to quit
- Collects 50 samples per letter — 1300 total

### Step 2 — Train the model
```bash
python train_model.py
```
- Reads dataset/landmarks.csv
- Trains Dense Neural Network
- Saves model to models/sign_model.h5
- Expected accuracy — 95%+

### Step 3 — Run real-time app
```bash
python main_app.py
```

---

## Controls

| Key | Action |
|---|---|
| Hold sign steady | Letter gets predicted |
| SPACE | Save current word to sentence |
| ENTER | Speak full sentence aloud |
| C | Clear everything |
| P | Save screenshot |
| Q | Quit application |

---

## Model Architecture
```
Input Layer      →  63 features (21 landmarks × 3 coordinates)
Dense Layer 1    →  512 neurons, ReLU, BatchNorm, Dropout 0.4
Dense Layer 2    →  256 neurons, ReLU, BatchNorm, Dropout 0.3
Dense Layer 3    →  128 neurons, ReLU, BatchNorm, Dropout 0.2
Dense Layer 4    →  64 neurons, ReLU
Output Layer     →  26 neurons, Softmax (A-Z)
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 96.71% |
| Test Loss | 0.0810 |
| Total Classes | 26 |
| Training Samples | 1300 |
| Training Epochs | 32 |

---

## System Flow
```
Webcam Input
     ↓
MediaPipe — 21 hand landmarks
     ↓
Normalize 63 features
     ↓
Dense Neural Network — predict letter
     ↓
20-frame buffer — majority vote
     ↓
Cooldown timer — 1.5 seconds
     ↓
Letter accepted — add to word
     ↓
SPACE — save word to sentence
     ↓
ENTER — pyttsx3 speaks sentence
```

---

## Dataset

- Collected personally using webcam
- 26 classes — A to Z
- 50 samples per letter
- 1300 total samples
- 63 features per sample
- Saved as dataset/landmarks.csv

---

## Why Personal Dataset?

| Public Dataset | Personal Dataset |
|---|---|
| Studio lighting | Room lighting |
| Different hand | Our hand |
| 70-80% accuracy | 96.71% accuracy |

---

## Challenges Faced

- MediaPipe solutions API not supported on Python 3.13 — switched to Tasks API
- TensorFlow install failed — used tensorflow-cpu
- Low accuracy with Kaggle dataset — collected personal data
- Double letter detection — added cooldown timer
- Flickering predictions — added 20-frame buffer
- I and Y confusion — collected extra samples

---

## Future Scope

- Add word level signs — HELLO, THANKS, NAMASTE
- Support dynamic signs J and Z with motion tracking
- Build Android or iOS mobile application
- Add Indian Sign Language support
- Real-time video call subtitle integration
- Upgrade to LSTM sequential model
- Add regional language translation

---

## Requirements
```
opencv-python
mediapipe==0.10.33
tensorflow-cpu
numpy
pyttsx3
scikit-learn
tqdm
pandas
```

Save as `requirements.txt` and install with:
```bash
pip install -r requirements.txt
```

---

## License

This project is made for educational purposes as a mini project submission for Semester 6, Computer Engineering, G H Patel College of Engineering & Technology.

---

## Acknowledgements

- MediaPipe by Google — hand landmark detection
- TensorFlow and Keras — neural network framework
- OpenCV — computer vision library
- pyttsx3 — text to speech engine
- ASL alphabet reference — lifeprint.com

---

JAI SHREE KRISHNA
```

---

## How to put this on GitHub

**Step 1** — Create new repository on github.com named `hand-gesture-detection`

**Step 2** — Create `README.md` file in your `sign_language_system` folder and paste all the above content

**Step 3** — Also create `requirements.txt` with this content:
```
opencv-python
mediapipe==0.10.33
tensorflow-cpu
numpy
pyttsx3
scikit-learn
tqdm
pandas
