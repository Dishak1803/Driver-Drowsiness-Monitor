# Driver Drowsiness Detection System

A real-time driver drowsiness monitoring system using computer vision and deep learning. It tracks eye closure and yawning patterns via webcam to detect signs of fatigue and trigger an alert.

## Features
- Real-time face, eye, and mouth landmark detection (dlib)
- Eye-closure classification via a trained CNN model
- Yawn detection via a trained CNN model
- Live webcam feed with on-screen status overlay
- Audible alarm on detected drowsiness

## Project Structure

```
Driver_Drowsiness/
├── webcam.py                  # Main real-time detection script
├── MP.py                      # Merges raw Kaggle eye datasets into a unified open/closed folder for training
├── train_eye.py               # Trains the eye-state model
├── train_yawn.py              # Trains the yawn-detection model
├── eye_model.h5                # Pre-trained eye-state model
├── yawn_model.h5                # Pre-trained yawn-detection model
├── eye_training_plot.png      # Training accuracy/loss plot (eye model)
├── yawn_training_plot.png     # Training accuracy/loss plot (yawn model)
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10 (recommended — matches the environment this was built and tested on)
- A working webcam
- Windows users: no compiler needed — `dlib` is installed via a prebuilt wheel (see below)

### 1. Clone the repository

```bash
git clone https://github.com/Dishak1803/Driver-Drowsiness-Monitor.git
cd Driver-Drowsiness-Monitor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `dlib`:** If installing `dlib` fails with a CMake/build error (common on Windows without Visual C++ Build Tools), install the prebuilt wheel instead:
> ```bash
> pip install dlib-binary
> ```

### 4. Run the detector

```bash
python webcam.py
```

Press `q` to quit the live feed.

## Dataset & Training (Optional)

Pre-trained models (`eye_model.h5`, `yawn_model.h5`) are already included, so this step isn't required to run the project.

Trained on Kaggle eye-state datasets (open/closed eyes). To retrain from scratch:

1. Download an eye-state dataset from Kaggle (e.g. search "open closed eyes dataset")
2. Place the downloaded dataset folders inside a `data/` folder in the project root, named to match what `MP.py` expects:
   ```
   Driver_Drowsiness/
   └── data/
       ├── dataset2/
       └── dataset3/
   ```
3. Run `python MP.py` — this merges both datasets into `data/eyes_combined/open/` and `data/eyes_combined/closed/`
4. Run `python train_eye.py` (and `train_yawn.py` for the yawn dataset, sourced similarly)

The `data/` folder is excluded from version control via `.gitignore`, so raw datasets stay local and don't bloat the repo.

## Tech Stack
- Python, OpenCV, dlib
- TensorFlow / Keras
- imutils, NumPy
