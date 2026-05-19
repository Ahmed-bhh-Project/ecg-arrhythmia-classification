# 🫀 ECG Arrhythmia Classification

> A deep learning system that detects **8 types of cardiac arrhythmias** from ECG signals and images, powered by a custom CNN and deployed via a Flask web interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Detected Arrhythmia Classes](#detected-arrhythmia-classes)
- [Project Architecture](#project-architecture)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Technologies Used](#technologies-used)
- [Disclaimer](#disclaimer)

---

## Overview

This project implements an end-to-end cardiac arrhythmia detection system using a Convolutional Neural Network (CNN) trained on ECG data from the **MIT-BIH Arrhythmia Database**. The system processes raw ECG signals (CSV format) or ECG images, segments individual heartbeats, classifies each beat into one of 8 categories, and presents the results through a clean Flask web interface.

Key capabilities:

- **Signal mode**: Upload a raw ECG signal as a CSV file. The system automatically segments each heartbeat, converts it to an image, and classifies it.
- **Image mode**: Upload a single ECG beat image for direct classification with a confidence score.
- **Beat viewer**: Browse all detected abnormal beats one by one, with class labels and estimated heart rate (BPM).

---

## Detected Arrhythmia Classes

| Abbreviation | Full Name |
|---|---|
| `NOR` | Normal Sinus Rhythm |
| `APC` | Atrial Premature Contraction |
| `LBBB` | Left Bundle Branch Block |
| `PAB` | Premature Atrial Beat |
| `PVC` | Premature Ventricular Contraction |
| `RBBB` | Right Bundle Branch Block |
| `VEB` | Ventricular Ectopic Beat |
| `VFE` | Ventricular Fibrillation / Flutter |

---

## Project Architecture

```
┌───────────────────────────────────────────────┐
│              Flask Web Application             │
│                                               │
│  /about    → Project description page         │
│  /upload   → Single ECG image prediction      │
│  /upload_csv → Full signal analysis (CSV)     │
│  /next/<n> → Beat-by-beat viewer              │
│  /info     → Arrhythmia information page      │
└───────────────────────┬───────────────────────┘
                        │
           ┌────────────▼────────────┐
           │   preprocessing.py      │
           │  Signal normalization   │
           │  R-peak detection       │
           │  Beat segmentation      │
           │  Image generation       │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │       model.py          │
           │   ECGCNN (5-layer CNN)  │
           │   8-class classifier    │
           │   ECGCNN_model.pt       │
           └─────────────────────────┘
```

---

## Model Architecture

The `ECGCNN` model is a custom 5-block Convolutional Neural Network built with PyTorch.

```
Input: Grayscale ECG image (1 × 128 × 128)
   │
   ├── Conv Block 1:  Conv2d(1→32)   + BatchNorm + ReLU + MaxPool → 32×64×64
   ├── Conv Block 2:  Conv2d(32→64)  + BatchNorm + ReLU + MaxPool → 64×32×32
   ├── Conv Block 3:  Conv2d(64→128) + BatchNorm + ReLU + MaxPool → 128×16×16
   ├── Conv Block 4:  Conv2d(128→256)+ BatchNorm + ReLU + MaxPool → 256×8×8
   ├── Conv Block 5:  Conv2d(256→512)+ BatchNorm + ReLU + MaxPool → 512×4×4
   │
   ├── Flatten → 8192
   ├── FC1: 8192 → 1024  + ReLU + Dropout(0.5)
   ├── FC2: 1024 → 256   + ReLU + Dropout(0.5)
   └── Output: 256 → 8 classes
```

---

## File Structure

```
ecg-arrhythmia-classification/
│
├── app (1).py                  # Main Flask application
├── model.py                    # CNN model definition (ECGCNN)
├── preprocessing.py            # Signal processing & beat segmentation
├── split_train_test_dataset.py # Dataset preparation script
├── ECGCNN_model.pt             # Trained model weights (PyTorch)
│
├── templates/                  # HTML templates (Jinja2)
│   ├── about.html
│   ├── index.html
│   ├── info.html
│   ├── upload_csv.html
│   └── viewer.html
│
├── static/                     # CSS, JS, and generated beat images
│   └── beats/                  # Saved abnormal beat images (runtime)
│
└── uploads/                    # Uploaded files (runtime)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Ahmed-bhh-Project/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

2. **Install dependencies**

```bash
pip install flask torch torchvision pillow numpy scipy scikit-learn matplotlib
```

3. **Run the application**

```bash
python "app (1).py"
```

4. **Open your browser** and navigate to `http://127.0.0.1:5000`

---

## Usage

### Option 1 — Analyze a full ECG signal (CSV)

1. Go to `/upload_csv`
2. Upload a `.csv` file containing a single-channel ECG signal (one sample per line)
3. The system will:
   - Normalize the signal
   - Detect R-peaks automatically
   - Segment each heartbeat into a 192-sample window
   - Convert each beat to a 128×128 grayscale image
   - Classify each beat with the CNN
   - Display all **abnormal beats** one by one with the estimated BPM and a summary of detected arrhythmia types

### Option 2 — Classify a single ECG beat image

1. Go to `/upload`
2. Upload a `.png` / `.jpg` image of a single ECG beat
3. The system returns:
   - The predicted arrhythmia class (abbreviation + full name)
   - The confidence probability (%)

---

## Pipeline

```
Raw ECG Signal (CSV)
        │
        ▼
  Normalization (sklearn.preprocessing.scale)
        │
        ▼
  R-Peak Detection (scipy.signal.find_peaks)
        │
        ▼
  Beat Segmentation (±96 samples around each R-peak)
        │
        ▼
  Quality Filtering (amplitude, std deviation, length checks)
        │
        ▼
  Beat → PNG Image (matplotlib, 128×128 grayscale)
        │
        ▼
  CNN Inference (ECGCNN → softmax → argmax)
        │
        ▼
  Classification Result + Confidence Score
```

---

## Technologies Used

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Signal Processing | SciPy, NumPy, scikit-learn |
| Image Processing | Pillow (PIL), Matplotlib |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript (Jinja2 templates) |
| Dataset | MIT-BIH Arrhythmia Database |

---

## Disclaimer

> ⚠️ **This project is for educational and research purposes only.** It is not a certified medical device and must not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for any cardiac concerns.

---

*Built with PyTorch and Flask — MIT-BIH Arrhythmia Database*
