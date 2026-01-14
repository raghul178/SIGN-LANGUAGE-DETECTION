# 🤟 SIGN LANGUAGE DETECTION

## 📌 Project Overview

**SIGN LANGUAGE DETECTION** is a real-time **Indian Sign Language (ISL)** recognition system developed during a **Hackathon at Sathyabama Institute of Science and Technology**. The project uses **computer vision** and **deep learning (LSTM)** to recognize hand gestures and convert them into meaningful words, which are then played using pre-recorded audio.

The system is designed to assist **speech- and hearing-impaired individuals** by enabling natural, camera-based communication.

📚 **Reference for gestures**: Official Indian Sign Language Dictionary
🔗 [https://indiansignlanguage.org/](https://indiansignlanguage.org/)

---

## 👥 Team Information

* **Team Size**: 6 members
* **Event**: Hackathon
* **Venue**: Sathyabama Institute of Science and Technology

---

## ✋ Supported Vocabulary (Current)

The model can currently recognize the following **9 ISL words**:

1. I
2. WANT
3. WATER
4. HI
5. YOU
6. OK
7. LOVE
8. THANK YOU
9. HELP

Each word is trained using real ISL gestures referenced from the official ISL website.

---

## 🧠 Technical Architecture

### 🔹 Hand Tracking

* **MediaPipe Hands** is used to extract **21 hand landmarks** from a live camera feed.
* Only **single-hand recognition** is used for stability and speed.

### 🔹 Feature Representation

* Each landmark is normalized relative to the wrist.
* A single frame → **42 features (x, y for 21 points)**
* A gesture → **30-frame sequence**

### 🔹 Deep Learning Model

* **LSTM (Long Short-Term Memory)** network implemented using **PyTorch**
* Sequence-based recognition avoids flickering and improves temporal accuracy

### 🔹 Audio Output

* Uses **pre-recorded `.wav` files** for each word
* Ensures fast, reliable, and non-blocking speech output

---

## 📂 Project Structure

```
SIGN LANGUAGE DETECTION/
│
├── recogonise_model.py        # GUI-only application (camera + prediction + audio)
├── collect_data.py            # Data collection script (ISL gestures)
├── train_model.py             # LSTM model training script
├── sign_lstm_torch.pth        # Trained PyTorch LSTM model
├── labels.npy                 # Model label mapping
│
├── data/                      # Collected gesture datasets
│   ├── I/
│   ├── WANT/
│   ├── WATER/
│   ├── HI/
│   ├── YOU/
│   ├── OK/
│   ├── LOVE/
│   ├── THANK YOU/
│   └── HELP/
│
├── audio/                     # Pre-generated audio files
│   ├── I.wav
│   ├── WANT.wav
│   ├── WATER.wav
│   ├── HI.wav
│   ├── YOU.wav
│   ├── OK.wav
│   ├── LOVE.wav
│   ├── THANK YOU.wav
│   └── HELP.wav
│
└── README.md
```
NOTE:the audio and data folders are not added to the repository as to support the creators viability.
---

## 📥 Data Collection (`collect_data.py`)

### Purpose

Used to **record gesture samples** for each ISL word.

### How it Works

* Opens webcam
* Detects hand landmarks using MediaPipe
* Collects **30-frame sequences**
* Saves each sample as `.npy` files under `data/<WORD>/`

### Usage

```bash
python collect_data.py
```

You will be prompted to enter the gesture label (e.g., `I`, `WANT`).

---

## 🏋️ Model Training (`train_model.py`)

### Purpose

Trains the **LSTM-based gesture recognition model**.

### Key Details

* Framework: **PyTorch**
* Input shape: `(30 frames × 42 features)`
* Output: Gesture class probabilities

### Output Files

* `sign_lstm_torch.pth` → trained model
* `labels.npy` → label index mapping

### Usage

```bash
python train_model.py
```

⚠️ Must be rerun whenever:

* New words are added
* Existing datasets are modified

---

## 🖥️ Real-Time GUI Application (`recogonise_model.py`)

### Features

* Live camera feed
* Real-time ISL recognition
* Confidence-based filtering
* Sentence builder
* Automatic audio playback using `.wav` files

### Run the Application

```bash
python recogonise_model.py
```

---

## 🧪 Environment & Requirements

### Python Version

* **Python 3.10 – 3.12 (Recommended: 3.10 or 3.11)**

### Required Libraries

Install all dependencies using:

```bash
pip install torch torchvision torchaudio
pip install opencv-python mediapipe numpy
pip install playsound==1.2.2 pillow
```

⚠️ `playsound==1.2.2` is required for stable audio playback.

---

## 📸 Hardware Requirements

* Webcam (720p or higher recommended)
* CPU-based inference (GPU optional)

---

## 🎯 Key Highlights

* ISL-compliant gestures
* Sequence-based LSTM recognition
* Flicker-free predictions
* No cloud / internet dependency
* Assistive-technology focused design

---

## 🚀 Future Enhancements

* Sentence grammar correction
* Two-hand gesture recognition
* Expanded ISL vocabulary
* Mobile / web deployment
* Accessibility modes (large text, high contrast)

---

## 📜 Acknowledgements

* **Indian Sign Language Dictionary**: (https://indiansignlanguage.org/)
* MediaPipe
* PyTorch
* OpenCV
* Hackathon organizers at Sathyabama Institute of Science and Technology

---

## 🤝 Contribution

This project was developed collaboratively by a **6-member team** during a hackathon. Contributions, suggestions, and improvements are welcome.
Member-1:RAGHUL P
Member-2:Pradeepkandhan P K
Member-3:Pranav T
Member-4:Praveen Raj E
Member-5:Nithyasri B K
Member-6:Pratika Prashantt Kulkarni
---

## 📄 License

This project is intended for **educational and research purposes**.

---

> ✨ *Bridging communication gaps using AI and Indian Sign Language*

