import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import threading
from playsound import playsound
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ================= CONFIG =================
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75
CONFIRM_FRAMES = 3
AUDIO_DIR = "audio"
# =========================================

mp_hands = mp.solutions.hands

# ---------- LSTM MODEL ----------
class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(42, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# ---------- LOAD MODEL ----------
labels = np.load("labels.npy", allow_pickle=True)

model = LSTMModel(len(labels))
model.load_state_dict(torch.load("sign_lstm_torch.pth", map_location="cpu"))
model.eval()

# ---------- MEDIAPIPE ----------
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# ---------- STATE ----------
sequence = deque(maxlen=SEQUENCE_LENGTH)
current_candidate = None
candidate_count = 0
last_confirmed = None

# ---------- AUDIO (WAV FILES ONLY) ----------
def play_audio(word):
    """
    Plays pre-recorded WAV file for the detected word.
    Non-blocking.
    """
    path = os.path.join(AUDIO_DIR, f"{word}.wav")
    if os.path.exists(path):
        threading.Thread(
            target=playsound,
            args=(path,),
            daemon=True
        ).start()

def extract_hand(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x - base_x, lm.y - base_y])
    return data

# =========================================================
# BACKEND CORE (GUI owns the camera)
# =========================================================
def get_prediction(cap):
    global current_candidate, candidate_count, last_confirmed

    ret, frame = cap.read()
    if not ret:
        return None, None, None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        sequence.clear()
        current_candidate = None
        candidate_count = 0
        last_confirmed = None
        return None, None, frame

    hand = result.multi_hand_landmarks[0]
    sequence.append(extract_hand(hand))

    if len(sequence) < SEQUENCE_LENGTH:
        return None, None, frame

    x = torch.tensor([sequence], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    confidence = probs[idx].item()
    label = labels[idx]

    # ---------- CONFIDENCE FILTER ----------
    if confidence >= CONFIDENCE_THRESHOLD:
        if label == current_candidate:
            candidate_count += 1
        else:
            current_candidate = label
            candidate_count = 1

        if candidate_count >= CONFIRM_FRAMES and label != last_confirmed:
            last_confirmed = label
            play_audio(label)   # 🔊 PLAY WAV FILE
            return label, confidence, frame

    return None, None, frame

# =========================================================
# GUI APPLICATION
# =========================================================
def run_gui():
    cap = cv2.VideoCapture(0)

    root = tk.Tk()
    root.title("Indian Sign Language – Sentence Builder")
    root.geometry("1100x650")
    root.configure(bg="#0f172a")

    sentence = []

    # -------- HEADER --------
    header = tk.Label(
        root,
        text="Indian Sign Language – Sentence Builder",
        bg="#020617",
        fg="#38bdf8",
        font=("Segoe UI", 20, "bold")
    )
    header.pack(fill="x", pady=10)

    # -------- CAMERA --------
    cam_label = tk.Label(root, bg="#020617")
    cam_label.pack(pady=15)

    # -------- SENTENCE --------
    sentence_var = tk.StringVar()
    sent_label = tk.Label(
        root,
        textvariable=sentence_var,
        bg="#020617",
        fg="#e5e7eb",
        font=("Segoe UI", 18),
        wraplength=900
    )
    sent_label.pack(pady=10)

    # -------- BUTTONS --------
    btn_frame = tk.Frame(root, bg="#0f172a")
    btn_frame.pack(pady=10)

    ttk.Button(
        btn_frame,
        text="🗑 Clear",
        command=lambda: (sentence.clear(), sentence_var.set(""))
    ).pack(side="left", padx=20)

    # -------- UPDATE LOOP --------
    def update():
        label, conf, frame = get_prediction(cap)

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            cam_label.configure(image=img)
            cam_label.image = img

        if label:
            if not sentence or sentence[-1] != label:
                sentence.append(label)
                sentence_var.set(" ".join(sentence))

        root.after(30, update)

    def on_close():
        cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    update()
    root.mainloop()

# =========================================================
# ENTRY POINT (GUI ONLY)
# =========================================================
if __name__ == "__main__":
    run_gui()


