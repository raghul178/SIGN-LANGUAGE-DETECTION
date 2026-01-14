import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

DATA_DIR = "data"
SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 100

def extract_hand(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x - base_x, lm.y - base_y])
    return data

gesture = input("Enter gesture label: ").strip()
save_path = os.path.join(DATA_DIR, gesture)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
sequence = []
sample_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
) as hands:

    print("Perform the gesture...")

    while sample_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            sequence.append(extract_hand(hand))

            if len(sequence) == SEQUENCE_LENGTH:
                np.save(
                    os.path.join(save_path, f"{sample_count}.npy"),
                    np.array(sequence, dtype=np.float32)
                )
                sample_count += 1
                sequence = []

        cv2.putText(
            frame,
            f"{gesture}: {sample_count}/{SAMPLES_PER_GESTURE}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collect LSTM Data", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
