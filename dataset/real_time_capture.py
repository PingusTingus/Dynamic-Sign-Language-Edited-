import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# Load the Trained Model
model = tf.keras.models.load_model("dataset/gesture_lstm_model.h5")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture Labels
gesture_labels = {0: "hello", 1: "help", 2: "i'm sick"}

# **Create a Buffer to Store 30 Frames**
frame_buffer = deque(maxlen=30)  # Stores up to 30 frames

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Process Hand Landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # Extract x, y, z values

            landmarks = np.array(landmarks).flatten()  # Flatten to shape (63,)
            frame_buffer.append(landmarks)  # Store in buffer

            # **Only Predict When We Have 30 Frames**
            if len(frame_buffer) == 30:
                input_sequence = np.array(frame_buffer).reshape(1, 30, 63)  # Reshape for LSTM
                prediction = model.predict(input_sequence)
                predicted_label = np.argmax(prediction)

                # Display Gesture
                gesture_text = f"Gesture: {gesture_labels[predicted_label]}"
                cv2.putText(frame, gesture_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Draw Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
