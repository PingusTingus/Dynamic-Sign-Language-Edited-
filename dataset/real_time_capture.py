import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque, Counter

# Load Trained Model
model = tf.keras.models.load_model("gesture_model.h5")

# Define Gesture Labels
gesture_labels = ["danger", "hello", "help", "I", "sick", "stop"]

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open Camera
cap = cv2.VideoCapture(0)

# Frame Buffer (Stores the last 30 frames)
frame_buffer = deque(maxlen=30)  # Stores exactly 30 frames

# Majority Voting System (Queue)
buffer_size = 10  # How many predictions to store for voting
predictions_queue = deque(maxlen=buffer_size)

# Delay Settings
last_prediction_time = time.time()
prediction_delay = 1.0  # seconds (delay before making a new prediction)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 Hand Landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Append to Frame Buffer
            frame_buffer.append(landmarks)

            # Ensure we have exactly 30 frames before making a prediction
            if len(frame_buffer) == 30:
                sequence = np.array(frame_buffer).reshape(1, 30, 63)  # Reshape for LSTM

                # Delay Before Making a New Prediction
                if time.time() - last_prediction_time > prediction_delay:
                    prediction = model.predict(sequence)
                    predicted_label = np.argmax(prediction)

                    # Store prediction in queue for majority voting
                    predictions_queue.append(predicted_label)

                    last_prediction_time = time.time()

    # Apply Majority Voting
    if len(predictions_queue) > 0:
        most_common_label = Counter(predictions_queue).most_common(1)[0][0]
        predicted_text = gesture_labels[most_common_label]
        cv2.putText(frame, f"Gesture: {predicted_text}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display Camera Feed
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
