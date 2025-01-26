import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  # For drawing landmarks

# Gesture Labels (Manually Define)
gesture_labels = {0: "good morning", 1: "good afternoon", 2: "good evening"}

# Ask user for a label
gesture_id = int(input(f"Enter gesture label {gesture_labels}: "))

# Open Webcam
cap = cv2.VideoCapture(0)

# Data Collection Variables
frames = []
num_frames = 30  # Collect 30 frames per gesture

while len(frames) < num_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Extract landmarks if a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # x, y, z coordinates
            frames.append(np.array(landmarks).flatten())  # Flatten to 63 values

    cv2.putText(frame, f"Recording Gesture: {gesture_labels[gesture_id]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recording Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop early
        break

cap.release()
cv2.destroyAllWindows()

# Convert to NumPy Array
frames = np.array(frames)

# Save Features and Labels
np.save(f"dataset/gesture_{gesture_labels[gesture_id]}.npy", frames)
np.save(f"dataset/gesture_labels.npy", np.array([gesture_id] * len(frames)))

print(f"✅ Gesture '{gesture_labels[gesture_id]}' recorded and saved!")
