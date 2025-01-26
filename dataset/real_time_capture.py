import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the trained LSTM model
model = tf.keras.models.load_model("gesture_lstm_model.h5")

# Load gesture labels
gesture_labels = ["Good Morning", "Good Afternoon", "Good Evening"]  # Update with your gestures

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Store last 30 frames for prediction
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            for lm in hand_landmark.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])

    # Ensure 21 keypoints x 3 (x, y, z) = 63 features
    if len(hand_landmarks) == 63:
        sequence.append(hand_landmarks)
        if len(sequence) > 30:  # Keep only the last 30 frames
            sequence.pop(0)

    # If we have 30 frames, make a prediction
    if len(sequence) == 30:
        input_data = np.expand_dims(sequence, axis=0)  # Reshape to (1, 30, 63)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Display the predicted gesture
        text = f"Gesture: {gesture_labels[predicted_class]}"
        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the webcam feed
    cv2.imshow("Real-Time Gesture Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_classes))

cap.release()
cv2.destroyAllWindows()
