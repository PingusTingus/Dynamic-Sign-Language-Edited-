"""
Filipino Sign Language - Real-Time Recognition
Uses trained model to recognize gestures from webcam
Compatible with Keras 3 and TensorFlow SavedModel format
Author: PingusTingus
Date: 2025-03-03 06:55:02
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque
import json

# --- Configuration ---
MODEL_PATH = "models/fsl_model.h5"  # Path to trained model
METADATA_PATH = "models/evaluation_results.json"  # Path to model metadata
DETECTION_THRESHOLD = 0.7  # Confidence threshold for detection
PREDICTION_FRAMES = 50  # Number of frames to collect before prediction
PREDICTION_SMOOTHING = 5  # Number of predictions to average
WINDOW_NAME = "Filipino Sign Language Recognition"

print("=" * 70)
print("💪 Filipino Sign Language Real-Time Recognition")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: 2025-03-03 06:55:02")
print("=" * 70)

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # Use lightest model for realtime
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

def load_model_and_metadata():
    """Load the trained model and metadata"""
    print("Loading model and metadata...")

    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model path {MODEL_PATH} does not exist.")
            return None, None

        # Handle different model formats for Keras 3 compatibility
        try:
            # First try the standard loading method
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Standard loading failed: {e}")
            print("Trying alternative loading methods...")

            # Try as a TensorFlow SavedModel
            try:
                model = tf.saved_model.load(MODEL_PATH)
                print("Loaded as TensorFlow SavedModel")
            except Exception as e2:
                print(f"SavedModel loading failed: {e2}")

                # Try as a TensorFlow SavedModel layer
                try:
                    # Create a wrapper for the SavedModel
                    class ModelWrapper:
                        def __init__(self, path):
                            self.model = tf.saved_model.load(path)
                            # Find the inference function (usually 'serving_default')
                            self.call_endpoint = list(self.model.signatures.keys())[0]

                        def predict(self, x, verbose=0):
                            # Convert to TensorFlow tensor if needed
                            if not isinstance(x, tf.Tensor):
                                x = tf.convert_to_tensor(x, dtype=tf.float32)
                            # Call the model with the appropriate signature
                            output = self.model.signatures[self.call_endpoint](tf.constant(x))
                            # Get the output tensor (usually named 'dense' or similar)
                            output_name = list(output.keys())[0]
                            return output[output_name].numpy()

                    # Create the wrapper
                    model = ModelWrapper(MODEL_PATH)
                    print("Model loaded via custom wrapper for TensorFlow SavedModel")
                except Exception as e3:
                    print(f"Custom wrapper loading failed: {e3}")
                    return None, None

        print(f"✅ Successfully loaded model from {MODEL_PATH}")

        # Try to load the metadata
        try:
            # Load metadata from file
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ Loaded metadata from {METADATA_PATH}")
            else:
                # Create default metadata if file doesn't exist
                print(f"⚠️ Metadata file {METADATA_PATH} not found.")

                # Look for any .json files in the models directory
                json_files = [f for f in os.listdir("models") if f.endswith('.json')]
                if json_files:
                    print(f"Found alternative metadata file: {json_files[0]}")
                    with open(os.path.join("models", json_files[0]), 'r') as f:
                        metadata = json.load(f)
                    print(f"✅ Loaded metadata from {json_files[0]}")
                else:
                    # Create minimal metadata
                    print("Creating minimal metadata...")
                    metadata = {
                        'gestures': ["Class0", "Class1", "Class2", "Class3", "Class4"],
                        'model_type': 'unknown'
                    }
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
            metadata = {
                'gestures': ["Class0", "Class1", "Class2", "Class3", "Class4"],
                'model_type': 'unknown'
            }

        return model, metadata
    except Exception as e:
        print(f"❌ Error in model loading process: {e}")
        return None, None

def extract_hand_landmarks(frame):
    """Extract and normalize hand landmarks from a frame"""
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, frame

    # Draw landmarks on frame
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract and normalize landmarks
        wrist = hand_landmarks.landmark[0]
        landmarks = np.array([
            [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
            for lm in hand_landmarks.landmark
        ], dtype=np.float32).flatten()

        return landmarks, frame

    return None, frame

def predict_gesture(model, landmarks_sequence, metadata):
    """Predict gesture from a sequence of landmarks"""
    # Make sure we have the right shape
    sequence = np.array(landmarks_sequence)

    # Reshape for model input [batch, timestamps, features]
    input_data = np.expand_dims(sequence, axis=0)

    try:
        # Get prediction
        prediction = model.predict(input_data, verbose=0)

        # Get class index and confidence
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]

        # Map to gesture name
        gestures = metadata.get('gestures', [])
        if class_idx < len(gestures):
            gesture_name = gestures[class_idx]
        else:
            gesture_name = f"Class{class_idx}"

        return gesture_name, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

def main():
    """Main function for real-time recognition"""
    # Load model and metadata
    model, metadata = load_model_and_metadata()
    if model is None or metadata is None:
        print("Cannot continue without model and metadata.")
        return

    # Initialize video capture
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera initialized successfully.")

    # Landmark buffer to store sequences
    landmarks_buffer = deque(maxlen=PREDICTION_FRAMES)

    # Prediction smoothing
    recent_predictions = deque(maxlen=PREDICTION_SMOOTHING)

    # For calculating FPS
    prev_time = time.time()
    fps_buffer = deque(maxlen=30)

    # Sentence building
    sentence = []
    last_gesture = None
    gesture_counter = 0

    print("\nControls:")
    print("  • Press 'q' to quit")
    print("  • Press 'c' to clear current sentence")
    print("  • Press 's' to save sentence to file")
    print("\nReady! Starting recognition...")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_buffer.append(fps)
        avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0

        # Resize for processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Extract landmarks
        landmarks, frame = extract_hand_landmarks(frame)

        # Update buffer if hand detected
        current_prediction = None
        if landmarks is not None:
            landmarks_buffer.append(landmarks)

            # Only predict if we have enough frames
            if len(landmarks_buffer) == PREDICTION_FRAMES:
                # Make prediction
                gesture, confidence = predict_gesture(model, list(landmarks_buffer), metadata)

                # Add to recent predictions for smoothing
                if confidence > DETECTION_THRESHOLD:
                    recent_predictions.append(gesture)

                # Get most common recent prediction
                if recent_predictions:
                    from collections import Counter
                    counter = Counter(recent_predictions)
                    most_common = counter.most_common(1)[0]
                    current_prediction = most_common[0]

                    # Update sentence
                    if current_prediction != last_gesture:
                        sentence.append(current_prediction)
                        last_gesture = current_prediction
                        gesture_counter = 1
                    else:
                        gesture_counter += 1
                        # Reset if held too long
                        if gesture_counter > 30:  # ~1 second at 30fps
                            last_gesture = None
                            gesture_counter = 0

                # Display prediction info on frame
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if confidence > DETECTION_THRESHOLD else (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if confidence > DETECTION_THRESHOLD else (0, 0, 255),
                    2
                )
        else:
            # Reset buffer if there's a gap in detection
            if len(landmarks_buffer) > 0 and len(landmarks_buffer) < PREDICTION_FRAMES/2:
                landmarks_buffer.clear()

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # Display sentence
        current_sentence = " ".join(sentence[-5:])  # Show last 5 gestures
        cv2.putText(
            frame,
            f"Sentence: {current_sentence}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            last_gesture = None
            print("Sentence cleared")
        elif key == ord('s'):
            with open("recognized_sentence.txt", "a") as f:
                f.write(" ".join(sentence) + "\n")
            print(f"Saved: {' '.join(sentence)}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")

if __name__ == "__main__":
    main()