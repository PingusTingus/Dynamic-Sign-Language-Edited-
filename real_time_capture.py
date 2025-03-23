#!/usr/bin/env python3
"""
Enhanced Filipino Sign Language Real-Time Recognition
Works with PC webcam and integrates enhanced features
Author: PingusTingus
Date: 2025-03-17 06:40:52
"""

import os
import cv2
import numpy as np
import time
from collections import deque
import json
import platform
import sys
import mediapipe as mp

# --- Configuration ---
CONFIG = {
    "model_dir": "models/enhanced/",  # Directory containing enhanced model
    "model_file": "best_model.keras",  # Model filename (h5 or keras)
    "tflite_model": "model_int8.tflite",  # TFLite model for edge devices
    "class_mapping_file": "class_mapping.json",  # Class mapping file
    "metadata_path": "evaluation_results.json",  # Metadata for legacy support
    "camera_id": 0,  # Camera ID (0 for default)
    "detection_threshold": 0.7,  # Confidence threshold for detection
    "frame_width": 640,  # Frame width
    "frame_height": 480,  # Frame height
    "process_resolution": (320, 240),  # Lower resolution for processing
    "process_every_n_frames": 2,  # Process every N frames for speed
    "prediction_frames": 50,  # Match model's sequence length
    "cooldown_period": 1.0,  # Seconds between predictions
    "prediction_smoothing": 5,  # Frames to smooth predictions
    "use_tflite": False,  # Use TFLite model (auto-detected)
    "use_picamera": False,  # Use Picamera if available
    "use_enhanced_features": True,  # Enable enhanced features
    "use_rotation_invariance": True,  # Apply rotation invariance
    "use_velocity_features": True  # Use velocity features if available
}

print("=" * 70)
print("💪 Enhanced Filipino Sign Language Real-Time Recognition")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🖥️ Platform: {platform.platform()}")
print(f"🔍 Using model from: {CONFIG['model_dir']}")
print("=" * 70)

# --- Automatically detect available modules ---

# Check for TensorFlow or TFLite
USE_TFLITE = False
try:
    print("Attempting to import TensorFlow...")
    import tensorflow as tf

    print(f"✅ TensorFlow {tf.__version__} available")

    # Check if we have TFLite installed as well
    try:
        tflite_interpreter = tf.lite.Interpreter
        print("✅ TensorFlow Lite support available through TF")
        CONFIG["use_tflite"] = True  # Enable TFLite option
    except:
        print("⚠️ TensorFlow Lite not available through TF")

except ImportError:
    print("Full TensorFlow not available...")

    # Try importing TFLite runtime
    try:
        import tflite_runtime.interpreter as tflite

        print("✅ TensorFlow Lite runtime available")
        CONFIG["use_tflite"] = True
        USE_TFLITE = True
    except ImportError:
        print("❌ Neither TensorFlow nor TensorFlow Lite runtime is available.")
        print("Please install one of these: pip install tensorflow or pip install tflite_runtime")
        sys.exit(1)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Determine if we're on a Pi or desktop and initialize appropriate hand model
is_raspberry_pi = platform.uname().system == "Linux" and "arm" in platform.uname().machine
if is_raspberry_pi:
    print("🍓 Running on Raspberry Pi, optimizing for performance")
    # Use lighter model for RPi
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,  # Increased from 0.3
        smooth_landmarks=True  # Enable internal MediaPipe smoothing
    )
else:
    print("💻 Running on desktop, optimizing for accuracy")
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Track both hands
        model_complexity=1,  # Use medium model for desktop
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )


class EnhancedFSLRecognition:
    """Enhanced FSL real-time recognition system"""

    def __init__(self, config=None):
        """Initialize the FSL recognition system"""
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        # Initialize gesture recognition variables
        self.landmarks_buffer = deque(maxlen=self.config["prediction_frames"])
        self.position_history = deque(maxlen=10)
        self.no_hands_counter = 0
        self.last_gesture_position = None
        self.cooldown_class = None
        self.cooldown_counter = 0
        self.is_cooldown_active = False

        # Initialize text output variable
        self.current_text = ""  # This will store recognized gestures

        # Initialize time tracking for FPS calculation
        self.prev_time = time.time()  # Initialize this variable to avoid the AttributeError

        # Enhanced feature variables
        self.prev_landmarks = None
        self.left_hand_prev = None
        self.right_hand_prev = None
        self.prev_annotated_frame = None

        # Enhanced buffer system for continuous recognition
        self.max_buffer_size = 120  # About 4 seconds at 30fps
        self.prediction_stride = 10  # Check for new gestures every N frames
        self.min_gesture_frames = 30  # Minimum frames for a valid gesture
        self.min_confidence = 0.7  # Minimum confidence for acceptance
        self.last_detected_class = None  # Track last detected gesture
        self.cooldown_frames = 15  # Frames to wait before detecting same gesture again
        self.cooldown_class = None  # Class currently in cooldown
        self.cooldown_counter = 0  # Counter for cooldown
        self.position_history = deque(maxlen=10)
        self.last_gesture_position = None
        self.no_hands_counter = 0

        # Load model and class mapping
        self.model, self.class_labels = self.load_model_and_classes()

        print(f"✅ Recognition system initialized with {len(self.class_labels)} gestures")



    def load_model_and_classes(self):
        """Load the trained model and class mapping"""
        print("\nLoading model and class labels...")

        model = None
        class_labels = []

        # Load class mapping first
        try:
            class_mapping_path = os.path.join(self.config["model_dir"], self.config["class_mapping_file"])

            if os.path.exists(class_mapping_path):
                with open(class_mapping_path, 'r') as f:
                    class_mapping = json.load(f)

                # Extract class labels
                if "classes" in class_mapping:
                    class_labels = class_mapping["classes"]
                    print(f"✅ Loaded {len(class_labels)} gesture classes")
            else:
                # Try legacy metadata path
                metadata_path = os.path.join(self.config["model_dir"], self.config["metadata_path"])
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    if "gestures" in metadata:
                        class_labels = metadata["gestures"]
                        print(f"✅ Loaded {len(class_labels)} gesture classes from metadata")
                else:
                    print(f"⚠️ No class mapping found at {class_mapping_path}")
                    print("Will use generic class names")
                    class_labels = [f"Class_{i}" for i in range(10)]  # Default classes

        except Exception as e:
            print(f"⚠️ Error loading class mapping: {e}")
            class_labels = [f"Class_{i}" for i in range(10)]  # Default classes

        # Try TFLite model if specified
        if self.config["use_tflite"]:
            tflite_path = os.path.join(self.config["model_dir"], self.config["tflite_model"])

            if os.path.exists(tflite_path):
                try:
                    if USE_TFLITE:  # Using tflite_runtime
                        interpreter = tflite.Interpreter(model_path=tflite_path)
                    else:  # Using tf.lite
                        interpreter = tf.lite.Interpreter(model_path=tflite_path)

                    interpreter.allocate_tensors()

                    # Get input and output details
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()

                    # Print model input shape
                    print(f"Model expects input shape: {input_details[0]['shape']}")

                    # Get sequence length from model input shape
                    if len(input_details[0]['shape']) >= 2:
                        self.config["prediction_frames"] = input_details[0]['shape'][1]
                        print(f"Setting sequence length to {self.config['prediction_frames']} from model")

                    # Create a wrapper class to match the Keras API
                    class TFLiteModel:
                        def __init__(self, interpreter, input_details, output_details):
                            self.interpreter = interpreter
                            self.input_details = input_details
                            self.output_details = output_details
                            self.input_shape = input_details[0]['shape']

                        def predict(self, x, verbose=0):
                            # Handle batch dimension
                            if self.input_shape[0] == 1 and x.shape[0] != 1:
                                x = np.expand_dims(x, 0)

                            # Check if shapes match or need reshaping
                            if x.shape != tuple(self.input_shape):
                                try:
                                    x = np.reshape(x, self.input_shape)
                                except:
                                    print("Failed to reshape input")
                                    return np.zeros((1, self.output_details[0]['shape'][-1]))

                            # Set input tensor
                            self.interpreter.set_tensor(
                                self.input_details[0]['index'],
                                x.astype(np.float32)
                            )

                            # Run inference
                            self.interpreter.invoke()

                            # Get output tensor
                            output = self.interpreter.get_tensor(
                                self.output_details[0]['index']
                            )

                            return output

                    model = TFLiteModel(interpreter, input_details, output_details)
                    print("✅ Successfully loaded TFLite model")

                except Exception as e:
                    print(f"❌ Error loading TFLite model: {e}")
                    print("Will try standard model")

        # Try standard Keras model if TFLite failed or not requested
        if model is None:
            try:
                # Try different model file extensions
                for ext in ['.keras', '.h5']:
                    model_path = os.path.join(self.config["model_dir"],
                                              self.config["model_file"].replace('.keras', ext))

                    if os.path.exists(model_path):
                        model = tf.keras.models.load_model(model_path)
                        print(f"✅ Successfully loaded Keras model from {model_path}")

                        # Get sequence length from model
                        if hasattr(model, 'input_shape') and len(model.input_shape) >= 2:
                            self.config["prediction_frames"] = model.input_shape[1]
                            print(f"Setting sequence length to {self.config['prediction_frames']} from model")

                        break

            except Exception as e:
                print(f"❌ Error loading Keras model: {e}")

        # Check if model loaded successfully
        if model is None:
            print("❌ Failed to load any model.")
            sys.exit(1)

        return model, class_labels

    def normalize_landmarks(self, landmarks):
        """Normalize landmarks for scale and translation invariance"""
        if landmarks is None or len(landmarks) == 0:
            return None

        # Extract x, y, z coordinates
        coords_x = landmarks[0::3]
        coords_y = landmarks[1::3]
        coords_z = landmarks[2::3] if len(landmarks) % 3 == 0 else []

        # Find wrist position (typically the first landmark)
        wrist_x, wrist_y = coords_x[0], coords_y[0]

        # Center landmarks relative to wrist
        coords_x = coords_x - wrist_x
        coords_y = coords_y - wrist_y

        # Calculate scale using distance from wrist to middle finger MCP joint
        # Index 9 is typically the middle finger MCP joint
        middle_finger_idx = min(9, len(coords_x) - 1)
        scale_reference = np.sqrt((coords_x[middle_finger_idx]) ** 2 +
                                  (coords_y[middle_finger_idx]) ** 2)
        scale_factor = 0.1 / max(scale_reference, 0.0001)  # Avoid division by zero

        # Apply scaling
        coords_x = coords_x * scale_factor
        coords_y = coords_y * scale_factor

        # Z-coordinates need special handling - normalize relative to wrist depth
        if len(coords_z) > 0:
            wrist_z = coords_z[0]
            coords_z = coords_z - wrist_z
            # Scale Z values to similar magnitude as X/Y
            max_z_dist = max(abs(np.max(coords_z)), abs(np.min(coords_z)), 0.0001)
            z_scale = 0.5 / max_z_dist  # Smaller scale for Z
            coords_z = coords_z * z_scale

        # Reassemble the landmark array
        normalized = []
        for i in range(len(coords_x)):
            normalized.append(coords_x[i])
            normalized.append(coords_y[i])
            if i < len(coords_z):
                normalized.append(coords_z[i])

        return np.array(normalized)

    def apply_rotation_invariance(self, landmarks):
        """Apply rotation invariance to landmarks"""
        if landmarks is None or not self.config["use_rotation_invariance"]:
            return landmarks

        # Extract x, y coordinates
        coords_x = landmarks[0::3]
        coords_y = landmarks[1::3]
        coords_z = landmarks[2::3] if len(landmarks) % 3 == 0 else []

        # Find wrist and middle finger MCP joint
        wrist_idx = 0
        middle_mcp_idx = min(9, len(coords_x) - 1)

        # Calculate orientation angle
        dx = coords_x[middle_mcp_idx] - coords_x[wrist_idx]
        dy = coords_y[middle_mcp_idx] - coords_y[wrist_idx]
        angle = np.arctan2(dy, dx)

        # Rotate to standard orientation (pointing upward)
        target_angle = -np.pi / 2  # -90 degrees (pointing up)
        rotation_angle = target_angle - angle

        # Apply rotation
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)

        rotated_x = coords_x * cos_theta - coords_y * sin_theta
        rotated_y = coords_x * sin_theta + coords_y * cos_theta

        # Reassemble the landmark array
        rotated_landmarks = []
        for i in range(len(rotated_x)):
            rotated_landmarks.append(rotated_x[i])
            rotated_landmarks.append(rotated_y[i])
            if i < len(coords_z):
                rotated_landmarks.append(coords_z[i])

        return np.array(rotated_landmarks)

    def calculate_velocity_features(self, current_landmarks):
        """Calculate velocity features from current and previous landmarks"""
        if not self.config["use_velocity_features"] or current_landmarks is None:
            return None

        if self.prev_landmarks is None:
            # No previous landmarks, return zeros
            return np.zeros_like(current_landmarks)

        # Check if shapes match
        if current_landmarks.shape != self.prev_landmarks.shape:
            # Shape mismatch - return zeros of appropriate shape
            print(f"Shape mismatch detected: current {current_landmarks.shape} vs previous {self.prev_landmarks.shape}")
            return np.zeros_like(current_landmarks)

        # Calculate velocity as difference between current and previous landmarks
        velocity = current_landmarks - self.prev_landmarks

        # Scale down velocity to reasonable range
        velocity = velocity * 0.5

        return velocity

    def preprocess_landmarks(self, landmarks):
        """Apply all preprocessing steps to landmarks"""
        if landmarks is None:
            return None

        # Make a copy to avoid modifying original
        processed = landmarks.copy()

        # 1. Apply normalization
        processed = self.normalize_landmarks(processed)

        # 2. Apply rotation invariance if enabled
        if self.config["use_rotation_invariance"]:
            processed = self.apply_rotation_invariance(processed)

        # 3. Calculate and include velocity features if enabled
        if self.config["use_velocity_features"]:
            velocity = self.calculate_velocity_features(processed)
            if velocity is not None:
                # Only update previous landmarks after successful processing
                self.prev_landmarks = processed.copy()
                processed = np.concatenate([processed, velocity])

        return processed

    def extract_hand_landmarks(self, frame):
        """Extract and normalize hand landmarks from a frame"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, self.config["process_resolution"])
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Process the image
        results = hands.process(frame_rgb)

        # Create annotated frame copy
        annotated_frame = frame.copy()

        # For smoother visualization
        smoothing_factor = 0.3  # Lower means more smoothing

        # Initialize left and right hand landmarks as None
        left_hand_landmarks = None
        right_hand_landmarks = None

        if not results.multi_hand_landmarks:
            # If no hands detected, gradually fade out previous detection
            if hasattr(self, 'prev_annotated_frame') and self.prev_annotated_frame is not None:
                # Blend previous frame with current frame
                alpha = 0.7  # Fade out rate
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0 - alpha, self.prev_annotated_frame, alpha, 0)
            return None, annotated_frame

        # Determine number of landmarks per hand
        landmarks_per_hand = len(results.multi_hand_landmarks[0].landmark) * 3  # x,y,z for each point

        # Process all detected hands (up to 2)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Determine handedness
            is_right = True
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label
                is_right = (handedness == "Right")

                # Draw handedness label
                handedness_text = "Right" if is_right else "Left"
                x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), int(
                    hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(annotated_frame, handedness_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Extract raw landmarks
            raw_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
            ], dtype=np.float32).flatten()

            # Apply temporal smoothing if previous landmarks exist
            hand_key = 'right_hand_prev' if is_right else 'left_hand_prev'
            if hasattr(self, hand_key) and getattr(self, hand_key) is not None:
                prev_landmarks = getattr(self, hand_key)
                # Only smooth if dimensions match
                if len(prev_landmarks) == len(raw_landmarks):
                    raw_landmarks = smoothing_factor * raw_landmarks + (1 - smoothing_factor) * prev_landmarks

            # Store for next frame
            setattr(self, hand_key, raw_landmarks)

            # Assign to left or right hand
            if is_right:
                right_hand_landmarks = raw_landmarks
            else:
                left_hand_landmarks = raw_landmarks

        # Handle single or dual-hand models
        if self.config.get("uses_both_hands", True):
            # We need both hands - if one is missing, use previous values if available
            if left_hand_landmarks is None:
                if hasattr(self, 'left_hand_prev') and self.left_hand_prev is not None:
                    left_hand_landmarks = self.left_hand_prev.copy()
                else:
                    left_hand_landmarks = np.zeros(landmarks_per_hand)

            if right_hand_landmarks is None:
                if hasattr(self, 'right_hand_prev') and self.right_hand_prev is not None:
                    right_hand_landmarks = self.right_hand_prev.copy()
                else:
                    right_hand_landmarks = np.zeros(landmarks_per_hand)

            # Ensure both have same dimensions
            if len(left_hand_landmarks) != len(right_hand_landmarks):
                max_dim = max(len(left_hand_landmarks), len(right_hand_landmarks))
                if len(left_hand_landmarks) < max_dim:
                    left_hand_landmarks = np.pad(left_hand_landmarks, (0, max_dim - len(left_hand_landmarks)))
                if len(right_hand_landmarks) < max_dim:
                    right_hand_landmarks = np.pad(right_hand_landmarks, (0, max_dim - len(right_hand_landmarks)))

            # Combine both hands
            combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])

            # Store annotated frame for smoother transitions
            self.prev_annotated_frame = annotated_frame.copy()

            # Apply preprocessing
            return self.preprocess_landmarks(combined_landmarks), annotated_frame
        else:
            # Single hand model - use the hand that's present, preferring right
            if right_hand_landmarks is not None:
                self.prev_annotated_frame = annotated_frame.copy()
                return self.preprocess_landmarks(right_hand_landmarks), annotated_frame
            elif left_hand_landmarks is not None:
                self.prev_annotated_frame = annotated_frame.copy()
                return self.preprocess_landmarks(left_hand_landmarks), annotated_frame

        self.prev_annotated_frame = annotated_frame.copy()
        return None, annotated_frame

    def process_frame(self, frame):
        """Process a frame and update predictions using continuous recognition"""
        return self.continuous_recognition_buffer(frame)

    def continuous_recognition_buffer(self, frame):
        """Process frame using continuous recognition buffer system with improved gesture switching"""
        # Extract landmarks
        landmarks, annotated_frame = self.extract_hand_landmarks(frame)
        landmarks_list = list(self.landmarks_buffer)

        # Track sudden changes in hand position for gesture boundaries
        if not hasattr(self, 'position_history'):
            self.position_history = deque(maxlen=10)
            self.last_gesture_position = None
            self.no_hands_counter = 0

        # If no hand detected
        if landmarks is None:
            # Visual indicator
            cv2.putText(annotated_frame, "No hands detected",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # Count frames without hands
            self.no_hands_counter += 1

            # If hands are gone for a while, reset the buffer
            if self.no_hands_counter > 15:  # About half a second at 30fps
                self.landmarks_buffer.clear()
                self.cooldown_class = None
                self.last_gesture_position = None
                print("Buffer reset - no hands detected")

            return None, annotated_frame
        else:
            # Reset no-hands counter when hands are detected
            self.no_hands_counter = 0

        # Calculate hand position (average of x,y coordinates)
        hand_position = np.mean(landmarks[0::3]), np.mean(landmarks[1::3])
        self.position_history.append(hand_position)

        # Check for significant position change
        position_changed = False
        if self.last_gesture_position is not None and len(self.position_history) > 5:
            # Calculate average recent position
            recent_x = np.mean([pos[0] for pos in self.position_history])
            recent_y = np.mean([pos[1] for pos in self.position_history])

            # Calculate distance from last gesture position
            dist = np.sqrt((recent_x - self.last_gesture_position[0]) ** 2 +
                           (recent_y - self.last_gesture_position[1]) ** 2)

            # If significant movement, consider it a new gesture attempt
            if dist > 0.15:  # Threshold for significant movement
                position_changed = True
                print(f"Significant hand position change detected: {dist:.3f}")
                # Clear cooldown to allow new gestures
                self.cooldown_class = None
                # Partially clear buffer to start fresh
                keep_frames = min(15, len(self.landmarks_buffer))
                self.landmarks_buffer = self.landmarks_buffer[-keep_frames:]

        # Add landmark to buffer
        self.landmarks_buffer.append(landmarks)

        # Keep buffer at maximum size
        if len(self.landmarks_buffer) > self.max_buffer_size:
            self.landmarks_buffer.pop(0)

        # Update cooldown if active
        if self.cooldown_class is not None:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_frames:
                self.cooldown_class = None
                self.cooldown_counter = 0
                print("Cooldown expired, ready for new gesture")

        # Display buffer fullness
        buffer_percentage = len(self.landmarks_buffer) / self.max_buffer_size * 100
        cv2.putText(annotated_frame,
                    f"Buffer: {len(self.landmarks_buffer)}/{self.max_buffer_size} ({buffer_percentage:.0f}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw buffer progress bar
        bar_width = int(200 * buffer_percentage / 100)
        cv2.rectangle(annotated_frame, (10, 70), (10 + bar_width, 85), (0, 255, 0), -1)
        cv2.rectangle(annotated_frame, (10, 70), (210, 85), (255, 255, 255), 2)

        # Only perform recognition if we have enough frames and:
        # 1. It's time to check based on stride, OR
        # 2. We detected a significant position change
        ready_to_check = (len(self.landmarks_buffer) >= self.config["prediction_frames"] and
                          (len(self.landmarks_buffer) % self.prediction_stride == 0 or position_changed))

        if ready_to_check:
            # Try multiple window positions to find the best gesture
            best_confidence = 0
            best_class = None
            best_window_position = 0

            # Try different window positions, prioritizing more recent frames
            max_start_position = len(self.landmarks_buffer) - self.config["prediction_frames"]

            # Check windows at different starting points
            window_positions = [
                max_start_position,  # Most recent frames
                max_start_position - 10,  # Slightly older
                max_start_position - 20,  # Even older
                0  # From the beginning
            ]

            # Filter valid window positions
            window_positions = [pos for pos in window_positions if pos >= 0 and pos <= max_start_position]

            # Try each window position
            for start_pos in window_positions:
                window = landmarks_list[start_pos:start_pos + self.config["prediction_frames"]]

                try:
                    # Prepare sequence
                    x = np.array(window)

                    # Reshape according to model expectations
                    if hasattr(self.model, 'input_shape'):
                        x = x.reshape(1, self.config["prediction_frames"], -1)
                    elif hasattr(self.model, 'input_details'):
                        x = x.reshape(1, self.config["prediction_frames"], -1)
                    else:
                        x = x.reshape(1, self.config["prediction_frames"], -1)

                    # Predict
                    prediction = self.model.predict(x, verbose=0)

                    # Get top 3 predictions for better analysis
                    top3_indices = np.argsort(prediction[0])[-3:][::-1]
                    top3_confidences = prediction[0][top3_indices]

                    class_idx = top3_indices[0]
                    confidence = top3_confidences[0]

                    # Track best prediction across windows
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_class = class_idx
                        best_window_position = start_pos

                except Exception as e:
                    print(f"Error during prediction: {e}")

            # If we found a good prediction
            if best_class is not None and best_confidence >= self.min_confidence:
                predicted_class = self.class_labels[best_class]

                # Check if this is a new gesture (different from the last detected)
                is_new_gesture = (predicted_class != self.cooldown_class)

                # If it's a new gesture with good confidence
                if is_new_gesture or position_changed:
                    # Add the recognized gesture to the current text
                    self.current_text += predicted_class + " "

                    # Store current position as last gesture position
                    if len(self.position_history) > 0:
                        recent_x = np.mean([pos[0] for pos in self.position_history])
                        recent_y = np.mean([pos[1] for pos in self.position_history])
                        self.last_gesture_position = (recent_x, recent_y)

                    # Set cooldown for this class to prevent rapid repeats
                    self.cooldown_class = predicted_class
                    self.cooldown_counter = 0

                    # Display recognition result on the annotated frame
                    cv2.putText(annotated_frame, f"Recognized: {predicted_class}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

                    # Show confidence on the frame
                    cv2.putText(annotated_frame, f"Confidence: {best_confidence:.2f}",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    # Print the recognized gesture and confidence to the console
                    print(f"Recognized: {predicted_class} (Confidence: {best_confidence:.2f})")

                    # Clear the buffer for the next gesture
                    self.landmarks_buffer = list(self.landmarks_buffer)[-15:]  # Keep only the most recent 15 frames


            else:
                # No confident prediction
                cv2.putText(annotated_frame, "Analyzing...",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

        # Add current text to the frame
        text_y = annotated_frame.shape[0] - 20
        cv2.putText(
            annotated_frame,
            f"Text: {self.current_text[-40:]}",  # Show last 40 chars for easier reading
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show cooldown status if active
        if self.cooldown_class is not None:
            frames_left = self.cooldown_frames - self.cooldown_counter
            cv2.putText(annotated_frame, f"Cooldown: {self.cooldown_class} ({frames_left})",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - self.prev_time)  # Avoid division by zero
        self.prev_time = current_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Indicate if position changed (for debugging)
        if position_changed:
            cv2.putText(annotated_frame, "Position Changed!",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return landmarks, annotated_frame

    def clear_text(self):
        """Clear the current text"""
        self.current_text = ""
        print("Text cleared")


def main():
    """Main function for real-time recognition"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Filipino Sign Language Recognition')
    parser.add_argument('--model-dir', default=CONFIG["model_dir"],
                        help=f'Directory containing model files (default: {CONFIG["model_dir"]})')
    parser.add_argument('--camera', type=int, default=CONFIG["camera_id"],
                        help=f'Camera ID (default: {CONFIG["camera_id"]})')
    parser.add_argument('--threshold', type=float, default=CONFIG["detection_threshold"],
                        help=f'Detection threshold (default: {CONFIG["detection_threshold"]})')
    parser.add_argument('--use-tflite', action='store_true',
                        help='Force use of TFLite model if available')
    parser.add_argument('--no-velocity', action='store_true',
                        help='Disable velocity features')
    parser.add_argument('--no-rotation', action='store_true',
                        help='Disable rotation invariance')
    parser.add_argument('--smoothing', type=float, default=0.3,
                        help='Temporal smoothing factor (0-1, lower is smoother)')

    args = parser.parse_args()

    # Update config based on arguments
    CONFIG["model_dir"] = args.model_dir
    CONFIG["camera_id"] = args.camera
    CONFIG["detection_threshold"] = args.threshold
    CONFIG["use_tflite"] = args.use_tflite or CONFIG["use_tflite"]
    CONFIG["use_velocity_features"] = not args.no_velocity
    CONFIG["use_rotation_invariance"] = not args.no_rotation
    CONFIG["process_every_n_frames"] = 1  # Process every frame for smoother visuals

    # Initialize recognition system
    recognition_system = EnhancedFSLRecognition(CONFIG)

    # Initialize camera
    print(f"Opening camera {CONFIG['camera_id']}...")
    cap = cv2.VideoCapture(CONFIG["camera_id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)  # Try to get 30fps if supported

    if not cap.isOpened():
        print("❌ Error: Could not open camera. Exiting.")
        return

    print("\n✅ Setup complete! Starting real-time recognition...")
    print("Press 'q' to quit, 'c' to clear text")

    # Main loop
    frame_count = 0
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process every frame for visualization, but only add to prediction buffer periodically
        _, annotated_frame = recognition_system.process_frame(frame)

        # Increment frame counter
        frame_count += 1

        # Display the frame
        cv2.imshow("Filipino Sign Language Recognition", annotated_frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            recognition_system.clear_text()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
