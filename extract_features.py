#!/usr/bin/env python3
"""
Enhanced Filipino Sign Language Feature Extraction
Integrates existing processing with advanced pipeline features
Author: PingusTingus
Date: 2025-03-17 06:15
"""

import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import time
import gc
import random
import re
import json
from collections import deque

# --- Configuration ---
DATASET_ROOT = r"C:\Users\Admin\OneDrive\Desktop\FSL SLR Dataset"  # Root folder of the dataset
CLIPS_PATH = os.path.join(DATASET_ROOT, "clips")
LABELS_PATH = os.path.join(DATASET_ROOT, "labels.csv")
OUTPUT_PATH = "dataset/"
PROCESSED_PATH = "data/processed/"  # New path for enhanced processed features
MAX_FRAMES = 50
PROCESS_EXISTING = True  # Set to False to skip already processed gestures
SAMPLE_RATE = 2  # Process every Nth frame
RESIZE_RESOLUTION = (320, 240)  # Lower resolution for processing speed
NUM_GESTURES = 10  # Number of gestures to process
RANDOM_SELECT = True  # Select gestures randomly
USE_DATA_AUGMENTATION = True  # Change to True if you want data augmentation
EXTRACT_ENHANCED_FEATURES = True  # Enable enhanced feature extraction
GENERATE_SEQUENCES = True  # Generate sequence data for training

print("=" * 70)
print("💪 Enhanced Filipino Sign Language Feature Extraction")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Processing {NUM_GESTURES} gestures {'(randomly selected)' if RANDOM_SELECT else '(first ones)'}")
print(f"👐 Enhanced: Processing BOTH hands for complete FSL gestures")
print(f"🚀 Advanced features: {'Enabled' if EXTRACT_ENHANCED_FEATURES else 'Disabled'}")
print("=" * 70)

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
print(f"📁 Output directories: {OUTPUT_PATH}, {PROCESSED_PATH}")

# MediaPipe setup
print("🔧 Initializing MediaPipe hands detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Track up to two hands
    model_complexity=0,  # Use lightest model for initial processing
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# --- Helper Functions ---
def sanitize_filename(name):
    """Convert a string to a valid filename"""
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove any non-alphanumeric, non-underscore characters
    name = re.sub(r'[^\w\-]', '', name)
    # Convert to lowercase
    name = name.lower()
    return name

# === ENHANCED FEATURE FUNCTIONS (FROM NEW PIPELINE) ===

def normalize_landmarks(landmarks, adaptive_scale=True):
    """
    Normalize landmarks to be invariant to camera position and hand size
    
    Parameters:
    - landmarks: Raw landmark points
    - adaptive_scale: Whether to use adaptive scaling based on hand size
    
    Returns:
    - Normalized landmarks
    """
    if len(landmarks) == 0:
        return landmarks
        
    # Extract x, y, z coordinates
    coords_x = landmarks[0::3]
    coords_y = landmarks[1::3]
    coords_z = landmarks[2::3] if len(landmarks) > len(coords_x)*2 else []
    
    # Find wrist position (typically the first landmark)
    wrist_x, wrist_y = coords_x[0], coords_y[0]
    
    # Center landmarks relative to wrist (more stable than center of mass)
    coords_x = coords_x - wrist_x
    coords_y = coords_y - wrist_y
    
    # Calculate scale using distance from wrist to middle finger MCP joint
    # This creates scale invariance regardless of distance from camera
    if adaptive_scale:
        # Index 9 is typically the middle finger MCP joint
        middle_finger_idx = min(9, len(coords_x) - 1)
        scale_reference = np.sqrt((coords_x[middle_finger_idx])**2 + 
                                 (coords_y[middle_finger_idx])**2)
        scale_factor = 0.1 / max(scale_reference, 0.0001)  # Avoid division by zero
    else:
        # Use maximum distance from wrist for scale
        max_dist = np.max(np.sqrt(coords_x**2 + coords_y**2))
        scale_factor = 1.0 / max(max_dist, 0.0001)  # Avoid division by zero
    
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

def rotation_invariant_features(landmarks):
    """
    Create rotation-invariant features so hand orientation doesn't affect recognition
    """
    if len(landmarks) == 0:
        return landmarks
    
    # Extract x, y coordinates
    coords_x = landmarks[0::3]
    coords_y = landmarks[1::3]
    coords_z = landmarks[2::3] if len(landmarks) > len(coords_x)*2 else []
    
    # Find key points to establish orientation
    # Wrist and middle finger MCP joint define main axis
    wrist_idx = 0
    middle_mcp_idx = min(9, len(coords_x)-1)
    
    # Calculate orientation angle
    dx = coords_x[middle_mcp_idx] - coords_x[wrist_idx]
    dy = coords_y[middle_mcp_idx] - coords_y[wrist_idx]
    angle = np.arctan2(dy, dx)
    
    # Rotate to standard orientation (pointing upward)
    target_angle = -np.pi/2  # -90 degrees (pointing up)
    rotation_angle = target_angle - angle
    
    # Create rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    
    # Apply rotation
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

def apply_enhanced_features(landmarks):
    """Apply enhanced feature processing to landmarks"""
    if EXTRACT_ENHANCED_FEATURES:
        # 1. Apply rotation invariance
        rotated_landmarks = rotation_invariant_features(landmarks)
        
        # 2. Apply normalization for scale invariance
        normalized_landmarks = normalize_landmarks(rotated_landmarks, adaptive_scale=True)
        
        # 3. Calculate zero velocity features for now (will be updated in real-time)
        # For static data, we'll just add zero velocity features as placeholders
        velocity_features = np.zeros_like(normalized_landmarks)
        
        # 4. Combine position and velocity features
        enhanced_features = np.concatenate([normalized_landmarks, velocity_features])
        
        return enhanced_features
    else:
        return landmarks

def extract_landmarks_from_video(video_path, max_frames=MAX_FRAMES):
    """Extract hand landmarks from video, processing both hands"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_sequence = []
    last_valid_landmarks = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every nth frame to improve speed
        if frame_count % SAMPLE_RATE == 0:
            # Resize for faster processing
            frame_rgb = cv2.cvtColor(cv2.resize(frame, RESIZE_RESOLUTION), cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # Initialize arrays for left and right hand landmarks
                left_hand_landmarks = None
                right_hand_landmarks = None

                # Process all detected hands (up to 2)
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Try to determine if this is left or right hand
                    is_right = True
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness = results.multi_handedness[i].classification[0].label
                        is_right = (handedness == "Right")

                    # Normalize landmarks relative to wrist
                    wrist = hand_landmarks.landmark[0]
                    normalized_landmarks = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                                                  for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

                    # Assign to left or right hand
                    if is_right:
                        right_hand_landmarks = normalized_landmarks
                    else:
                        left_hand_landmarks = normalized_landmarks

                # If we have at least one hand
                if left_hand_landmarks is not None or right_hand_landmarks is not None:
                    # Handle missing hand by using zeros or mirror
                    if left_hand_landmarks is None:
                        # If we only have right hand, use zeros for left hand
                        left_hand_landmarks = np.zeros_like(right_hand_landmarks)

                    if right_hand_landmarks is None:
                        # If we only have left hand, use zeros for right hand
                        right_hand_landmarks = np.zeros_like(left_hand_landmarks)

                    # Combine both hands' landmarks
                    combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
                    
                    # Apply enhanced features if enabled
                    if EXTRACT_ENHANCED_FEATURES:
                        combined_landmarks = apply_enhanced_features(combined_landmarks)

                    frame_sequence.append(combined_landmarks)
                    last_valid_landmarks = combined_landmarks

            # If no hands detected but we have previous landmarks, use those
            elif last_valid_landmarks is not None:
                frame_sequence.append(last_valid_landmarks)

        frame_count += 1

    cap.release()

    # Check if we have enough frames
    if len(frame_sequence) < 10:  # Require at least 10 valid frames
        return None

    # Adjust sequence to have exactly max_frames frames
    if len(frame_sequence) == max_frames:
        return np.array(frame_sequence)

    elif len(frame_sequence) > max_frames:
        # Downsample to max_frames
        indices = np.linspace(0, len(frame_sequence) - 1, max_frames).astype(int)
        return np.array([frame_sequence[i] for i in indices])

    else:
        # Need to pad/interpolate to reach max_frames
        # First, convert to numpy array
        sequence = np.array(frame_sequence)

        # Create empty result array
        result = np.zeros((max_frames, sequence.shape[1]), dtype=np.float32)

        # Copy existing frames
        result[:len(sequence)] = sequence

        # Fill remaining frames with last frame
        for i in range(len(sequence), max_frames):
            result[i] = sequence[-1]

        return result

def load_dataset_metadata():
    """Load dataset metadata from CSV files"""
    print("\n📊 Loading dataset metadata...")

    # Load labels CSV
    try:
        labels_df = pd.read_csv(LABELS_PATH)
        print(f"✅ Loaded labels file with {len(labels_df)} entries")
    except Exception as e:
        print(f"⚠️ Error loading labels file: {e}")
        return None

    # Create mapping from ID to gesture name and category
    id_mapping = {}
    categories = {}

    for _, row in labels_df.iterrows():
        gesture_id = str(row['id'])
        id_mapping[gesture_id] = row['label']
        categories[gesture_id] = row['category']

    return {
        'id_to_label': id_mapping,
        'categories': categories
    }

def prepare_sequences(landmark_data, labels, max_sequences=None):
    """
    Create sequence data suitable for the enhanced model training
    
    Args:
        landmark_data: Dictionary of gesture data (label -> sequences)
        labels: List of unique label names
        max_sequences: Maximum sequences per gesture (None = all)
        
    Returns:
        X_sequences, y_labels arrays ready for model training
    """
    print("\n🧩 Preparing sequences for model training...")
    
    X_sequences = []
    y_labels = []
    
    for idx, label in enumerate(tqdm(labels, desc="Creating sequences")):
        if label not in landmark_data:
            print(f"⚠️ Warning: No data found for label '{label}', skipping")
            continue
            
        sequences = landmark_data[label]
        
        # Limit number of sequences per gesture if specified
        if max_sequences is not None and len(sequences) > max_sequences:
            sequences = sequences[:max_sequences]
            
        for sequence in sequences:
            X_sequences.append(sequence)
            y_labels.append(idx)  # Use integer index as label
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_labels = np.array(y_labels)
    
    print(f"✅ Created {len(X_sequences)} sequences across {len(labels)} gestures")
    
    return X_sequences, y_labels


def process_dataset():
    """Process gestures from the dataset"""
    # Load metadata
    metadata = load_dataset_metadata()
    if metadata is None:
        print("⛔ Cannot proceed without metadata.")
        return

    # Find all gesture folders
    print("\n🔍 Scanning for gesture folders...")
    gesture_folders = []
    for item in os.listdir(CLIPS_PATH):
        folder_path = os.path.join(CLIPS_PATH, item)
        if os.path.isdir(folder_path):
            gesture_folders.append(folder_path)

    total_gestures = len(gesture_folders)
    print(f"📁 Found {total_gestures} gesture folders")

    # Select gestures
    if total_gestures > NUM_GESTURES:
        if RANDOM_SELECT:
            # Set a seed for reproducible results
            random.seed(42)
            print(f"🎲 Randomly selecting {NUM_GESTURES} gestures...")
            gesture_folders = random.sample(gesture_folders, NUM_GESTURES)
        else:
            print(f"📋 Taking first {NUM_GESTURES} gestures...")
            gesture_folders = gesture_folders[:NUM_GESTURES]
    else:
        print(f"⚠️ Found only {total_gestures} gestures, using all of them")

    # Track processed gestures for metadata
    processed_data = []

    # Store all sequences for creating training data
    all_landmark_data = {}
    all_labels = []

    # Process each gesture folder
    for gesture_folder in gesture_folders:
        gesture_id = os.path.basename(gesture_folder)

        # Get label from metadata
        if gesture_id in metadata['id_to_label']:
            gesture_label = metadata['id_to_label'][gesture_id]
            category = metadata['categories'].get(gesture_id, "unknown")
        else:
            gesture_label = f"unknown_{gesture_id}"
            category = "unknown"

        # Create a safe filename from the label
        safe_label = sanitize_filename(gesture_label)
        output_file = os.path.join(OUTPUT_PATH, f"{safe_label}.npy")

        # Also define enhanced output file
        enhanced_output_file = os.path.join(PROCESSED_PATH, f"{safe_label}.npy")

        # Skip if already processed and not forcing reprocess
        if not PROCESS_EXISTING and (os.path.exists(output_file) or os.path.exists(enhanced_output_file)):
            print(f"⏭️ Skipping {gesture_label} (ID: {gesture_id}) - already processed")

            # If we're skipping but need the data for sequence generation, load it
            if GENERATE_SEQUENCES and os.path.exists(enhanced_output_file):
                try:
                    sequences = np.load(enhanced_output_file, allow_pickle=True)
                    all_landmark_data[gesture_label] = sequences
                    all_labels.append(gesture_label)
                except Exception as e:
                    print(f"⚠️ Error loading existing data: {e}")

            processed_data.append({
                'id': gesture_id,
                'label': gesture_label,
                'safe_label': safe_label,
                'category': category
            })
            continue

        # Delete existing files if reprocessing
        if os.path.exists(output_file) and PROCESS_EXISTING:
            os.remove(output_file)
        if os.path.exists(enhanced_output_file) and PROCESS_EXISTING:
            os.remove(enhanced_output_file)

        # Find all MOV files in this gesture folder
        video_files = glob.glob(os.path.join(gesture_folder, "*.MOV"))

        if not video_files:
            print(f"⚠️ No MOV files found for {gesture_label} (ID: {gesture_id})")
            continue

        print(f"\n🎬 Processing '{gesture_label}' (ID: {gesture_id}, Category: {category})")
        print(f"   Found {len(video_files)} videos")

        # Process all videos for this gesture
        gesture_sequences = []
        enhanced_gesture_sequences = []

        for video_file in tqdm(video_files, desc=f"Extracting {gesture_label}", colour="green"):
            # Extract landmarks
            sequence = extract_landmarks_from_video(video_file)

            if sequence is not None:
                # Store original sequence (one per video)
                gesture_sequences.append(sequence)

                # For enhanced features, we've already applied them in extract_landmarks_from_video
                # if EXTRACT_ENHANCED_FEATURES is True
                enhanced_gesture_sequences.append(sequence)

                # IMPORTANT CHANGE: Only add augmented samples if explicitly enabled
                if USE_DATA_AUGMENTATION:
                    # Apply basic noise augmentation
                    noise_level = 0.005
                    noise = np.random.normal(0, noise_level, sequence.shape)
                    noisy_sequence = sequence + noise
                    gesture_sequences.append(noisy_sequence)
                    enhanced_gesture_sequences.append(noisy_sequence)

                    # Add more sophisticated augmentations
                    for i in range(1):  # Create 1 additional augmented sample
                        # Small random scaling
                        scale_factor = np.random.uniform(0.95, 1.05)
                        scaled_sequence = sequence * scale_factor

                        # Small rotation jitter for some frames
                        for frame_idx in range(0, len(sequence), 5):  # Every 5th frame
                            if frame_idx < len(sequence):
                                jitter_angle = np.random.uniform(-0.05, 0.05)  # Small angle in radians
                                cos_theta = np.cos(jitter_angle)
                                sin_theta = np.sin(jitter_angle)

                                # Apply small rotation to X-Y coordinates
                                for lm_idx in range(0, len(sequence[frame_idx]), 3):
                                    if lm_idx + 1 < len(sequence[frame_idx]):
                                        x = scaled_sequence[frame_idx][lm_idx]
                                        y = scaled_sequence[frame_idx][lm_idx + 1]
                                        scaled_sequence[frame_idx][lm_idx] = x * cos_theta - y * sin_theta
                                        scaled_sequence[frame_idx][lm_idx + 1] = x * sin_theta + y * cos_theta

                        gesture_sequences.append(scaled_sequence)
                        enhanced_gesture_sequences.append(scaled_sequence)

        # Save if we have valid sequences
        if len(gesture_sequences) > 0:
            # Save original format
            np.save(output_file, np.array(gesture_sequences))

            # Save enhanced features if enabled
            if EXTRACT_ENHANCED_FEATURES:
                np.save(enhanced_output_file, np.array(enhanced_gesture_sequences))

                # Store for sequence generation
                if GENERATE_SEQUENCES:
                    all_landmark_data[gesture_label] = enhanced_gesture_sequences
                    all_labels.append(gesture_label)

            print(f"✅ Saved {len(gesture_sequences)} sequences for '{gesture_label}'")
            print(f"   Feature vector size: {gesture_sequences[0].shape}")

            processed_data.append({
                'id': gesture_id,
                'label': gesture_label,
                'safe_label': safe_label,
                'category': category,
                'feature_shape': gesture_sequences[0].shape,
                'num_samples': len(gesture_sequences),
                'uses_both_hands': True  # Flag that we're using both hands in this dataset
            })
        else:
            print(f"⚠️ No valid sequences extracted for '{gesture_label}'")

        # Cleanup to free memory
        gc.collect()

    # Save metadata for training script
    metadata_dict = {
        'gestures': processed_data,
        'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_gestures': len(processed_data),
        'max_frames': MAX_FRAMES,
        'id_to_label': {item['id']: item['label'] for item in processed_data},
        'label_to_file': {item['label']: f"{item['safe_label']}.npy" for item in processed_data},
        'uses_both_hands': True,  # Important for the real-time script to know
        'data_augmentation_used': USE_DATA_AUGMENTATION,  # Document if augmentation was used
        'enhanced_features': EXTRACT_ENHANCED_FEATURES  # Document if enhanced features were used
    }

    np.save(os.path.join(OUTPUT_PATH, "metadata.npy"), metadata_dict)

    # Also save enhanced metadata
    if EXTRACT_ENHANCED_FEATURES:
        np.save(os.path.join(PROCESSED_PATH, "metadata.npy"), metadata_dict)

    # Also save a readable JSON version
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=4)

    if EXTRACT_ENHANCED_FEATURES:
        with open(os.path.join(PROCESSED_PATH, "metadata.json"), 'w') as f:
            json.dump(metadata_dict, f, indent=4)

    print(f"\n💾 Saved metadata for {len(processed_data)} gestures")

    # Generate sequence data for training if enabled
    if GENERATE_SEQUENCES and len(all_labels) > 0:
        print("\n📊 Generating sequence data for training...")

        # Create sequences suitable for the enhanced model
        X_sequences, y_labels = prepare_sequences(all_landmark_data, all_labels)

        # Save sequences
        sequence_data = {
            'X': X_sequences,
            'y': y_labels,
            'labels': all_labels,
            'sequence_length': MAX_FRAMES,
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_features': EXTRACT_ENHANCED_FEATURES
        }

        sequences_file = os.path.join(PROCESSED_PATH, "sequences.pkl")
        with open(sequences_file, 'wb') as f:
            import pickle
            pickle.dump(sequence_data, f)

        print(f"✅ Saved {len(X_sequences)} training sequences to {sequences_file}")

    # Generate a summary table
    print("\n📋 Gesture Processing Summary:")
    print("-" * 105)
    print(f"{'ID':<6} {'Label':<30} {'Filename':<25} {'Category':<15} {'Samples':<10} {'Status':<10}")
    print("-" * 105)

    for item in processed_data:
        status = "Processed" if os.path.exists(os.path.join(OUTPUT_PATH, f"{item['safe_label']}.npy")) else "Failed"
        samples = item.get('num_samples', 'N/A')
        print(
            f"{item['id']:<6} {item['label'][:30]:<30} {item['safe_label'][:25]:<25} {item['category'][:15]:<15} {samples:<10} {status:<10}")

    print("-" * 105)


def main():
    """Main function"""
    start_time = time.time()

    # Check if dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"⛔ Dataset root folder '{DATASET_ROOT}' not found!")
        print("   Please make sure the path is correct.")
        return

    if not os.path.exists(CLIPS_PATH):
        print(f"⛔ Clips folder '{CLIPS_PATH}' not found!")
        return

    # Process dataset
    process_dataset()

    # Print statistics
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print(f"✨ Feature extraction completed in {duration:.1f} seconds")
    print(f"📂 Processed features saved to {OUTPUT_PATH}")
    print(f"📂 Enhanced features saved to {PROCESSED_PATH}")
    print(f"🔢 Processed {NUM_GESTURES} gestures with BOTH hands")
    print(f"🧪 Data augmentation: {'Enabled' if USE_DATA_AUGMENTATION else 'Disabled'}")
    print(f"🚀 Enhanced features: {'Enabled' if EXTRACT_ENHANCED_FEATURES else 'Disabled'}")
    print(f"🧩 Sequence generation: {'Enabled' if GENERATE_SEQUENCES else 'Disabled'}")
    print("=" * 70)


if __name__ == "__main__":
    main()