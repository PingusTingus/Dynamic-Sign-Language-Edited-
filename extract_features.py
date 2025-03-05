"""
Filipino Sign Language Feature Extraction
Customized for FSL SLR Dataset structure - Processes only 5 gestures
Uses gesture labels for file naming instead of IDs
Author: PingusTingus
Date: 2025-03-03 06:42:20
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

# --- Configuration ---
DATASET_ROOT = r"C:\Users\Admin\OneDrive\Desktop\FSL SLR Dataset"  # Root folder of the dataset
CLIPS_PATH = os.path.join(DATASET_ROOT, "clips")
LABELS_PATH = os.path.join(DATASET_ROOT, "labels.csv")
OUTPUT_PATH = "dataset/"
MAX_FRAMES = 50
PROCESS_EXISTING = True  # Set to False to skip already processed gestures
SAMPLE_RATE = 2  # Process every Nth frame
RESIZE_RESOLUTION = (320, 240)  # Lower resolution for processing speed
NUM_GESTURES = 10  # Process exactly 5 gestures
RANDOM_SELECT = True  # Select gestures randomly

print("=" * 70)
print("💪 Filipino Sign Language Feature Extraction")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: 2025-03-03 06:42:20")
print(f"🎯 Processing {NUM_GESTURES} gestures {'(randomly selected)' if RANDOM_SELECT else '(first ones)'}")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"📁 Output directory: {OUTPUT_PATH}")

# MediaPipe setup
print("🔧 Initializing MediaPipe hands detection...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Only track one hand for performance
    model_complexity=0,  # Use lightest model
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

def extract_landmarks_from_video(video_path, max_frames=MAX_FRAMES):
    """Extract hand landmarks from video"""
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
                for hand_landmarks in results.multi_hand_landmarks:
                    # Normalize landmarks relative to wrist
                    wrist = hand_landmarks.landmark[0]
                    landmarks = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                                        for lm in hand_landmarks.landmark], dtype=np.float32).flatten()

                    frame_sequence.append(landmarks)
                    last_valid_landmarks = landmarks
                    break  # Only use first detected hand

            # If no hand detected but we have previous landmarks, use those
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

def process_dataset():
    """Process exactly 5 gestures from the dataset"""
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

    # Select 5 gestures
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

        # Skip if already processed and not forcing reprocess
        if not PROCESS_EXISTING and os.path.exists(output_file):
            print(f"⏭️ Skipping {gesture_label} (ID: {gesture_id}) - already processed")
            processed_data.append({
                'id': gesture_id,
                'label': gesture_label,
                'safe_label': safe_label,
                'category': category
            })
            continue

        # Delete existing file if reprocessing
        if os.path.exists(output_file) and PROCESS_EXISTING:
            os.remove(output_file)

        # Find all MOV files in this gesture folder
        video_files = glob.glob(os.path.join(gesture_folder, "*.MOV"))

        if not video_files:
            print(f"⚠️ No MOV files found for {gesture_label} (ID: {gesture_id})")
            continue

        print(f"\n🎬 Processing '{gesture_label}' (ID: {gesture_id}, Category: {category})")
        print(f"   Found {len(video_files)} videos")

        # Process all videos for this gesture
        gesture_sequences = []

        for video_file in tqdm(video_files, desc=f"Extracting {gesture_label}", colour="green"):
            # Extract landmarks
            sequence = extract_landmarks_from_video(video_file)

            if sequence is not None:
                # Store original sequence
                gesture_sequences.append(sequence)

                # Add data augmentation with slight noise
                noise_level = 0.005
                noise = np.random.normal(0, noise_level, sequence.shape)
                noisy_sequence = sequence + noise
                gesture_sequences.append(noisy_sequence)

        # Save if we have valid sequences
        if len(gesture_sequences) > 0:
            np.save(output_file, np.array(gesture_sequences))
            print(f"✅ Saved {len(gesture_sequences)} sequences for '{gesture_label}' as {safe_label}.npy")

            processed_data.append({
                'id': gesture_id,
                'label': gesture_label,
                'safe_label': safe_label,
                'category': category
            })
        else:
            print(f"⚠️ No valid sequences extracted for '{gesture_label}'")

        # Cleanup to free memory
        gc.collect()

    # Save metadata for training script
    metadata_dict = {
        'gestures': processed_data,
        'extraction_date': '2025-03-03 06:42:20',
        'num_gestures': len(processed_data),
        'max_frames': MAX_FRAMES,
        'id_to_label': {item['id']: item['label'] for item in processed_data},
        'label_to_file': {item['label']: f"{item['safe_label']}.npy" for item in processed_data}
    }

    np.save(os.path.join(OUTPUT_PATH, "metadata.npy"), metadata_dict)

    # Also save a readable JSON version
    import json
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=4)

    print(f"\n💾 Saved metadata for {len(processed_data)} gestures")

    # Generate a summary table
    print("\n📋 Gesture Processing Summary:")
    print("-" * 90)
    print(f"{'ID':<6} {'Label':<30} {'Filename':<25} {'Category':<15} {'Status':<10}")
    print("-" * 90)

    for item in processed_data:
        status = "Processed" if os.path.exists(os.path.join(OUTPUT_PATH, f"{item['safe_label']}.npy")) else "Failed"
        print(f"{item['id']:<6} {item['label'][:30]:<30} {item['safe_label'][:25]:<25} {item['category'][:15]:<15} {status:<10}")

    print("-" * 90)

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
    print(f"🔢 Processed {NUM_GESTURES} gestures")
    print("=" * 70)

if __name__ == "__main__":
    main()