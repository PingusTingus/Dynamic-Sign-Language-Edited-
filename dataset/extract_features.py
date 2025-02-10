import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm  # ✅ Progress Bar Library

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ Dataset Paths
dataset_path = r"C:\Users\Admin\Downloads\FSL-105 A dataset for recognizing 105 Filipino sign language videos\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips"
output_path = "dataset/"
os.makedirs(output_path, exist_ok=True)

# ✅ Extract Existing Gesture Files
existing_files = {os.path.basename(f).replace("gesture_", "").replace(".npy", "") for f in glob.glob(os.path.join(output_path, "gesture_*.npy"))}

# ✅ Extract Gesture Labels
gesture_folders = sorted(glob.glob(os.path.join(dataset_path, "*")))
gesture_labels = {os.path.basename(folder): i for i, folder in enumerate(gesture_folders)}

print("📂 Gesture Labels Assigned:", gesture_labels)

# ✅ Function to Extract Hand Landmarks from Video
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                frame_sequence.append(np.array(landmarks).flatten())  # Flatten to (63,)

    cap.release()

    if len(frame_sequence) < 10:  # Ignore short videos
        return None

    return np.array(frame_sequence)

# ✅ Process and Save Features for **New** Gestures Only
MAX_FRAMES = 50  # Standardized sequence length
new_gestures = [gesture for gesture in gesture_labels if gesture not in existing_files]

if not new_gestures:
    print("✅ No new gestures found. All gestures are already extracted.")
else:
    print(f"🔄 Processing {len(new_gestures)} new gestures...")

for gesture_name in new_gestures:
    gesture_folder = os.path.join(dataset_path, gesture_name)
    video_files = glob.glob(os.path.join(gesture_folder, "*.MOV"))  # Adjust extension if needed
    gesture_sequences = []

    print(f"📌 Extracting '{gesture_name}' ({len(video_files)} videos)...")

    # ✅ Use tqdm for Progress Bar
    for video_file in tqdm(video_files, desc=f"Processing {gesture_name}"):
        sequence = extract_landmarks_from_video(video_file)
        if sequence is not None:
            gesture_sequences.append(sequence)

    if len(gesture_sequences) == 0:
        print(f"⚠ Warning: No valid sequences found for '{gesture_name}'")
        continue

    # ✅ Pad Sequences for Consistency
    padded_sequences = pad_sequences(gesture_sequences, maxlen=MAX_FRAMES, padding="post", dtype="float32")

    # ✅ Save Extracted Features
    np.save(os.path.join(output_path, f"gesture_{gesture_name}.npy"), padded_sequences)
    print(f"✅ Saved {len(padded_sequences)} sequences for '{gesture_name}'")

print("\n🚀 Feature Extraction Completed Successfully!")
