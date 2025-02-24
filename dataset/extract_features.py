import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

reprocess_existing = True

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

dataset_path = r"C:\Users\Admin\Downloads\TESTING\FSL-105 A dataset for recognizing 105 Filipino sign language videos\clips"
output_path = "dataset/"
os.makedirs(output_path, exist_ok=True)

existing_files = {os.path.basename(f).replace("gesture_", "").replace(".npy", "") for f in glob.glob(os.path.join(output_path, "gesture_*.npy"))}

gesture_folders = sorted(glob.glob(os.path.join(dataset_path, "*")))
gesture_labels = {os.path.basename(folder): i for i, folder in enumerate(gesture_folders)}

print("📂 Gesture Labels Assigned:", gesture_labels)

def extract_landmarks_from_video(video_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    last_valid_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                landmarks = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
                frame_sequence.append(landmarks)
                last_valid_landmarks = landmarks
        else:
            if last_valid_landmarks is not None:
                frame_sequence.append(last_valid_landmarks)

    cap.release()

    if len(frame_sequence) < 10:
        return None

    frame_sequence = adjust_sequence_length(frame_sequence, max_frames)

    return np.array(frame_sequence, dtype=np.float32)


def adjust_sequence_length(sequence, max_frames):
    sequence = np.array(sequence)
    current_length = len(sequence)

    if current_length == max_frames:
        return sequence

    elif current_length > max_frames:
        indices = np.linspace(0, current_length - 1, max_frames).astype(int)
        return sequence[indices]

    else:
        indices = np.linspace(0, current_length - 1, max_frames).astype(int)
        interpolated_sequence = np.zeros((max_frames, sequence.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            interpolated_sequence[i] = sequence[idx]
        return interpolated_sequence


MAX_FRAMES = 50

if reprocess_existing:
    gestures_to_process = gesture_labels.keys()
    print("🔄 Re-extracting ALL gestures...")
else:
    gestures_to_process = [gesture for gesture in gesture_labels if gesture not in existing_files]
    if not gestures_to_process:
        print("✅ No new gestures found. All gestures are already extracted.")
    else:
        print(f"🔄 Processing {len(gestures_to_process)} new gestures...")

for gesture_name in gestures_to_process:
    gesture_folder = os.path.join(dataset_path, gesture_name)
    video_files = glob.glob(os.path.join(gesture_folder, "*.MOV"))
    gesture_sequences = []

    output_file = os.path.join(output_path, f"gesture_{gesture_name}.npy")
    if reprocess_existing and os.path.exists(output_file):
        os.remove(output_file)

    print(f"📌 Extracting '{gesture_name}' ({len(video_files)} videos)...")

    for video_file in tqdm(video_files, desc=f"Processing {gesture_name}", colour="green"):
        sequence = extract_landmarks_from_video(video_file, MAX_FRAMES)
        if sequence is not None:
            gesture_sequences.append(sequence)

    if len(gesture_sequences) == 0:
        print(f"⚠ Warning: No valid sequences found for '{gesture_name}'")
        continue

    np.save(output_file, np.array(gesture_sequences, dtype=np.float32))
    print(f"✅ Saved {len(gesture_sequences)} sequences for '{gesture_name}'")

print("\n🚀 Feature Extraction Completed Successfully!")
