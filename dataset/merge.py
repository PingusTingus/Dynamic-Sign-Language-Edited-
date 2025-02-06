import numpy as np
import glob
import os

# Path to dataset folder
dataset_path = "dataset/"

# Get all gesture files
gesture_files = glob.glob(os.path.join(dataset_path, "gesture_*.npy"))
data = []
labels = []

# Assign a unique label to each gesture file
gesture_label_map = {gesture_file.split("/")[-1].replace("gesture_", "").replace(".npy", ""): i
                     for i, gesture_file in enumerate(gesture_files)}

print("Gesture Label Mapping:", gesture_label_map)

# Sliding window parameters
frame_length = 30  # Number of frames per sample
stride = 5  # Overlapping step size

for gesture_file in gesture_files:
    gesture_name = gesture_file.split("/")[-1].replace("gesture_", "").replace(".npy", "")
    gesture_data = np.load(gesture_file)
    num_frames = len(gesture_data)

    # Skip files that have fewer than 30 frames (to avoid errors)
    if num_frames < frame_length:
        print(f"⚠ Warning: Skipping {gesture_name} - Only {num_frames} frames available (Need 30).")
        continue

    # Apply sliding window to create overlapping samples
    for i in range(0, num_frames - frame_length + 1, stride):
        sample = gesture_data[i : i + frame_length]  # Get a block of 30 frames
        data.append(sample)
        labels.append(gesture_label_map[gesture_name])

        # Debug: Print how many samples are generated per gesture
        if i == 0:
            print(f"✅ Added samples for {gesture_name}: {num_frames // stride} samples")

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Save the correctly formatted dataset
np.save("dataset/X.npy", X)
np.save("dataset/y.npy", y)

print(f"✅ Dataset merged successfully! Final shapes: X={X.shape}, y={y.shape}")
