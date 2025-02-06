import numpy as np
import glob
import os

dataset_path = "dataset/"
gesture_files = glob.glob(os.path.join(dataset_path, "gesture_*.npy"))

for gesture_file in gesture_files:
    gesture_name = gesture_file.split("/")[-1].replace("gesture_", "").replace(".npy", "")
    gesture_data = np.load(gesture_file)
    print(f"📝 Gesture: {gesture_name} - Frames: {len(gesture_data)}")
