import numpy as np
import os
from collections import Counter

# Gesture Labels (Make sure this matches what you used in `capture.py`)
gesture_labels = {0: "good morning", 1: "good afternoon", 2: "good evening"}

# Initialize dataset arrays
X, y = [], []

for label, gesture_name in gesture_labels.items():
    filename = f"dataset/gesture_{gesture_name}.npy"

    if os.path.exists(filename):
        data = np.load(filename)
        X.append(data)  # Append gesture samples
        y.append(np.full((data.shape[0],), label))  # Assign labels

        print(f"✅ Loaded '{gesture_name}' - {data.shape}")
    else:
        print(f"⚠️ Warning: {filename} not found!")

# Convert to NumPy arrays
X = np.vstack(X)  # Merge all gesture data
y = np.hstack(y)  # Merge labels

# Save the merged dataset
np.save("dataset/X.npy", X)
np.save("dataset/y.npy", y)

# Print label distribution
print(f"🎯 Dataset Created - X shape: {X.shape}, y shape: {y.shape}")
print(f"📊 Label distribution: {Counter(y)}")
