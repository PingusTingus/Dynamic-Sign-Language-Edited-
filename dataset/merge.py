import numpy as np
import os

# Initialize empty arrays for all gestures
X = []
y = []

# Gesture Labels (Manually Define)
gesture_labels = {0: "good morning", 1: "good afternoon", 2: "good evening"}

# Load all recorded gestures
for label, gesture_name in gesture_labels.items():
    feature_path = f"dataset/gesture_{gesture_name}.npy"
    label_path = f"dataset/gesture_labels.npy"

    if os.path.exists(feature_path):
        features = np.load(feature_path)
        labels = np.load(label_path)

        X.append(features)
        y.append(labels)

# Convert to NumPy arrays
X = np.vstack(X)
y = np.hstack(y)

# Save the final dataset
np.save("dataset/X.npy", X)
np.save("dataset/y.npy", y)

print(f"✅ Final dataset created: {X.shape}, Labels: {y.shape}")
