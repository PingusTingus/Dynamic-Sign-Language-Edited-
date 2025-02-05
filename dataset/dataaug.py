import numpy as np
import random

# Load dataset
X = np.load("dataset/X.npy")  # Shape (samples, 30, 63)
y = np.load("dataset/y.npy")  # Shape (samples,)

def augment_data(X, y, num_augmentations=3):
    augmented_X, augmented_y = [], []

    for i in range(len(X)):
        sample = X[i]  # One gesture sample (30 frames, 63 keypoints)

        for _ in range(num_augmentations):
            augmented_sample = sample.copy()

            # Apply Jittering
            noise = np.random.normal(0, 0.01, augmented_sample.shape)  # Small noise
            augmented_sample += noise

            # Apply Scaling
            scale_factor = random.uniform(0.9, 1.1)
            augmented_sample *= scale_factor

            # Apply Rotation
            angle = random.uniform(-5, 5)  # Rotate ±5 degrees
            angle_rad = np.deg2rad(angle)
            rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                        [np.sin(angle_rad), np.cos(angle_rad)]])

            for j in range(0, 63, 3):  # Apply rotation to (x, y) only
                xy = augmented_sample[:, j:j+2]  # Extract (x, y)
                augmented_sample[:, j:j+2] = np.dot(xy, rotation_matrix)

            # Flip Horizontally (50% chance)
            if random.random() > 0.5:
                for j in range(0, 63, 3):
                    augmented_sample[:, j] = 1.0 - augmented_sample[:, j]  # Flip x-axis

            # Store Augmented Data
            augmented_X.append(augmented_sample)
            augmented_y.append(y[i])

    # Convert to NumPy arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    return np.concatenate((X, augmented_X)), np.concatenate((y, augmented_y))

# Apply Data Augmentation
X_augmented, y_augmented = augment_data(X, y, num_augmentations=3)

# Save the new dataset
np.save("dataset/X_augmented.npy", X_augmented)
np.save("dataset/y_augmented.npy", y_augmented)

print(f"✅ Data Augmentation Complete! New dataset size: {X_augmented.shape}")
