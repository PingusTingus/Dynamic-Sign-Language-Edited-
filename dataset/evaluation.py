import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load dataset
X_test = np.load("dataset/X.npy")  # Ensure this is the test set
y_test = np.load("dataset/y.npy")  # Ensure this is the test set

# Load trained model
model = tf.keras.models.load_model("gesture_lstm_model.h5")

# Debugging: Print dataset shapes before processing
print(f"📌 Original X_test Shape: {X_test.shape}")
print(f"📌 Original y_test Shape: {y_test.shape}")

# Ensure X_test has the correct shape (num_samples, 30, 63)
num_frames = 30  # Number of frames per sample
num_features = 63  # Features per frame
num_samples = X_test.shape[0] // num_frames  # Calculate correct number of samples

expected_total_elements = num_samples * num_frames * num_features
actual_total_elements = X_test.size

if expected_total_elements != actual_total_elements:
    print(f"❌ ERROR: Expected {expected_total_elements} elements, but found {actual_total_elements}")
    print("Possible causes: Incorrect data collection, feature extraction issues, or wrong array format.")
    exit()

# ✅ Reshape `X_test`
X_test = X_test.reshape(num_samples, num_frames, num_features)
print(f"✅ Reshaped X_test Shape: {X_test.shape}")

# ✅ Fix label encoding before converting to categorical
unique_labels = np.unique(y_test)
label_map = {label: idx for idx, label in enumerate(unique_labels)}  # Ensure labels start at 0

y_test = np.array([label_map[label] for label in y_test[:num_samples]])  # Map labels
num_classes = len(unique_labels)  # Set correct number of classes

# ✅ Convert labels to one-hot encoding
y_test = to_categorical(y_test, num_classes=num_classes)
print(f"✅ One-hot Encoded y_test Shape: {y_test.shape}")

# ✅ Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"✅ Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

from sklearn.metrics import classification_report

# Generate report
print(classification_report(y_test, predicted_classes))
