import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import glob
import os

# ✅ **Load Dataset**
dataset_path = "dataset/"
gesture_files = glob.glob(os.path.join(dataset_path, "gesture_*.npy"))

# 🔹 **Gesture Label Mapping**
gesture_labels = {}
label_counter = 0

# Create a mapping from gesture names to numerical labels
for file in gesture_files:
    gesture_name = os.path.basename(file).replace("gesture_", "").replace(".npy", "")
    if gesture_name not in gesture_labels:
        gesture_labels[gesture_name] = label_counter
        label_counter += 1

print("\n📂 Label Mapping:")
print(gesture_labels)

X, y = [], []

# ✅ **Load Data**
for file in gesture_files:
    gesture_name = os.path.basename(file).replace("gesture_", "").replace(".npy", "")
    label = gesture_labels[gesture_name]

    data = np.load(file, allow_pickle=True)
    for sample in data:
        X.append(sample)
        y.append(label)

# Convert to NumPy Arrays
X = np.array(X, dtype=object)
y = np.array(y)

# ✅ **Apply Padding to Ensure Equal Frame Lengths**
X_padded = pad_sequences(X, padding="post", dtype="float32")

print(f"\n✅ Training Data Shape: {X_padded.shape}")
print(f"✅ Labels Shape: {y.shape}")

# ✅ **Convert Labels to Categorical**
num_classes = len(gesture_labels)
y_categorical = to_categorical(y, num_classes=num_classes)

# ✅ **Define LSTM Model**
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_padded.shape[1], X_padded.shape[2])),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer with num_classes categories
])

# ✅ **Compile Model**
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ **Train Model**
print("\n🚀 Training Model...")
history = model.fit(X_padded, y_categorical, epochs=50, batch_size=16, validation_split=0.2)

# ✅ **Save Model**
model.save("dataset/gesture_model.h5")
print("\n✅ Model Training Completed and Saved!")
