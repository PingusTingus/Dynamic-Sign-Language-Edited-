import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

# Debugging: Print the shape and unique values
print(f"📌 Original X Shape: {X.shape}")  # Expected (num_samples * 30, 63)
print(f"📌 Original y Shape: {y.shape}, Unique Labels: {np.unique(y)}")  # Expected (num_samples,)

# Define expected parameters
num_frames = 30  # 30 frames per gesture
num_features = 63  # 63 features per frame
num_samples = X.shape[0] // num_frames  # Calculate the number of gestures

# Validate total elements match expected shape
expected_total_elements = num_samples * num_frames * num_features
actual_total_elements = X.size

if expected_total_elements != actual_total_elements:
    print(f"❌ ERROR: Expected {expected_total_elements} elements, but found {actual_total_elements}")
    print("Possible causes: Incorrect data collection or feature extraction issues.")
    exit()

# Reshape X into (num_samples, num_frames, num_features)
X = X.reshape(num_samples, num_frames, num_features)
print(f"✅ Reshaped X Shape: {X.shape}")

# Fix: Normalize label indices (Ensure 0-based indexing)
unique_labels = np.unique(y)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y])  # Map labels to 0,1,2,...

# Fix: Convert labels to categorical properly
num_classes = len(unique_labels)
y = to_categorical(y[:num_samples], num_classes=num_classes)  # Match labels with X
print(f"✅ Reshaped y Shape: {y.shape}, Num Classes: {num_classes}")

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(num_frames, num_features), return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

# Save the model
model.save("dataset/gesture_lstm_model.h5")
print("✅ Model saved successfully!")
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
