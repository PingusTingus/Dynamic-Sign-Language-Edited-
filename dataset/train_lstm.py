import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load Dataset
X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

# Reshape Data for LSTM Input
X = X.reshape(-1, 30, 63)  # 30 frames per gesture, 21 hand landmarks (x, y, z)
num_classes = len(np.unique(y))

# Convert Labels to Categorical
y = to_categorical(y, num_classes=num_classes)

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 63)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("dataset/gesture_lstm_model.h5")

print("✅ Model training complete! Saved as 'gesture_lstm_model.h5'.")
