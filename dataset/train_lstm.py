import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Load Existing Model (if available)
model_path = "dataset/gesture_lstm_model.h5"

try:
    model = load_model(model_path)
    print(f"✅ Loaded existing model: {model_path}")
except:
    print("⚠️ No existing model found. Training a new model instead.")
    model = None

# Load Updated Dataset (Augmented)
X = np.load("dataset/X_augmented.npy")  # Ensure dataset is preprocessed
y = np.load("dataset/y_augmented.npy")

# Model Parameters
sequence_length = 30  # Number of frames per gesture
num_features = 63  # Each frame has 21 keypoints (x, y, z)
num_classes = len(np.unique(y))  # Automatically detect gesture classes

# Reshape Data for LSTM Input
X = X.reshape(-1, sequence_length, num_features)

# Convert Labels to Categorical
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# If no existing model, create a new one
if model is None:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    print("🔄 Created a new model.")

# Compile Model (Lower Learning Rate for Fine-Tuning)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Define Early Stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Fine-Tune Model with New Data
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Overwrite Existing Model with Fine-Tuned Version
model.save(model_path)
print(f"✅ Model fine-tuned and saved at {model_path}!")
