import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ✅ Load Preprocessed Dataset
dataset_path = "dataset/"
gesture_files = glob.glob(os.path.join(dataset_path, "gesture_*.npy"))

gesture_labels = {os.path.basename(f).replace("gesture_", "").replace(".npy", ""): i for i, f in enumerate(gesture_files)}
num_classes = len(gesture_labels)

X, y = [], []
for file in gesture_files:
    gesture_name = os.path.basename(file).replace("gesture_", "").replace(".npy", "")
    label = gesture_labels[gesture_name]
    data = np.load(file, allow_pickle=True)

    for sample in data:
        X.append(sample)
        y.append(label)

X = np.array(X, dtype="object")  # ✅ Convert to object for proper padding
y = np.array(y)

# ✅ Apply Padding
MAX_FRAMES = 50  # Keep consistent with extraction script
X_padded = pad_sequences(X, maxlen=MAX_FRAMES, padding="post", dtype="float32")

# ✅ Convert Labels to Categorical
y_categorical = to_categorical(y, num_classes=num_classes)

# ✅ Split Data (Train 80% / Test 20%)
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

# ✅ Define CNN-LSTM Model
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(MAX_FRAMES, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True, activation="relu")),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# ✅ Compile Model with Optimized Learning Rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Learning Rate Adjustment
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=1)

# ✅ Train Model
print("\n🚀 Training Model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                    callbacks=[lr_reduction])

# ✅ Save Model
model.save("dataset/gesture_model.h5")
print("\n✅ Model Training Completed and Saved!")
