import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MultiHeadAttention, LayerNormalization, Input, Add
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os

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

X = np.array(X, dtype="object")
y = np.array(y)

MAX_FRAMES = 50
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_FRAMES, padding="post", dtype="float32")

y_categorical = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

input_layer = Input(shape=(MAX_FRAMES, X_train.shape[2]))

x = Conv1D(128, kernel_size=5, activation="relu", padding="same")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = LSTM(256, return_sequences=True, activation="tanh")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(query=x, value=x, key=x)
x = Add()([attention_output, x])  # Residual Connection
x = LayerNormalization()(x)

x = LSTM(128, return_sequences=True, activation="tanh")(x)
x = LSTM(64, activation="tanh")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = AdamW(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

print("\n🚀 Training Model...")
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[lr_reduction, early_stopping])

model.save("dataset/gesture_model.h5")
print("\n✅ Training Completed with Optimized Model!")
