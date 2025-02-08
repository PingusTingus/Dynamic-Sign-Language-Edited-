import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import glob
import os

# ✅ **Load the trained model**
model = load_model("dataset/gesture_model.h5")
print("✅ Model Loaded Successfully!")

# ✅ **Load dataset for evaluation**
dataset_path = "dataset/"
gesture_files = glob.glob(os.path.join(dataset_path, "gesture_*.npy"))

# 🔹 **Gesture Label Mapping**
gesture_labels = {}
label_counter = 0

for file in gesture_files:
    gesture_name = os.path.basename(file).replace("gesture_", "").replace(".npy", "")
    if gesture_name not in gesture_labels:
        gesture_labels[gesture_name] = label_counter
        label_counter += 1

print("\n📂 Label Mapping:")
print(gesture_labels)

# ✅ **Load Evaluation Data**
X_test, y_test = [], []

for file in gesture_files:
    gesture_name = os.path.basename(file).replace("gesture_", "").replace(".npy", "")
    label = gesture_labels[gesture_name]

    data = np.load(file, allow_pickle=True)
    for sample in data:
        X_test.append(sample)
        y_test.append(label)

# Convert to NumPy Arrays
X_test = np.array(X_test, dtype=object)
y_test = np.array(y_test)

# ✅ **Apply Padding for Consistency**
X_test_padded = pad_sequences(X_test, padding="post", dtype="float32")

# ✅ **Convert Labels to Categorical**
num_classes = len(gesture_labels)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# ✅ **Evaluate Model**
loss, accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=1)
print(f"\n🎯 Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# ✅ **Make Predictions**
y_pred_probs = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ **Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)

# ✅ **Plot Confusion Matrix**
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=gesture_labels.keys(), yticklabels=gesture_labels.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ **Print Classification Report**
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=gesture_labels.keys()))
