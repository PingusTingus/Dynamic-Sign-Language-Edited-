import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# ✅ **Load the trained model**
model_path = "dataset/gesture_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")
model = load_model(model_path)
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

gesture_names = list(gesture_labels.keys())  # ✅ Convert dict_keys to list

print("\n📂 Label Mapping:", gesture_labels)

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

# ✅ **Ensure Consistent Sequence Lengths**
print("\n✅ Checking Sequence Lengths Before Padding:")
sequence_lengths = [len(seq) for seq in X_test]
print("Unique sequence lengths:", set(sequence_lengths))

# ✅ **Apply Padding for Consistency (Fixes TypeError)**
MAX_FRAMES = 50  # Ensure consistency with training

# Convert X_test to a list of lists before padding
X_test_list = [np.asarray(seq).tolist() for seq in X_test]  # Convert NumPy arrays to Python lists
X_test_padded = pad_sequences(sequences=X_test_list, maxlen=MAX_FRAMES, dtype="float32", padding="post")

print("✅ Padded Sequences Shape:", X_test_padded.shape)

# ✅ **Convert Labels to Categorical**
num_classes = len(gesture_labels)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# ✅ **Evaluate Model**
print("✅ Evaluating Model...")
loss, accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=1)
print(f"\n🎯 Model Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# ✅ **Make Predictions**
y_pred_probs = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ **Ensure y_test is in Label Format**
if len(y_test.shape) > 1:
    y_test = np.argmax(y_test, axis=1)  # Convert from one-hot encoding to labels

# ✅ **Check Shapes for Confusion Matrix**
print("\n✅ Checking Shapes for Confusion Matrix:")
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

# ✅ **Confusion Matrix**
print("✅ Creating Confusion Matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)

# ✅ **Plot Confusion Matrix**
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=gesture_names, yticklabels=gesture_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ **Print Classification Report (Fixes Argument Error)**
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=gesture_names))
