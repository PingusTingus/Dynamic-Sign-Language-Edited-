import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model
model = tf.keras.models.load_model("dataset/gesture_lstm_model.h5")

# Load Test Data
X_test = np.load("dataset/X.npy")
y_test = np.load("dataset/y.npy")

# Reshape X_test for LSTM
X_test = X_test.reshape(X_test.shape[0], 30, 63)

# Predict Classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print Classification Report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["hello", "help", "i'm sick"], yticklabels=["hello", "help", "i'm sick"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
