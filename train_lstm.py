"""
Filipino Sign Language Model Training
For use with extracted features
Author: PingusTingus
Date: 2025-03-03 07:00:13
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import glob

# --- Configuration ---
INPUT_DIR = "dataset/"  # Directory with extracted feature files
OUTPUT_DIR = "models/"  # Directory to save models and results
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
TEST_SPLIT = 0.2

print("=" * 70)
print("🧠 Filipino Sign Language Model Training")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: 2025-03-03 07:00:13")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load extracted features and create labels"""
    print("\n📂 Loading data from:", INPUT_DIR)

    # Look for all .npy files except metadata.npy
    feature_files = [f for f in glob.glob(os.path.join(INPUT_DIR, "*.npy"))
                     if not f.endswith("metadata.npy")]

    if not feature_files:
        print("❌ No feature files found in", INPUT_DIR)
        return None, None, None

    print(f"Found {len(feature_files)} feature files")

    # Try to load metadata
    metadata = {}
    try:
        if os.path.exists(os.path.join(INPUT_DIR, "metadata.json")):
            with open(os.path.join(INPUT_DIR, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                print("✅ Loaded metadata from JSON file")
        elif os.path.exists(os.path.join(INPUT_DIR, "metadata.npy")):
            metadata = np.load(os.path.join(INPUT_DIR, "metadata.npy"), allow_pickle=True).item()
            print("✅ Loaded metadata from NPY file")
    except Exception as e:
        print(f"⚠️ Warning: Could not load metadata: {e}")
        print("Will create labels from filenames instead.")

    # Load features and create labels
    features = []
    labels = []
    class_names = []

    for i, file_path in enumerate(feature_files):
        filename = os.path.basename(file_path)
        class_name = os.path.splitext(filename)[0]  # Remove .npy extension

        try:
            # Load feature data
            feature_data = np.load(file_path)
            num_samples = len(feature_data)
            print(f"  • Loaded {num_samples} samples from {filename}")

            # Add to features and labels
            features.append(feature_data)
            labels.extend([i] * num_samples)
            class_names.append(class_name)

        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")

    if not features:
        print("❌ No valid feature data found")
        return None, None, None

    # Combine all features and convert to numpy array
    X = np.vstack(features)
    y = np.array(labels)

    print(f"\n📊 Dataset summary:")
    print(f"  • Total samples: {len(X)}")
    print(f"  • Input shape: {X.shape}")
    print(f"  • Number of classes: {len(class_names)}")
    print(f"  • Class names: {class_names}")

    return X, y, class_names

def build_model(input_shape, num_classes):
    """Build an enhanced bidirectional LSTM model"""
    model = Sequential([
        # Input normalization
        BatchNormalization(input_shape=input_shape),

        # LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        BatchNormalization(),

        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.4),

        # Classification layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(X, y, class_names):
    """Train the model and save results"""
    # Convert labels to one-hot encoding
    from tensorflow.keras.utils import to_categorical
    y_one_hot = to_categorical(y, num_classes=len(class_names))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=TEST_SPLIT, random_state=42, stratify=y)

    print(f"\n🔍 Train/Test Split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timestamps, features)

    # Build model
    print("\n🏗️ Building model...")
    model = build_model(input_shape, len(class_names))
    model.summary()

    # Callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train model
    print("\n🏋️‍♂️ Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Get predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))

    # Save model
    print("\n💾 Saving model...")

    # Save as Keras 3 compatible model (.keras format)
    keras_model_path = os.path.join(OUTPUT_DIR, 'fsl_model.keras')
    model.save(keras_model_path)
    print(f"✅ Saved model to {keras_model_path}")

    # Also try to save as H5 format as backup
    try:
        h5_model_path = os.path.join(OUTPUT_DIR, 'fsl_model.h5')
        model.save(h5_model_path)
        print(f"✅ Saved H5 backup to {h5_model_path}")
    except Exception as e:
        print(f"⚠️ Could not save H5 backup: {e}")

    # Save metadata
    results = {
        "gestures": class_names,
        "accuracy": float(test_acc),
        "training_date": "2025-03-03 07:00:13",
        "model_type": "BiLSTM",
        "input_shape": list(input_shape)
    }

    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved metadata to {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    return model, test_acc

def main():
    """Main function for model training"""
    start_time = time.time()

    # Load data
    X, y, class_names = load_data()

    if X is None:
        print("❌ Exiting due to data loading errors.")
        return

    # Train model
    model, accuracy = train_model(X, y, class_names)

    # Print final stats
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print(f"✨ Training completed in {duration:.1f} seconds")
    print(f"📊 Final accuracy: {accuracy:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()