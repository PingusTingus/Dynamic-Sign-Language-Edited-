#!/usr/bin/env python3
"""
Enhanced Filipino Sign Language Model Training
Integrates existing LSTM-GRU model with advanced features
Author: PingusTingus
Date: 2025-03-17 06:30
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, GRU, Conv1D,
    Bidirectional, TimeDistributed, GlobalAveragePooling1D,
    Input, Reshape, multiply, add, Activation, concatenate, Flatten
)
from tensorflow.keras.layers import Bidirectional, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import glob
import random
import pandas as pd
import pickle

# --- Configuration ---
INPUT_DIR = "dataset/"             # Original data directory
PROCESSED_DIR = "data/processed/"  # Enhanced processed data
OUTPUT_DIR = "models/enhanced/"    # Directory to save models and results
USE_ENHANCED_DATA = True           # Whether to use the enhanced data if available
EPOCHS = 100                       # Increased epochs since we now have early stopping
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
TEST_SPLIT = 0.2                   # 80% training, 20% testing
VALIDATION_SPLIT = 0.1             # 10% of training data used for validation
RANDOM_SEED = 42
MODEL_ARCHITECTURE = "attention"   # Options: "hybrid", "attention", "lightweight"

print("=" * 70)
print("🧠 Enhanced Filipino Sign Language Model Training")
print(f"🧑‍💻 User: PingusTingus")
print(f"🕒 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🏛️ Model architecture: {MODEL_ARCHITECTURE}")
print(f"🚀 Using enhanced data: {USE_ENHANCED_DATA}")
print("=" * 70)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load extracted features and create labels with proper class balancing"""
    
    # First, try to load the pre-processed sequence data if using enhanced features
    if USE_ENHANCED_DATA:
        sequence_path = os.path.join(PROCESSED_DIR, "sequences.pkl")
        if os.path.exists(sequence_path):
            print(f"\n📂 Loading enhanced sequence data from: {sequence_path}")
            try:
                with open(sequence_path, 'rb') as f:
                    data = pickle.load(f)
                
                X = data['X']
                y = data['y']
                labels = data['labels']
                
                print(f"✅ Loaded {len(X)} pre-processed sequences with shape {X.shape}")
                print(f"✅ Found {len(labels)} classes")
                
                return X, y, labels
            except Exception as e:
                print(f"⚠️ Error loading sequence data: {e}")
                print("Falling back to original data loading method.")
    
    # Otherwise, use the original data loading method
    print(f"\n📂 Loading data from:", INPUT_DIR if not USE_ENHANCED_DATA else PROCESSED_DIR)
    data_dir = INPUT_DIR if not USE_ENHANCED_DATA else PROCESSED_DIR

    # Look for all .npy files except metadata.npy
    feature_files = [f for f in glob.glob(os.path.join(data_dir, "*.npy"))
                     if not f.endswith("metadata.npy")]

    if not feature_files:
        print("❌ No feature files found in", data_dir)
        return None, None, None

    print(f"Found {len(feature_files)} feature files")

    # Try to load metadata
    metadata = {}
    try:
        if os.path.exists(os.path.join(data_dir, "metadata.json")):
            with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                print("✅ Loaded metadata from JSON file")
        elif os.path.exists(os.path.join(data_dir, "metadata.npy")):
            metadata = np.load(os.path.join(data_dir, "metadata.npy"), allow_pickle=True).item()
            print("✅ Loaded metadata from NPY file")
    except Exception as e:
        print(f"⚠️ Warning: Could not load metadata: {e}")
        print("Will create labels from filenames instead.")

    # Load features and create labels
    features = []
    labels = []
    class_names = []
    sample_counts = []

    # First pass - determine class names and collect stats
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        class_name = os.path.splitext(filename)[0]  # Remove .npy extension
        class_names.append(class_name)

        # Get sample count for this class
        try:
            feature_data = np.load(file_path)
            sample_counts.append(len(feature_data))
        except:
            sample_counts.append(0)

    # Create a dataframe to help us analyze class distribution
    class_df = pd.DataFrame({
        'class_name': class_names,
        'sample_count': sample_counts
    })

    # Sort by sample count to see imbalance
    class_df_sorted = class_df.sort_values(by='sample_count', ascending=False)
    print("\n📊 Class distribution:")
    print(class_df_sorted)

    # Calculate stats
    mean_samples = class_df['sample_count'].mean()
    median_samples = class_df['sample_count'].median()
    min_samples = class_df['sample_count'].min()
    max_samples = class_df['sample_count'].max()

    print(f"\nClass statistics:")
    print(f"  • Mean samples per class: {mean_samples:.1f}")
    print(f"  • Median samples per class: {median_samples:.1f}")
    print(f"  • Min samples: {min_samples}")
    print(f"  • Max samples: {max_samples}")
    print(f"  • Max/Min ratio: {max_samples/min_samples if min_samples > 0 else 'N/A':.1f}")

    # Reset for second pass
    features = []
    labels = []
    class_names = []

    # Second pass - load the data
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

    # Add small random noise to features to improve generalization
    np.random.seed(RANDOM_SEED)
    noise_level = 0.002  # Very subtle noise
    X = X + np.random.normal(0, noise_level, X.shape)

    print(f"  • Added subtle random noise for generalization (level: {noise_level})")

    return X, y, class_names

def check_feature_dimensions(features):
    """Check if features were extracted with one or two hands"""
    first_feature = features[0][0]  # First sample, first frame
    num_landmarks = len(first_feature)

    print(f"Feature vector size: {num_landmarks}")

    # Check if we have velocity features (enhanced features)
    has_velocity_features = False
    
    if USE_ENHANCED_DATA and num_landmarks % 6 == 0:  # For 21 landmarks with xyz and velocities
        has_velocity_features = True
        print("✅ Detected enhanced features with velocity information")

    if num_landmarks > 100:  # Likely two hands (each hand has 21 landmarks x 3 coordinates)
        print("✅ Detected features from BOTH hands")
        return True, has_velocity_features
    else:
        print("ℹ️ Detected features from ONE hand only")
        return False, has_velocity_features

def build_hybrid_model(input_shape, num_classes, both_hands=True, has_velocity=False):
    """Build the original hybrid LSTM-GRU model with attention mechanism"""
    # Determine if we're using features from both hands
    feature_size = input_shape[1]  # Number of features per timestep
    
    # Adjust model complexity based on feature size
    lstm_units = 128 if both_hands else 64
    
    # Create a model with attention mechanism
    inputs = keras.Input(shape=input_shape)
    
    # Normalize inputs
    x = BatchNormalization()(inputs)
    
    # First bidirectional LSTM layer
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    
    # GRU layer (better for gesture sequences)
    x = Bidirectional(GRU(lstm_units, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # Attention mechanism (simple attention using time-distributed dense)
    attention = TimeDistributed(Dense(1, activation='tanh'))(x)
    attention = keras.layers.Flatten()(attention)
    attention_weights = keras.layers.Activation('softmax')(attention)
    
    # Apply attention
    attention_weights = keras.layers.RepeatVector(lstm_units * 2)(attention_weights)
    attention_weights = keras.layers.Permute([2, 1])(attention_weights)
    
    # Weighted average using attention weights
    merged = keras.layers.Multiply()([x, attention_weights])
    merged = GlobalAveragePooling1D()(merged)
    
    # Classification layers
    x = Dense(lstm_units, activation='relu')(merged)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_enhanced_attention_model(input_shape, num_classes, both_hands=True, has_velocity=False):
    """Build an enhanced model with advanced attention mechanism"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization,
        Reshape, multiply, add, Activation, concatenate, Conv1D
    )
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Normalize inputs
    x = BatchNormalization()(inputs)
    
    # Add convolutional feature extraction
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Split into position and velocity streams if we have enhanced features
    if has_velocity:
        # Extract position and velocity features
        position_stream = Bidirectional(LSTM(96, return_sequences=True))(x)
        position_stream = Dropout(0.3)(position_stream)
        
        velocity_stream = Bidirectional(LSTM(64, return_sequences=True))(x)
        velocity_stream = Dropout(0.3)(velocity_stream)
        
        # Combine streams
        x = concatenate([position_stream, velocity_stream])
    else:
        # Standard bidirectional LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
    
    # Self-attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = Reshape((input_shape[0], 1))(attention)
    
    # Apply attention
    attended = multiply([x, attention])
    
    # Global context
    context = GlobalAveragePooling1D()(attended)
    
    # Dense layers
    x = Dense(128, activation='relu')(context)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lightweight_model(input_shape, num_classes, both_hands=True, has_velocity=False):
    """Build lightweight model for resource-constrained devices"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, SimpleRNN, Dropout, BatchNormalization,
        GlobalAveragePooling1D
    )
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Normalize inputs
    x = BatchNormalization()(inputs)
    
    # Reduce dimensionality first for efficiency
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Simple RNN layers (more efficient than LSTM/GRU)
    x = SimpleRNN(48, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = SimpleRNN(32, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Output layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, y, class_names):
    """Train the model and save results"""
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Check if we're using features from one or both hands and if we have velocity features
    both_hands, has_velocity = check_feature_dimensions(X)

    # Convert labels to one-hot encoding
    from tensorflow.keras.utils import to_categorical
    y_one_hot = to_categorical(y, num_classes=len(class_names))

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)

    print(f"\n🔍 Train/Test Split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Testing samples: {len(X_test)}")

    # Get input shape
    input_shape = X_train.shape[1:]  # (timestamps, features)

    # Build model based on selected architecture
    print(f"\n🏗️ Building {MODEL_ARCHITECTURE} model...")

    if MODEL_ARCHITECTURE == "hybrid":
        model = build_hybrid_model(input_shape, len(class_names), both_hands, has_velocity)
    elif MODEL_ARCHITECTURE == "attention":
        model = build_enhanced_attention_model(input_shape, len(class_names), both_hands, has_velocity)
    elif MODEL_ARCHITECTURE == "lightweight":
        model = build_lightweight_model(input_shape, len(class_names), both_hands, has_velocity)
    else:
        print(f"⚠️ Unknown model architecture: {MODEL_ARCHITECTURE}, using hybrid model")
        model = build_hybrid_model(input_shape, len(class_names), both_hands, has_velocity)

    model.summary()

    # Callbacks for training
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.keras')
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
            patience=7,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]

    # Train model
    print("\n🏋️‍♂️ Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,  # Use portion of training data for validation
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    print("\n📊 Evaluating model on holdout test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Get predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices (both raw and normalized)
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xticks(rotation=90, ha='right')

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=90, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    print(f"✅ Saved confusion matrix to {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

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
    print(f"✅ Saved training history to {os.path.join(OUTPUT_DIR, 'training_history.png')}")

    # Save model
    print("\n💾 Saving model...")

    # Save as Keras 3 compatible model (.keras format)
    keras_model_path = os.path.join(OUTPUT_DIR, 'fsl_model.keras')
    model.save(keras_model_path)
    print(f"✅ Saved model to {keras_model_path}")

    # Save as h5 format for maximum compatibility
    h5_model_path = os.path.join(OUTPUT_DIR, 'best_model.h5')
    try:
        model.save(h5_model_path)
        print(f"✅ Saved H5 model to {h5_model_path}")
    except Exception as e:
        print(f"⚠️ Could not save H5 model: {e}")

    # Convert to TFLite format for Raspberry Pi deployment
    print("\n🔄 Converting to TFLite format for Raspberry Pi...")

    tflite_model_path = os.path.join(OUTPUT_DIR, 'model.tflite')
    try:
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply optimization flags for LSTM models
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # This allows LSTM ops
        ]

        # Convert the model
        tflite_model = converter.convert()

        # Save the model
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ Saved TFLite model to {tflite_model_path}")
        print(f"   Model size: {os.path.getsize(tflite_model_path) / (1024 * 1024):.2f} MB")

        # Create a more efficient quantized version for better Pi performance
        print("\n🔄 Creating quantized TFLite model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Enable more aggressive integer quantization
        try:
            # Try full integer quantization first
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            # Generate representative dataset for quantization
            def representative_dataset():
                # Use a small subset of training data (30 samples or 10%, whichever is smaller)
                num_samples = min(30, len(X_train) // 10)
                for i in range(num_samples):
                    sample = X_train[i:i + 1].astype(np.float32)
                    yield [sample]

            converter.representative_dataset = representative_dataset

            # Convert to quantized model
            quantized_tflite_model = converter.convert()
            quantized_model_path = os.path.join(OUTPUT_DIR, 'model_int8.tflite')
            with open(quantized_model_path, 'wb') as f:
                f.write(quantized_tflite_model)

            print(f"✅ Saved integer-quantized TFLite model to {quantized_model_path}")
            print(f"   Model size: {os.path.getsize(quantized_model_path) / (1024 * 1024):.2f} MB")
        except Exception as e:
            print(f"⚠️ Error creating int8 quantized model: {e}")

            try:
                print("   Trying float16 quantization instead...")
                # Try float16 quantization as fallback
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

                quantized_tflite_model = converter.convert()
                quantized_model_path = os.path.join(OUTPUT_DIR, 'model_float16.tflite')
                with open(quantized_model_path, 'wb') as f:
                    f.write(quantized_tflite_model)

                print(f"✅ Saved float16-quantized TFLite model to {quantized_model_path}")
                print(f"   Model size: {os.path.getsize(quantized_model_path) / (1024 * 1024):.2f} MB")
            except Exception as e2:
                print(f"⚠️ Error creating float16 quantized model: {e2}")
                print("   Falling back to standard quantization...")

                # Try with standard float quantization as last resort
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                quantized_tflite_model = converter.convert()

                quantized_model_path = os.path.join(OUTPUT_DIR, 'model_quantized.tflite')
                with open(quantized_model_path, 'wb') as f:
                    f.write(quantized_tflite_model)

                print(f"✅ Saved standard-quantized TFLite model to {quantized_model_path}")
                print(f"   Model size: {os.path.getsize(quantized_model_path) / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"⚠️ Error converting to TFLite: {e}")
        print("   You can convert later using the optimize_for_pi.py script")

    # Save class mapping for real-time inference
    class_mapping = {
        "classes": class_names,
        "index_to_class": {str(i): name for i, name in enumerate(class_names)},
        "class_to_index": {name: str(i) for i, name in enumerate(class_names)}
    }

    with open(os.path.join(OUTPUT_DIR, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=4)

    print(f"✅ Saved class mapping to {os.path.join(OUTPUT_DIR, 'class_mapping.json')}")

    # Save metadata and evaluation results
    results = {
        "gestures": class_names,
        "accuracy": float(test_acc),
        "training_date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "model_architecture": MODEL_ARCHITECTURE,
        "input_shape": [int(dim) for dim in input_shape],
        "uses_both_hands": both_hands,
        "uses_velocity_features": has_velocity,
        "trained_by": "PingusTingus",
        "confusion_matrix": {
            "raw": cm.tolist(),
            "class_names": class_names
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved metadata to {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print(report)

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Find most confused class pairs
    misclassified = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                misclassified.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': int(cm[i, j])
                })

    # Sort by count
    misclassified.sort(key=lambda x: x['count'], reverse=True)

    # Show top confused pairs
    if misclassified:
        print("\nTop confused gesture pairs:")
        print("-" * 50)
        print(f"{'True':<20} {'Predicted':<20} {'Count'}")
        print("-" * 50)
        for i in range(min(10, len(misclassified))):
            item = misclassified[i]
            print(f"{item['true']:<20} {item['predicted']:<20} {item['count']}")

    return model, test_acc


def main():
    """Main function for model training"""
    start_time = time.time()

    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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

    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print("\n" + "=" * 70)
    print(f"✨ Training completed in {hours}h {minutes}m {seconds}s")
    print(f"📊 Final accuracy: {accuracy:.4f}")
    print(f"📋 Model architecture: {MODEL_ARCHITECTURE}")
    print(f"💾 Models and evaluation results saved to {OUTPUT_DIR}")
    print("=" * 70)

    # Print current time
    print(f"🕒 Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Check for command line arguments
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced FSL Model Training')
    parser.add_argument('--input-dir', default=PROCESSED_DIR,
                        help=f'Directory with processed data (default: {PROCESSED_DIR})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help=f'Directory to save model (default: {OUTPUT_DIR})')
    parser.add_argument('--model-type', choices=['hybrid', 'attention', 'lightweight'],
                        default=MODEL_ARCHITECTURE,
                        help=f'Model architecture to use (default: {MODEL_ARCHITECTURE})')
    parser.add_argument('--use-original-data', action='store_true',
                        help='Use original data instead of enhanced data')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Maximum training epochs (default: {EPOCHS})')

    args = parser.parse_args()

    # Update config based on args
    PROCESSED_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MODEL_ARCHITECTURE = args.model_type
    USE_ENHANCED_DATA = not args.use_original_data
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run main function
    main()