import numpy as np
import glob

# Path to dataset folder
dataset_path = "dataset/"

# List all gesture files
gesture_files = glob.glob(dataset_path + "gesture_*.npy")

# Show available gestures
print("\n📂 Available Gestures in Dataset:")
for i, file in enumerate(gesture_files):
    print(f"{i + 1}. {file.split('/')[-1]}")

# Select a gesture to inspect
gesture_name = input("\n🔍 Enter the gesture filename (without '.npy'): ")
file_path = f"{dataset_path}gesture_{gesture_name}.npy"

try:
    data = np.load(file_path, allow_pickle=True)
    print(f"\n📌 Loaded {gesture_name}.npy")
    print(f"Shape: {data.shape}")  # Check the shape of the stored data

    # Check if data is structured as multiple sequences
    if isinstance(data[0], np.ndarray):
        print(f"\n📝 Number of Samples: {len(data)}")
        print(f"🎥 Frames in First Sample: {len(data[0])}")
        print(f"🎯 Features per Frame: {len(data[0][0])}")

        # Print the first frame of the first sample
        print("\n🔍 First Frame of the First Sample:")
        print(data[0][0])  # Print the first frame's landmark data
    else:
        print("⚠ Warning: Data structure might be incorrect.")

except FileNotFoundError:
    print("❌ Error: File not found! Make sure the gesture name is correct.")
except Exception as e:
    print(f"❌ Error: {e}")
