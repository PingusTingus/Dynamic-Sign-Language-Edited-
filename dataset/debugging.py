import numpy as np

sample_gesture = "hello"
sample_file = f"dataset/gesture_{sample_gesture}.npy"
data = np.load(sample_file, allow_pickle=True)

# Check the first frame's shape
print(f"Shape of first frame: {data[0].shape}")  # Should be (63,)
