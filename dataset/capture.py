import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import threading
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import padding function

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Global Variables
capturing = False
frames = []
dataset_path = "dataset/"
no_hands_start_time = None  # Timer for stopping capture when hands disappear
hand_presence_buffer = 1  # Time (in seconds) to wait before stopping capture
gesture_name = None  # Stores current gesture name

# Ensure dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize Tkinter UI
root = tk.Tk()
root.title("Gesture Capture Program")
root.geometry("900x600")
root.configure(bg="white")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# UI Elements
video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

gesture_name_label = tk.Label(root, text="Gesture Name:", font=("Arial", 12), bg="white")
gesture_name_label.pack(pady=5)

gesture_name_var = tk.StringVar()
gesture_name_entry = ttk.Entry(root, textvariable=gesture_name_var, font=("Arial", 12), width=20)
gesture_name_entry.pack(pady=5)

status_label = tk.Label(root, text="Status: Waiting for Hand...", font=("Arial", 14, "bold"), bg="white", fg="black")
status_label.pack(pady=10)

# Function to Start Capture Automatically
def start_capture():
    global capturing, frames, gesture_name
    if capturing:
        return  # Avoid starting again if already capturing

    gesture_name = gesture_name_var.get().strip()
    if not gesture_name:
        status_label.config(text="Error: Enter Gesture Name!", fg="red")
        return

    capturing = True
    frames = []
    status_label.config(text="Status: Recording...", fg="red")
    print("🟢 Capture started... Recording entire gesture.")

# Function to Stop Capture Automatically
def stop_capture():
    global capturing
    if not capturing:
        return  # Prevent stopping if not capturing

    capturing = False
    status_label.config(text="Status: Gesture Recorded!", fg="green")
    print(f"🛑 Capture stopped. Recorded {len(frames)} frames.")

    # Save Data
    if gesture_name:
        save_gesture(gesture_name)
    else:
        status_label.config(text="Error: Enter Gesture Name!", fg="red")

# Function to Save Entire Gesture
def save_gesture(gesture_name):
    global frames
    dataset_file = f"{dataset_path}gesture_{gesture_name}.npy"

    if len(frames) < 10:  # Ignore very short captures
        print(f"⚠ Not enough frames ({len(frames)}) to create a valid sample. Discarding.")
        return

    frames = np.array(frames)

    # Check if file exists and load existing data
    if os.path.exists(dataset_file):
        existing_data = np.load(dataset_file, allow_pickle=True)
        existing_data = list(existing_data)
        existing_data.append(frames)
    else:
        existing_data = [frames]

    # Pad all sequences to match the longest sequence
    max_length = max(len(seq) for seq in existing_data)
    padded_data = pad_sequences(existing_data, maxlen=max_length, padding="post", dtype="float32")

    # Save the padded dataset
    np.save(dataset_file, padded_data)

    status_label.config(text=f"✅ Gesture '{gesture_name}' saved!", fg="blue")
    print(f"✅ Gesture '{gesture_name}' saved to {dataset_file} (Max Length: {max_length} frames)")

# Function to Update Video Feed & Detect Hands Automatically
def update_video():
    global capturing, frames, no_hands_start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Error: Cannot read frame from the camera.")
            root.after(33, update_video)
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_detected = False
        num_hands_detected = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_detected = True
                num_hands_detected += 1

        if hand_detected:
            print(f"✋ Hand detected! ({num_hands_detected} hand(s) visible)")
            status_label.config(text="Status: Hand Detected! Recording...", fg="orange")

            if not capturing:
                start_capture()  # Auto-start when hand is detected

            # Extract and store landmarks if capturing
            if capturing:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                flattened_landmarks = np.array(landmarks).flatten()
                frames.append(flattened_landmarks)
                print(f"📸 Frame captured. Total frames: {len(frames)}")

        # If hands disappear, start countdown to stop capture
        if not hand_detected:
            if capturing:
                if no_hands_start_time is None:
                    no_hands_start_time = time.time()
                elif time.time() - no_hands_start_time > hand_presence_buffer:
                    print(f"🛑 No hands detected for {hand_presence_buffer} seconds. Stopping capture.")
                    stop_capture()
                    status_label.config(text="Status: Waiting for Hand...", fg="black")
            else:
                no_hands_start_time = None  # Reset timer if not capturing

        # Convert OpenCV frame to Tkinter-compatible format
        img = Image.fromarray(frame_rgb)
        img = img.resize((500, 350))
        img_tk = ImageTk.PhotoImage(image=img)

        video_label.img_tk = img_tk
        video_label.config(image=img_tk)
        root.update_idletasks()
        root.update()

# Start Video Feed in a Separate Thread
video_thread = threading.Thread(target=update_video, daemon=True)
video_thread.start()

# Run Tkinter Main Loop
root.mainloop()

# Release Camera
cap.release()
cv2.destroyAllWindows()
