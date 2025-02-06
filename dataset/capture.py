import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os
import threading
from collections import deque

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,  # Allow two hands
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global Variables
capturing = False
frames = []
frame_count = 30  # Number of frames per gesture
last_landmark = None
capture_active = False
no_hands_start_time = None  # Timer for stopping capture when no hands detected
hand_presence_buffer = 1  # Time (in seconds) to wait before stopping capture
motion_window = deque(maxlen=5)  # Stores last 5 motion values

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

start_button = ttk.Button(root, text="Start Capture", state=tk.NORMAL)
start_button.pack(pady=10)

status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 14, "bold"), bg="white", fg="black")
status_label.pack(pady=10)

# Function to Capture Gesture (Manually Started)
def start_capture():
    global capturing, frames, no_hands_start_time, motion_window
    capturing = True
    frames = []
    no_hands_start_time = None  # Reset timer
    motion_window.clear()  # Reset motion window
    start_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Recording...", fg="red")
    print("🟢 Capture started... Waiting for movement.")

# Function to Stop Gesture Capture (Stops Only After Both Hands Are Gone for Buffer Time)
def stop_capture():
    global capturing
    capturing = False
    start_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Done!", fg="green")
    print("🛑 Capture stopped.")

    # Save Data
    gesture_name = gesture_name_var.get().strip()
    if gesture_name:
        save_gesture(gesture_name)
    else:
        status_label.config(text="Error: Enter Gesture Name!", fg="red")

# Function to Save Gesture Data (Auto-Generates `.npy` Files)
def save_gesture(gesture_name):
    global frames
    dataset_path = f"dataset/gesture_{gesture_name}.npy"

    # ✅ Prevent saving empty captures
    if len(frames) == 0:
        print(f"⚠ Warning: No frames recorded for gesture '{gesture_name}'. Skipping save.")
        status_label.config(text=f"⚠ No frames recorded! Try again.", fg="red")
        return  # Stop execution to prevent errors

    if not os.path.exists("dataset"):
        os.makedirs("dataset")  # Create dataset folder if it doesn't exist

    if os.path.exists(dataset_path):
        existing_data = np.load(dataset_path)

        # ✅ Prevent merging empty arrays
        if existing_data.shape[0] == 0:
            print(f"⚠ Warning: Existing gesture file '{gesture_name}' is empty. Overwriting with new data.")
            np.save(dataset_path, frames)
        else:
            frames = np.vstack((existing_data, frames))  # Append new data
    else:
        print(f"📂 Creating new dataset file for: {gesture_name}")

    np.save(dataset_path, frames)
    status_label.config(text=f"✅ Gesture '{gesture_name}' saved!", fg="blue")
    print(f"✅ Gesture '{gesture_name}' saved to {dataset_path}")


# Function to Check Hand Movement
def hand_moving(current_landmark):
    global last_landmark, motion_window

    if last_landmark is None:
        last_landmark = current_landmark
        return True  # Assume movement at start

    # Compute Euclidean distance between last and current frame
    movement = np.linalg.norm(np.array(current_landmark) - np.array(last_landmark))
    last_landmark = current_landmark

    # Store movement in rolling window
    motion_window.append(movement)

    # Compute average movement over last 5 frames
    avg_movement = sum(motion_window) / len(motion_window)

    return avg_movement > 0.005  # Threshold to determine if the hand is moving

# Function to Update Tkinter Video Feed in a Separate Thread
def update_video():
    global capturing, frames, capture_active, no_hands_start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Error: Cannot read frame from the camera.")
            root.after(33, update_video)  # Retry after 33ms
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_detected = False  # Reset flag each frame
        num_hands_detected = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_detected = True
                num_hands_detected += 1

                # Extract Landmarks
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                flattened_landmarks = np.array(landmarks).flatten()

                if capturing:
                    if hand_moving(flattened_landmarks):
                        frames.append(flattened_landmarks)

        # If both hands disappear, start a buffer timer before stopping capture
        if num_hands_detected == 0:
            if capturing:
                if no_hands_start_time is None:
                    no_hands_start_time = time.time()  # Start countdown
                elif time.time() - no_hands_start_time > hand_presence_buffer:  # Stop if both hands are gone for buffer time
                    print(f"🛑 No hands detected for {hand_presence_buffer} seconds. Stopping capture.")
                    stop_capture()
            else:
                no_hands_start_time = None  # Reset timer when not capturing

        # Print hand detected message only once per detection
        if hand_detected and not capture_active:
            print(f"✋ Hand(s) detected! ({num_hands_detected} hand(s) visible)")
            capture_active = True

        # Convert OpenCV frame to Tkinter-compatible format
        img = Image.fromarray(frame_rgb)
        img = img.resize((500, 350))  # Resize for display
        img_tk = ImageTk.PhotoImage(image=img)

        # FORCE UI UPDATE
        video_label.img_tk = img_tk  # Prevent garbage collection
        video_label.config(image=img_tk)
        root.update_idletasks()
        root.update()

# Bind UI Button to Start Capture
start_button.config(command=start_capture)

# Start Video Feed in a Separate Thread to Prevent Freezing
video_thread = threading.Thread(target=update_video, daemon=True)
video_thread.start()

# Run Tkinter Main Loop
root.mainloop()

# Release Camera
cap.release()
cv2.destroyAllWindows()
