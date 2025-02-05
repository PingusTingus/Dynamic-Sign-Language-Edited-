import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os
import threading

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global Variables
capturing = False
frames = []
frame_count = 30  # Number of frames per gesture
last_landmark = None
capture_active = False

# Initialize Tkinter UI
root = tk.Tk()
root.title("Gesture Capture Program")
root.geometry("900x600")
root.configure(bg="white")

# OpenCV Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS fix

# UI Elements
video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

gesture_name_label = tk.Label(root, text="Gesture Name:", font=("Arial", 12), bg="white")
gesture_name_label.pack(pady=5)

gesture_name_var = tk.StringVar()
gesture_name_entry = ttk.Entry(root, textvariable=gesture_name_var, font=("Arial", 12), width=20)
gesture_name_entry.pack(pady=5)

start_button = ttk.Button(root, text="Start Capture")
start_button.pack(pady=10)

stop_button = ttk.Button(root, text="Stop Capture", state=tk.DISABLED)
stop_button.pack(pady=10)

status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 14, "bold"), bg="white", fg="black")
status_label.pack(pady=10)

# Function to Capture Gesture
def start_capture():
    global capturing, frames
    capturing = True
    frames = []
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Recording...", fg="red")

# Function to Stop Gesture Capture
def stop_capture():
    global capturing
    capturing = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Done!", fg="green")

# Function to Update Tkinter Video Feed in a Separate Thread
def update_video():
    global capturing, frames, capture_active

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Error: Cannot read frame from the camera.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert OpenCV frame to Tkinter-compatible format
        img = Image.fromarray(frame_rgb)
        img = img.resize((500, 350))  # Resize for display
        img_tk = ImageTk.PhotoImage(image=img)

        # FORCE UI UPDATE
        video_label.img_tk = img_tk  # Prevent garbage collection
        video_label.config(image=img_tk)
        root.update_idletasks()
        root.update()

# Start Video Feed in a Separate Thread to Prevent Freezing
video_thread = threading.Thread(target=update_video, daemon=True)
video_thread.start()

# Run Tkinter Main Loop
root.mainloop()

# Release Camera
cap.release()
cv2.destroyAllWindows()
