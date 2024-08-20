import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.ensemble import IsolationForest
import joblib
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Load the trained models
cnn_model = load_model("3d_cnn.h5")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
iso_forest_model = joblib.load("isolationforest-anomaly_detection_model.h5")

# Load the LSTM model for accident detection
lstm_model = load_model("lstm-anomaly_detection_model.h5")

# Function to preprocess a video for the 3D CNN model
def preprocess_video_cnn(video_path, target_shape=(64, 64)):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preprocessed_frames = []

        while True:
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
            normalized_frame = blurred_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()
        preprocessed_frames = np.expand_dims(preprocessed_frames, axis=-1)
        return np.array(preprocessed_frames), frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

# Function to preprocess a video for the ResNet model
def preprocess_video_resnet(video_path, target_shape=(224, 224)):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preprocessed_frames = []

        while True:
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()
        preprocessed_frames = np.array(preprocessed_frames)
        preprocessed_frames = preprocess_input(preprocessed_frames)
        return preprocessed_frames, frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

# Function to preprocess a video for the LSTM model
def preprocess_video_lstm(video_path, target_shape=(64, 64), num_frames=16):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_rate = int(video_cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the number of frames to consider for LSTM
        target_frames = min(num_frames, frame_count)

        # Initialize an empty list to store preprocessed frames
        preprocessed_frames = []

        # Read frames and preprocess
        for _ in range(target_frames):
            success, frame = video_cap.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, target_shape[::-1])
            blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
            normalized_frame = blurred_frame.astype(np.float32) / 255.0
            preprocessed_frames.append(normalized_frame)

        video_cap.release()

        # Pad frames if necessary to make num_frames
        while len(preprocessed_frames) < num_frames:
            preprocessed_frames.append(np.zeros(target_shape[::-1] + (3,)))

        preprocessed_frames = np.array(preprocessed_frames)
        preprocessed_frames = np.expand_dims(preprocessed_frames, axis=0)
        return preprocessed_frames, frame_count

    except Exception as e:
        print(f"Error preprocessing video: {e}")
        return None, None

# Function to test a video for accident detection using all models
def test_video(video_path):
    # Preprocess the video for the 3D CNN model
    preprocessed_frames_cnn, num_frames_cnn = preprocess_video_cnn(video_path)
    cnn_predictions = cnn_model.predict(preprocessed_frames_cnn)
    
    # Preprocess the video for the ResNet model
    preprocessed_frames_resnet, num_frames_resnet = preprocess_video_resnet(video_path)
    resnet_features = resnet_model.predict(preprocessed_frames_resnet)

    # Preprocess the video for the LSTM model
    preprocessed_frames_lstm, num_frames_lstm = preprocess_video_lstm(video_path)

    # Predict using the LSTM model
    lstm_prediction = lstm_model.predict(preprocessed_frames_lstm)

    # Detect anomalies using the Isolation Forest model
    resnet_predictions = iso_forest_model.predict(resnet_features)

    # Combining the predictions from all models
    cnn_accident = np.any(cnn_predictions > 0.5)
    lstm_accident = lstm_prediction > 0.5
    resnet_accident = -1 in resnet_predictions

    # If any model detects an accident, we consider it an accident
    if cnn_accident or lstm_accident or resnet_accident:
        result_label.config(text="Accident Detected", fg="red", font=("Helvetica", 16, "bold"))
    else:
        result_label.config(text="No Accident Detected", fg="green", font=("Helvetica", 16, "bold"))

# GUI function to select a video file
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        update_label(file_path)
        test_video(file_path)
        play_video_loop(file_path)

# Create the main application window
root = tk.Tk()
root.title("Accident Detection")
root.geometry("800x650")

# Create a Frame for video display
video_frame = tk.Frame(root, width=640, height=480, bg="black")
video_frame.pack(pady=20)

# Create a Canvas for displaying video
canvas = tk.Canvas(video_frame, width=640, height=480, bg="black")
canvas.pack()

# Create a label to display file path
label = tk.Label(root, text="Selected File: ", font=("Helvetica", 14))
label.pack()

# Create a label to display accident detection result
result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"))
result_label.pack(pady=10)

# Create a button to select a file
button = tk.Button(root, text="Select Video File", font=("Helvetica", 14), command=select_file)
button.pack()

# Function to update the label with selected file path
def update_label(file_path):
    label.config(text="Selected File: " + file_path)

# Function to play video on canvas with frame timing
def play_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            root.update()
            imgtk.__del__()

            # Wait for the appropriate amount of time to maintain frame rate
            delay = int(1000 / frame_rate)
            root.after(delay)

        else:
            break

    cap.release()

# Function to play video in loop
def play_video_loop(file_path):
    while True:
        play_video(file_path)

# Run the main event loop
root.mainloop()