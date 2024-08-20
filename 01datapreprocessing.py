import os
import numpy as np
import cv2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_video(video_path, target_shape=(64, 64)):
    try:
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize an empty list to store preprocessed frames
        preprocessed_frames = []

        while True:
            success, frame = video_cap.read()

            if not success:
                break

            # Resize the frame and apply preprocessing steps (e.g., Gaussian blur, normalization)
            resized_frame = cv2.resize(frame, target_shape[::-1])  # Ensure correct resizing
            blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
            normalized_frame = blurred_frame.astype(np.float32) / 255.0

            # Append the preprocessed frame to the list
            preprocessed_frames.append(normalized_frame)

        video_cap.release()

        return preprocessed_frames, frame_count

    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        return None, None

output_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

non_accident_annotation_file = r"C:\Users\nagen\OneDrive\Desktop\miniproject\noncrash.txt"
accident_annotation_file = r"C:\Users\nagen\OneDrive\Desktop\miniproject\accident.txt"

non_accident_video_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\dataset\videos\noaccident"
accident_video_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\dataset\videos\accident"

video_folders = [(non_accident_video_folder, non_accident_annotation_file, "non_accident"),
                 (accident_video_folder, accident_annotation_file, "accident")]

for video_folder, annotation_file, label_type in video_folders:
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    frames_list = []
    labels_list = []

    for line in tqdm(lines):
        parts = line.strip().split(',')
        vidname = parts[0]
        labels_str = ','.join(parts[1:])
        labels = [int(label) for label in labels_str[1:-1].split(',')]  # Extract labels and convert to integers

        video_path = os.path.join(video_folder, f"{vidname}.mp4")

        # Preprocess the video
        preprocessed_frames, num_frames = preprocess_video(video_path)

        if preprocessed_frames is not None:
            frames_list.append(preprocessed_frames)
            labels_list.append(labels)

    frames = np.concatenate(frames_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"Total number of {label_type} frames: {len(frames)}")
    print(f"Total number of {label_type} labels: {len(labels)}")

    np.save(os.path.join(output_folder, f"{label_type}_preprocessed_frames.npy"), frames)
    np.save(os.path.join(output_folder, f"{label_type}_labels.npy"), labels)