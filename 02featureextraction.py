import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224)) 
    preprocessed_frame = preprocess_input(resized_frame)  
    return preprocessed_frame

def extract_spatial_features(frame):
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    spatial_features = model.predict(preprocessed_frame)
    return spatial_features

video_folders = [
    (r"C:\Users\nagen\OneDrive\Desktop\miniproject\dataset\videos\accident", r"C:\Users\nagen\OneDrive\Desktop\miniproject\accident.txt", "accident"),
    (r"C:\Users\nagen\OneDrive\Desktop\miniproject\dataset\videos\noaccident", r"C:\Users\nagen\OneDrive\Desktop\miniproject\noncrash.txt", "noaccident")
]

save_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\outputpath\spatial_features"


for video_folder, annotation_file, label_type in video_folders:
   
    save_folder_video = os.path.join(save_folder, label_type)
    os.makedirs(save_folder_video, exist_ok=True)

    
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    video_names = [line.split(',')[0] for line in lines] 
    labels_list = []
    for line in lines:
        labels_str = ','.join(line.strip().split(',')[1:])
        labels = [int(label) for label in labels_str[1:-1].split(',')] 
        labels_list.append(labels)

    
    for video_name, labels in tqdm(zip(video_names, labels_list)):
        video_path = os.path.join(video_folder, f"{video_name}.mp4")
        save_path = os.path.join(save_folder_video, f"{video_name}_spatial_features.npy")
        labels_path = os.path.join(save_folder_video, f"{video_name}_labels.npy")

        
        if os.path.exists(save_path):
            logger.info(f"Spatial features already exist for {video_name}. Skipping...")
        else:
            cap = cv2.VideoCapture(video_path)
            spatial_features_list = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                
                spatial_features = extract_spatial_features(frame)
                spatial_features_list.append(spatial_features)

            cap.release()

            
            spatial_features_array = np.array(spatial_features_list)

            
            np.save(save_path, spatial_features_array)
            logger.info(f"Spatial features saved for {video_name} at: {save_path}")

        
        np.save(labels_path, np.array(labels))
        logger.info(f"Labels saved for {video_name} at: {labels_path}")