import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


spatial_features_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\outputpath\spatial_features"
model_save_path = r"C:\Users\nagen\OneDrive\Desktop\miniproject\isolationforest-anomaly_detection_model.h5"


video_folders = ["accident", "noaccident"]
all_features = []
all_labels = []

for folder in video_folders:
    folder_path = os.path.join(spatial_features_folder, folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_spatial_features.npy"):
            features = np.load(os.path.join(folder_path, file_name))
            label_file = file_name.replace("_spatial_features.npy", "_labels.npy")
            label_path = os.path.join(folder_path, label_file)
            if os.path.exists(label_path):
                labels = np.load(label_path)
                all_features.append(features)
                all_labels.append(labels)
            else:
                print(f"Labels file not found for: {file_name}")


if all_features and all_labels:
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
else:
    print("No spatial features and labels found. Please check the data paths.")

print("Shape of all_features:", all_features.shape)
print("Shape of all_labels:", all_labels.shape)


if all_features.shape[0] > 0 and all_labels.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
else:
    print("Insufficient data for training and testing.")


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


if 'X_train' in locals() and X_train.shape[0] > 0 and 'X_test' in locals() and X_test.shape[0] > 0:
    model = IsolationForest(contamination=0.1)  # Adjust contamination parameter as needed
    model.fit(X_train)

   
    joblib.dump(model, model_save_path)
    print("Model saved successfully at:", model_save_path)

   
    y_pred = model.predict(X_test)

    
    y_pred_binary = np.where(y_pred == 1, 0, 1)

    
    accident_indices = np.where(y_test == 0)[0]
    non_accident_indices = np.where(y_test == 1)[0]

    accuracy_accident = accuracy_score(y_test[accident_indices], y_pred_binary[accident_indices])
    accuracy_non_accident = accuracy_score(y_test[non_accident_indices], y_pred_binary[non_accident_indices])

    
    print("Accuracy for Accident class:", accuracy_accident)
    print("Accuracy for Non-Accident class:", accuracy_non_accident)

    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))
else:
    print("Insufficient data for training the model.")