DATASET LINK:https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F?usp=sharing


Accident and Anomaly Detection in Videos
This project focuses on detecting accidents and anomalies in video sequences using deep learning techniques.

Key Components:

3D CNN for accident detection.
ResNet50 for feature extraction.
Isolation Forest and LSTM for anomaly detection.
Steps to Run:

Install dependencies: pip install -r requirements.txt
Train models:
python train_3dcnn.py
python train_isolation_forest.py
python train_lstm.py
Detect accidents: python detect_accident.py --video <path_to_video>
Detect anomalies: python detect_anomaly.py --video <path_to_video>
