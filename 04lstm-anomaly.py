import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten

# Define constants
output_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\output"
num_frames = 1 
height, width, channels = 64, 64, 3  # Dimensions of each frame
epochs = 10
batch_size = 32

# Load preprocessed frames
accident_frames = np.load(os.path.join(output_folder, "accident_preprocessed_frames.npy"))
non_accident_frames = np.load(os.path.join(output_folder, "non_accident_preprocessed_frames.npy"))

# Load accident and non-accident labels
accident_labels = np.load(os.path.join(output_folder, "accident_labels.npy"))
non_accident_labels = np.load(os.path.join(output_folder, "non_accident_labels.npy"))

# Concatenate frames and labels
all_frames = np.concatenate([accident_frames, non_accident_frames], axis=0)
all_labels = np.concatenate([np.ones(len(accident_frames)), np.zeros(len(non_accident_frames))], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_frames, all_labels, test_size=0.2, random_state=42)

# Reshape data to sequences of frames
X_train = X_train.reshape(-1, num_frames, height, width, channels)
X_test = X_test.reshape(-1, num_frames, height, width, channels)

# Define model architecture
model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=(num_frames, height, width, channels)),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D((2, 2), padding='same')),
    TimeDistributed(Flatten()),
    LSTM(32, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using sequences of frames
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the model
model.save(os.path.join(output_folder, "lstm-anomaly_detection_model.h5"))