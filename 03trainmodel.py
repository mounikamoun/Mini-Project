import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import os

output_folder = r"C:\Users\nagen\OneDrive\Desktop\miniproject\output"

# Load preprocessed data for accident videos
accident_train_frames = np.load(os.path.join(output_folder, "accident_preprocessed_frames.npy"))
accident_train_labels = np.load(os.path.join(output_folder, "accident_labels.npy"))

# Load preprocessed data for non-accident videos
non_accident_train_frames = np.load(os.path.join(output_folder, "non_accident_preprocessed_frames.npy"))
non_accident_train_labels = np.load(os.path.join(output_folder, "non_accident_labels.npy"))

# Combine frames and labels
train_frames = np.concatenate([accident_train_frames, non_accident_train_frames], axis=0)
train_labels = np.concatenate([accident_train_labels, non_accident_train_labels], axis=0)

# Split data into training and validation sets
train_frames, val_frames, train_labels, val_labels = train_test_split(train_frames, train_labels, test_size=0.2, random_state=42)

print(f"Number of training frames: {len(train_frames)}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of validation frames: {len(val_frames)}")
print(f"Number of validation labels: {len(val_labels)}")

# Check the shape of training frames
print(f"Shape of training frames: {train_frames.shape}")

# Reshape frames to 5D tensor
train_frames = np.expand_dims(train_frames, axis=-1)
val_frames = np.expand_dims(val_frames, axis=-1)

# Define 3D CNN model
model = models.Sequential([
    layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=train_frames.shape[1:], padding='same'),
    layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_frames, train_labels, batch_size=32, epochs=10, validation_data=(val_frames, val_labels))

# Evaluate the model on the validation data
test_loss, test_accuracy = model.evaluate(val_frames, val_labels)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
model_path = os.path.join(output_folder, "3d_cnn.h5")

# Save the model
model.save(model_path)

print("Model saved successfully at:", model_path) 