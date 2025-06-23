# mfcc_cnn_train.py

"""
Trains a Convolutional Neural Network (CNN) on MFCC features for Arabic dialect classification.
Model architecture:
- 3 convolutional layers (3x3 filters, ReLU)
- MaxPooling layers
- Fully connected dense layer
- Softmax output
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import random

def load_data(mfcc_features_file="mfcc_features.npy", mfcc_labels_file="mfcc_labels.npy"):
    X = np.load(mfcc_features_file, allow_pickle=True)
    y = np.load(mfcc_labels_file, allow_pickle=True)

    # Pad MFCC feature matrices to uniform size for CNN input
    max_len = max([mfcc.shape[0] for mfcc in X])
    X_padded = np.array([np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant') for mfcc in X])
    X_padded = np.expand_dims(X_padded, -1)  # Add channel dimension (for Conv2D)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X_padded, y_categorical, le.classes_

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    X, y, class_labels = load_data()
    input_shape = X.shape[1:]  # (frames, n_mfcc, 1)
    num_classes = y.shape[1]

    model = build_model(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train-validation split (80/20)
    split_idx = int(len(X) * 0.8)
    indices = np.arange(len(X))
    random.shuffle(indices)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)

    model.save("mfcc_cnn_model.h5")
    print("Training complete. Model saved as mfcc_cnn_model.h5")

