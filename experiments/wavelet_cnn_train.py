# wavelet_cnn_train.py

"""
Trains a Convolutional Neural Network (CNN) on flattened Wavelet features for Arabic dialect classification.
This experiment is designed to compare spatial learning capacity (CNN) on non-spectral input.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_wavelet_data(features_file="wavelet_features.npy", labels_file="wavelet_labels.npy"):
    X = np.load(features_file, allow_pickle=True)
    y = np.load(labels_file, allow_pickle=True)

    # Pad all feature vectors to the same length for Dense layers input
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([np.pad(x, (0, max_len - x.shape[0]), mode='constant') for x in X])

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X_padded, y_categorical, le.classes_

def build_dense_cnn_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    X, y, class_labels = load_wavelet_data()
    input_dim = X.shape[1]
    num_classes = y.shape[1]

    model = build_dense_cnn_model(input_dim, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train-validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)

    model.save("wavelet_cnn_model.h5")
    print("Training complete. Model saved as wavelet_cnn_model.h5")
