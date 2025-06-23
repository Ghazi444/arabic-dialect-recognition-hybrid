# wavelet_rnn_train.py

"""
Trains a Recurrent Neural Network (LSTM) on flattened Wavelet features for Arabic dialect classification.
This experiment is part of the hybrid architecture comparison.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_wavelet_data(features_file="wavelet_features.npy", labels_file="wavelet_labels.npy"):
    X = np.load(features_file, allow_pickle=True)
    y = np.load(labels_file, allow_pickle=True)

    # Pad feature vectors to make them suitable for LSTM input (as sequences)
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([np.pad(x, (0, max_len - x.shape[0]), mode='constant') for x in X])

    # Reshape for LSTM → (samples, timesteps, features) → treat flattened array as sequence of scalars
    X_padded = np.expand_dims(X_padded, -1)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X_padded, y_categorical, le.classes_

def build_wavelet_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    X, y, class_labels = load_wavelet_data()
    input_shape = X.shape[1:]  # (timesteps, features=1)
    num_classes = y.shape[1]

    model = build_wavelet_rnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train-validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                        validation_data=(X_val, y_val), verbose=1)

    model.save("wavelet_rnn_model.h5")
    print("Training complete. Model saved as wavelet_rnn_model.h5")
