# mfcc_rnn_train.py

"""
Trains a Recurrent Neural Network (RNN, LSTM) on MFCC features for Arabic dialect classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import random

def load_data(mfcc_features_file="mfcc_features.npy", mfcc_labels_file="mfcc_labels.npy"):
    X = np.load(mfcc_features_file, allow_pickle=True)
    y = np.load(mfcc_labels_file, allow_pickle=True)

    # Pad sequences to same length for LSTM input
    max_len = max([mfcc.shape[0] for mfcc in X])
    X_padded = np.array([np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)), mode='constant') for mfcc in X])

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X_padded, y_categorical, le.classes_

def build_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == "__main__":
    X, y, class_labels = load_data()
    input_shape = X.shape[1:]  # (frames, n_mfcc)
    num_classes = y.shape[1]

    model = build_rnn_model(input_shape, num_classes)
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

    model.save("mfcc_rnn_model.h5")
    print("Training complete. Model saved as mfcc_rnn_model.h5")
