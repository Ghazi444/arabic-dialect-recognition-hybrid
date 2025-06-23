# print_classification_report.py

"""
Generates and prints a classification report (Accuracy, Precision, Recall, F1-score) for a trained model.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import json

def load_dataset(val_json):
    with open(val_json, "r") as f:
        samples = json.load(f)
    return samples

def main(model_path, features_file, labels_file, val_json):
    model = tf.keras.models.load_model(model_path)
    X_val = np.load(features_file, allow_pickle=True)
    y_val = np.load(labels_file, allow_pickle=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_val)

    if len(X_val.shape) == 2:  # Dense input
        pass
    elif len(X_val.shape) == 3:  # (samples, frames, n_features) → CNN or RNN input
        if len(X_val.shape) == 3 and model.input_shape[-1] == 1:
            X_val = np.expand_dims(X_val, -1)

    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)

    report = classification_report(y_encoded, y_pred_labels, target_names=le.classes_, digits=4)
    print(report)

if __name__ == "__main__":
    # Example usage, edit as needed per model:
    # MFCC + CNN example:
    main(
        model_path="mfcc_cnn_model.h5",
        features_file="mfcc_features.npy",
        labels_file="mfcc_labels.npy",
        val_json="val_dataset.json"
    )
