# generate_confusion_matrices.py

"""
Generates and saves confusion matrix for a trained model on validation data.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

def load_dataset(val_json):
    with open(val_json, "r") as f:
        samples = json.load(f)
    return samples

def load_labels(samples):
    return [sample['dialect'] for sample in samples]

def main(model_path, features_file, labels_file, val_json, output_image="confusion_matrix.png"):
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

    cm = confusion_matrix(y_encoded, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {model_path}")
    plt.savefig(output_image)
    print(f"Confusion matrix saved to {output_image}")

if __name__ == "__main__":
    # Example usage, edit as needed per model:
    # MFCC + CNN example:
    main(
        model_path="mfcc_cnn_model.h5",
        features_file="mfcc_features.npy",
        labels_file="mfcc_labels.npy",
        val_json="val_dataset.json",
        output_image="confusion_matrix_mfcc_cnn.png"
    )
