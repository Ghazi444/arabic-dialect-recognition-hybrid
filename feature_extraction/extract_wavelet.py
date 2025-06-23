# extract_wavelet.py

"""
Extracts Discrete Wavelet Transform (DWT) coefficients from preprocessed Arabic audio files.
Parameters:
- Wavelet: Daubechies-4 ('db4')
- Decomposition Level: 3
"""

import pywt
import numpy as np
import librosa
import json
import os

def extract_wavelet(audio_path, wavelet='db4', level=3, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    # Flatten coefficients into a 1D feature vector
    flattened = np.hstack([c.flatten() for c in coeffs])
    return flattened

def extract_wavelet_dataset(input_json, output_features="wavelet_features.npy", output_labels="wavelet_labels.npy"):
    with open(input_json, "r") as f:
        samples = json.load(f)

    features = []
    labels = []

    for sample in samples:
        audio_path = sample['path']
        dialect = sample['dialect']
        try:
            feature_vector = extract_wavelet(audio_path)
            features.append(feature_vector)
            labels.append(dialect)
        except Exception as e:
            print(f"Failed on {audio_path}: {e}")

    features = np.array(features, dtype=object)
    labels = np.array(labels, dtype=object)

    np.save(output_features, features, allow_pickle=True)
    np.save(output_labels, labels, allow_pickle=True)
    print(f"Extracted Wavelet features for {len(features)} audio samples.")

if __name__ == "__main__":
    extract_wavelet_dataset("train_dataset.json", output_features="wavelet_features.npy", output_labels="wavelet_labels.npy")

