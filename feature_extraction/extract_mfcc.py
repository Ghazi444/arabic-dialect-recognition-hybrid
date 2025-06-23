# extract_mfcc.py

"""
Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from preprocessed Arabic audio files.
Parameters:
- 13 MFCCs
- 25 ms window (frame_length = 400 samples at 16kHz)
- 10 ms hop length (hop_length = 160 samples at 16kHz)
"""

import librosa
import numpy as np
import json
import os

def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=160, n_fft=400)
    return mfcc.T  # Shape: (frames, n_mfcc)

def extract_mfcc_dataset(input_json, output_features="mfcc_features.npy", output_labels="mfcc_labels.npy"):
    with open(input_json, "r") as f:
        samples = json.load(f)

    features = []
    labels = []

    for sample in samples:
        audio_path = sample['path']
        dialect = sample['dialect']
        try:
            mfcc_matrix = extract_mfcc(audio_path)
            features.append(mfcc_matrix)
            labels.append(dialect)
        except Exception as e:
            print(f"Failed on {audio_path}: {e}")

    np.save(output_features, features, allow_pickle=True)
    np.save(output_labels, labels, allow_pickle=True)
    print(f"Extracted MFCCs for {len(features)} audio samples.")

if __name__ == "__main__":
    extract_mfcc_dataset("train_dataset.json", output_features="mfcc_features.npy", output_labels="mfcc_labels.npy")

