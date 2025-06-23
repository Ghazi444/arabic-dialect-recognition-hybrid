"""
Preprocesses Arabic audio files:
- Resample to 16kHz mono
- Trim silence
- Normalize audio
"""

import librosa
import soundfile as sf
import os
import json

def preprocess_audio_file(input_path, output_path, sr=16000):
    y, _ = librosa.load(input_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y)
    y /= max(abs(y))  # Normalize to [-1, 1]
    sf.write(output_path, y, sr)

def preprocess_dataset(input_json, output_dir):
    with open(input_json, "r") as f:
        samples = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    processed_samples = []

    for sample in samples:
        input_path = sample['audio']
        dialect = sample['dialect']
        filename = os.path.basename(input_path).replace(".mp3", ".wav").replace(".ogg", ".wav")
        output_path = os.path.join(output_dir, filename)
        try:
            preprocess_audio_file(input_path, output_path)
            processed_samples.append({'path': output_path, 'dialect': dialect})
        except Exception as e:
            print(f"Failed on {input_path}: {e}")

    print(f"Processed {len(processed_samples)} files.")
    with open("processed_dataset.json", "w") as f:
        json.dump(processed_samples, f, indent=2)

if __name__ == "__main__":
    preprocess_dataset("filtered_dataset.json", "processed_audio/")
