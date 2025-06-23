"""
Splits preprocessed dataset into training and validation sets (80/20 split by default).
"""

import json
import random

def split_dataset(input_json, train_output="train_dataset.json", val_output="val_dataset.json", train_ratio=0.8):
    with open(input_json, "r") as f:
        samples = json.load(f)

    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_set = samples[:split_idx]
    val_set = samples[split_idx:]

    with open(train_output, "w") as f_train:
        json.dump(train_set, f_train, indent=2)

    with open(val_output, "w") as f_val:
        json.dump(val_set, f_val, indent=2)

    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")

if __name__ == "__main__":
    split_dataset("processed_dataset.json")
