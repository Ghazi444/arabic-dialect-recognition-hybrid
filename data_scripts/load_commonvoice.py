"""
Loads and filters the Arabic subset of Mozilla Common Voice dataset.
Filters by country metadata to group into dialect classes:
Egyptian, Levantine, and Gulf Arabic.
"""

from datasets import load_dataset
import json

def load_and_filter_commonvoice():
    dataset = load_dataset("mozilla-foundation/common_voice_12_0", "ar", split="train")

    country_to_dialect = {
        "Egypt": "Egyptian",
        "Jordan": "Levantine", "Palestine": "Levantine",
        "Lebanon": "Levantine", "Syria": "Levantine",
        "Saudi Arabia": "Gulf", "UAE": "Gulf", "Qatar": "Gulf", "Kuwait": "Gulf"
    }

    filtered_samples = []
    for item in dataset:
        country = item.get('locale', None)
        if country in country_to_dialect and item.get('path') and item.get('audio'):
            filtered_samples.append({
                'audio': item['audio']['path'],   # full path to audio file
                'dialect': country_to_dialect[country]
            })

    print(f"Total filtered samples: {len(filtered_samples)}")
    return filtered_samples

if __name__ == "__main__":
    samples = load_and_filter_commonvoice()
    # Save to JSON for later use
    with open("filtered_dataset.json", "w") as f:
        json.dump(samples, f, indent=2)
