# Arabic Dialect Recognition — Hybrid Deep Learning and Signal Processing

This repository contains the complete implementation of a research project investigating hybrid models for Arabic dialect classification using combinations of classical signal processing techniques and deep learning architectures. The experiments explore Mel-Frequency Cepstral Coefficients (MFCC) and Discrete Wavelet Transform (DWT) features in conjunction with Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

The research evaluates four hybrid configurations:
- MFCC + CNN
- MFCC + RNN
- Wavelet + CNN
- Wavelet + RNN

## 📁 Project Structure

arabic-dialect-recognition-hybrid/
├── data_scripts/ # Dataset loading, filtering, preprocessing, and splitting
│ ├── load_commonvoice.py
│ ├── preprocess_audio.py
│ └── split_train_val.py
├── feature_extraction/ # Feature extraction scripts
│ ├── extract_mfcc.py
│ └── extract_wavelet.py
├── experiments/ # Experiment scripts for training each model
│ ├── mfcc_cnn_train.py
│ ├── mfcc_rnn_train.py
│ ├── wavelet_cnn_train.py
│ └── wavelet_rnn_train.py
├── evaluation/ # Evaluation scripts (classification report + confusion matrices)
│ ├── generate_confusion_matrices.py
│ └── print_classification_report.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation

csharp
Copy
Edit

## 📦 Installation Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
Tested with Python ≥ 3.8.

🗂 Dataset
The experiments use the Arabic subset of Common Voice 12.0, a large, crowd-sourced multilingual speech dataset. Dialects are assigned based on speaker country metadata:

Egyptian Arabic → Egypt

Levantine Arabic → Jordan, Palestine, Lebanon, Syria

Gulf Arabic → Saudi Arabia, UAE, Qatar, Kuwait

The dataset was filtered, resampled to 16 kHz, and normalized before feature extraction. Approx. 6 hours of speech were used across the three dialect groups.

🧰 Usage Workflow
1️⃣ Load and filter dataset:

bash
python data_scripts/load_commonvoice.py
2️⃣ Preprocess audio files (resample, normalize, trim):

bash
python data_scripts/preprocess_audio.py
3️⃣ Split dataset into training and validation sets:

bash
python data_scripts/split_train_val.py
4️⃣ Extract features:

bash
python feature_extraction/extract_mfcc.py      # for MFCC-based models
python feature_extraction/extract_wavelet.py   # for Wavelet-based models
5️⃣ Run experiments (train models):


python experiments/mfcc_cnn_train.py
python experiments/mfcc_rnn_train.py
python experiments/wavelet_cnn_train.py
python experiments/wavelet_rnn_train.py
6️⃣ Evaluate models:

bash

python evaluation/generate_confusion_matrices.py
python evaluation/print_classification_report.py
📄 Citation
If you use this code in your academic work, please cite the associated paper or this repository.

📧 Contact
Developed by Ghazal Shwayat
GitHub: https://github.com/YOUR_GITHUB_USERNAME
