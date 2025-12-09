# MELD Emotion Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)

A comprehensive multimodal emotion recognition system using the MELD (Multimodal EmotionLines Dataset). This project implements multiple deep learning approaches to classify emotions from conversational dialogue, combining text and audio modalities for improved performance. The main code can be found in `final_code.ipynb`.

## Overview

This repository contains implementations of various neural network architectures for emotion recognition:

- **LSTM Model**: Text-based emotion classification using contextual embeddings
- **1D CNN Model**: Audio-based emotion classification using pre-computed audio embeddings
- **Fusion Model**: Multimodal approach combining text and audio features
- **2D CNN Model**: Spectrogram-based audio emotion recognition (experimental)

## Features

- **Multimodal Learning**: Combines text and audio modalities for robust emotion recognition
- **Multiple Architectures**: LSTM, CNN, and fusion models for comprehensive comparison
- **Spectrogram Generation**: Automated pipeline for converting raw audio to 2D spectrograms
- **Comprehensive Evaluation**: Per-class performance analysis and model comparison
- **Production Ready**: Inference scripts for real-world deployment

## Dataset

The project uses the **MELD (Multimodal EmotionLines Dataset)**, which contains:

- **13,000+ utterances** from Friends TV show dialogues
- **7 emotion classes**: neutral, joy, surprise, anger, sadness, disgust, fear
- **Multimodal data**: Text transcripts and audio clips
- **Emotion distribution**: Heavily imbalanced (neutral ~47%, minority classes ~2-3%)

### Data Splits

- **Train**: 9,989 utterances
- **Validation**: 1,109 utterances
- **Test**: 2,610 utterances

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/clow427/MELD-emotion-recognition.git
cd MELD-emotion-recognition

# Install dependencies
pip install -r requirements.txt

# For spectrogram generation (optional)
pip install librosa tqdm
```

### Dataset Setup

1. Download the MELD dataset from the official source
2. Extract to `MELD.Raw/` directory with the following structure:
   ```
   MELD.Raw/
   ├── train_sent_emo.csv
   ├── dev_sent_emo.csv
   ├── test_sent_emo.csv
   ├── text_emotion.pkl
   ├── audio_emotion.pkl
   ├── train_splits/
   ├── dev_splits_complete/
   └── output_repeated_splits_test/
   ```

## Models

### 1. LSTM Model (Text-Only)

- **Architecture**: Bidirectional LSTM with attention
- **Input**: Pre-computed contextual text embeddings
- **Performance**: Baseline text-based classification

### 2. 1D CNN Model (Audio-Only)

- **Architecture**: 1D Convolutional Neural Network
- **Input**: Pre-computed audio embeddings (300D)
- **Features**: Multiple conv blocks with batch normalization and dropout

### 3. Fusion Model (Multimodal)

- **Architecture**: Combined LSTM + 1D CNN branches
- **Fusion**: Late fusion with dense layers
- **Input**: Text embeddings + Audio embeddings
- **Performance**: State-of-the-art multimodal results

### 4. 2D CNN Model (Spectrogram-Based)

- **NOT IMPLEMENTED**
- **Architecture**: 2D Convolutional Neural Network
- **Input**: Mel spectrograms (128×128×1)
- **Features**: Raw audio processing pipeline

## Usage

### Training Models

```python
# Run the main training notebook
jupyter notebook "final code.ipynb"
```

### Inference

```python
# Load trained models
from tensorflow.keras.models import load_model

fusion_model = load_model('best_fusion_model.keras')

# Make predictions
predictions = fusion_model.predict([text_input, audio_input])
```

### Spectrogram Generation

```python
# Generate spectrograms from raw audio
jupyter notebook "audio_to_spectrogram.ipynb"
```

## Project Structure

```
MELD-emotion-recognition/
├── MELD.Raw/                    # Raw dataset (not included)
│   ├── train_sent_emo.csv
│   ├── dev_sent_emo.csv
│   ├── test_sent_emo.csv
│   ├── text_emotion.pkl
│   └── audio_emotion.pkl
├── MELD_spectrograms/           # Generated spectrograms
│   ├── X_audio_train.npy
│   ├── X_audio_val.npy
│   ├── X_audio_test.npy
│   └── README.md
├── audio_to_spectrogram.ipynb   # Spectrogram generation
├── final code.ipynb            # Main training notebook
├── FINAL_revised_Another_copy_of_code.ipynb
├── fusion_inference.py          # Inference script
├── test_spectrograms.py         # Testing utilities
├── best_*.keras                 # Trained model weights
├── requirements.txt
├── LICENSE
└── README.md
```

## Acknowledgments

- **MELD Dataset**: Poria et al. "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations"
- **TensorFlow/Keras**: For the deep learning framework
- **Librosa**: For audio processing capabilities
