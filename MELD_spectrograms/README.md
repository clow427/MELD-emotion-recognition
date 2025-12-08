# MELD Audio Spectrograms

This directory contains pre-computed 2D Mel spectrograms for the MELD (Multimodal EmotionLines Dataset) emotion recognition project. These spectrograms are used as input features for the CNN model in the multimodal emotion recognition pipeline.

## Files

- `X_audio_train.npy` - Training set spectrograms
- `X_audio_val.npy` - Validation set spectrograms
- `X_audio_test.npy` - Test set spectrograms

## Purpose

The spectrograms serve as the audio modality input for the multimodal emotion recognition system. They are processed through a 2D Convolutional Neural Network (CNN) to extract audio features that complement the text-based features extracted by the LSTM model.

### Key Characteristics:

- **Format**: 2D Mel spectrograms (128×128×1)
- **Duration**: Fixed 3-second audio clips
- **Sample Rate**: 16kHz
- **Features**: 128 Mel frequency bins
- **Time Steps**: 128 (padded/truncated for consistency)

## Generation Process

Spectrograms are generated using the `audio_to_spectrogram.ipynb` notebook. This process ensures perfect alignment between audio features and emotion labels.

### Prerequisites

Before running the spectrogram generation:

1. **MELD Dataset**: Download and extract the MELD dataset to `MELD.Raw/` directory
2. **Audio Files**: Ensure audio files are in the correct directories:
   - Train: `MELD.Raw/train_splits/`
   - Validation: `MELD.Raw/dev_splits_complete/`
   - Test: `MELD.Raw/output_repeated_splits_test/`
3. **Text Embeddings**: `MELD.Raw/text_emotion.pkl` must exist
4. **Labels**: CSV files must be present:
   - `MELD.Raw/train_sent_emo.csv`
   - `MELD.Raw/dev_sent_emo.csv`
   - `MELD.Raw/test_sent_emo.csv`

### Required Dependencies

```bash
pip install librosa numpy pandas scikit-learn tqdm
```

### Step-by-Step Generation

1. **Open the notebook**:

   ```bash
   jupyter notebook audio_to_spectrogram.ipynb
   ```

2. **Run Cell 1**: Load text embeddings and extract utterance IDs

   - Loads `MELD.Raw/text_emotion.pkl`
   - Processes CSV label files
   - Creates aligned ID lists for train/val/test splits

3. **Run Cell 2**: Generate spectrograms
   - Processes audio files in order specified by ID lists
   - Converts each audio file to 128×128 Mel spectrogram
   - Handles missing files by inserting zero spectrograms
   - Maintains perfect alignment with emotion labels

### Processing Details

**Audio Processing Parameters:**

- **Sample Rate**: 16,000 Hz
- **Duration**: 3.0 seconds (truncated/padded)
- **Mel Bins**: 128
- **Time Steps**: 128 (fixed width)
- **Normalization**: Min-max scaling (0-1)

**File Naming Convention:**

- Audio files: `dia{DIALOGUE_ID}_utt{UTTERANCE_ID}.mp4`
- Supports both `.mp4` and `.wav` formats

**Error Handling:**

- Missing audio files → Zero spectrograms (maintains alignment)
- Corrupted files → Zero spectrograms
- AppleDouble files (.\_\*) → Automatically filtered out

## Data Format

Each `.npy` file contains a NumPy array with shape:

```
(samples, 128, 128, 1)
```

Where:

- `samples`: Number of utterances in the split
- `128`: Mel frequency bins
- `128`: Time steps
- `1`: Single channel (grayscale spectrogram)

## Integration with Main Pipeline

These spectrograms are loaded directly in `FINAL_revised_Another_copy_of_code.ipynb`:

```python
# Load spectrograms
spectrogram_path = 'MELD_spectrograms'
X_audio_train = np.load(os.path.join(spectrogram_path, 'X_audio_train.npy'))
X_audio_val = np.load(os.path.join(spectrogram_path, 'X_audio_val.npy'))
X_audio_test = np.load(os.path.join(spectrogram_path, 'X_audio_test.npy'))
```

The spectrograms are then fed into the CNN model, which processes them alongside text embeddings in the fusion model for multimodal emotion recognition.

## Important Notes

- **Alignment Critical**: Spectrograms must be generated using the exact same ID lists as the text processing to ensure label alignment
- **Memory Usage**: Each spectrogram is ~16KB, so large datasets require sufficient RAM
- **Reproducibility**: The process uses fixed random seeds for consistent results
- **Storage**: Consider compressing the `.npy` files if storage space is limited

## Expected Output

After successful generation, you should see output similar to:

```
SPECTROGRAM GENERATION COMPLETE
======================================================================

Final Shapes:
  Train: (9989, 128, 128, 1)
  Val:   (1109, 128, 128, 1)
  Test:  (2610, 128, 128, 1)
```
