"""
Test script to verify the audio-to-spectrogram conversion works correctly.
"""
import os
import numpy as np

# Check if the spectrogram files exist
SAVE_DIR = 'MELD_spectrograms/'

print("Checking existing spectrogram files...")
print("-" * 70)

files_to_check = [
    'X_audio_train.npy',
    'X_audio_val.npy', 
    'X_audio_test.npy'
]

for filename in files_to_check:
    filepath = os.path.join(SAVE_DIR, filename)
    if os.path.exists(filepath):
        data = np.load(filepath)
        print(f"✓ {filename}")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
        print()
    else:
        print(f"✗ {filename} - NOT FOUND")
        print()

# Check audio directories exist
print("\nChecking audio source directories...")
print("-" * 70)

audio_dirs = {
    'Train': 'MELD.Raw/train_splits',
    'Validation': 'MELD.Raw/dev_splits_complete',
    'Test': 'MELD.Raw/output_repeated_splits_test'
}

for name, path in audio_dirs.items():
    if os.path.exists(path):
        mp4_files = [f for f in os.listdir(path) if f.endswith('.mp4') and not f.startswith('._')]
        print(f"✓ {name}: {path}")
        print(f"  Contains {len(mp4_files)} MP4 files")
    else:
        print(f"✗ {name}: {path} - NOT FOUND")
    print()

print("\nAll checks complete!")
