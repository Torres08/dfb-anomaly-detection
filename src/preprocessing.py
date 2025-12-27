
# python src/preprocessing.py
import os
import numpy as np
import librosa
import soundfile as sf

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
DURATION = 3.0      # We analyze 3 seconds of audio
N_MELS = 128        # Height of the spectrogram image
HOP_LENGTH = 517    # Width ≈ 128 pixels

def audio_to_spectrogram(file_path):
    
    """
    1. Loads audio
    2. Converts to Log-Mel Spectrogram (Image)
    3. Normalizes to [0, 1]
    """

    try:
        # 1. Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # 2. Pad or Crop to exact length
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]

        # 3. Convert to Mel Spectrogram, heatmap size: (128, 128)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        
        # 4.  Convert to Log Scale (Decibels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 5. Normalize to [0, 1]
        min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
        norm_spec = (mel_spec_db - min_val) / (max_val - min_val)

        # 6. Fix width rounding errors to ensure 128x128
        if norm_spec.shape[1] != 128:
            norm_spec = librosa.util.fix_length(norm_spec, size=128, axis=1)

        # 7. Add channel dimension (128, 128, 1)
        return norm_spec[..., np.newaxis]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset():
    # Define paths for data
    raw_path = "data/raw/normal"
    output_path = "data/processed/X_train_normal.npy"
    
    # Check if folder has files
    if not os.path.exists(raw_path) or not os.listdir(raw_path):
        print(f" No files found in {raw_path}!")
        print("Please copy your .wav files there first.")
        return

    print(f"Processing audio files from {raw_path}...")
    files = [f for f in os.listdir(raw_path) if f.endswith('.wav')]
    
    data_list = []
    for f in files:
        path = os.path.join(raw_path, f)
        img = audio_to_spectrogram(path)
        if img is not None:
            data_list.append(img)

    if len(data_list) > 0:
        X_data = np.array(data_list)
        np.save(output_path, X_data)
        print(f"Success! :D Saved {X_data.shape} to {output_path}")
    else:
        print("No valid audio files processed :( ")

if __name__ == "__main__":
    process_dataset()