import librosa
import keras
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm  # progress bar
import argparse

def load_and_normalize_audio(file_path, target_sr=48000):
    """
    Load audio file in various formats and normalize it.
    Returns a 1D numpy array or None on error.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if audio.ndim > 1:
            # if multi-channel, average channels unless second channel is all zeros
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        audio = librosa.util.normalize(audio)
        return audio

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# create spectrogram layer
spect = keras.layers.MelSpectrogram(
    fft_length=2048,
    num_mel_bins=128,
    sampling_rate=48000,
    min_freq=3000,
    max_freq=9000,
    power_to_db=True
)

p = argparse.ArgumentParser(description="Convert WAV files to normalized Mel spectrograms")
p.add_argument("in_dir", nargs="?", default='.', help="input folder (default=current directory)")
p.add_argument("out_dir", nargs="?", default='converted', help="output folder (default=./converted)")
args = p.parse_args()

input_dir = Path(args.in_dir).resolve()
output_dir = Path(args.out_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

wav_files = list(input_dir.glob('*.wav'))  # make a list so we can show total
if not wav_files:
    print("No .wav files found in", input_dir)

# use tqdm to show progress (with filenames)
for wav_file in tqdm(wav_files, desc="Processing WAV files", unit="file"):
    audio = load_and_normalize_audio(wav_file)
    if audio is None:
        # loader already printed an error
        continue

    # MelSpectrogram layer expects batch dimension: (batch, samples)
    audio_batch = np.expand_dims(audio, axis=0).astype(np.float32)

    try:
        mel_tensor = spect(audio_batch)         # returns a tensor-like object
        mel = mel_tensor.numpy()                # convert to numpy
        mel = np.squeeze(mel, axis=0)           # remove batch dimension

        # Standardize the spectrogram
        mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)
    except Exception as e:
        print(f"Error computing spectrogram for {wav_file}: {e}")
        continue

    out_path = output_dir / f"{wav_file.stem}_spect.npy"
    try:
        np.save(out_path, mel)
    except Exception as e:
        print(f"Error saving spectrogram for {wav_file} to {out_path}: {e}")
        continue

