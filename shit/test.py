import os
import sys
import keras
import scipy, scipy.io, scipy.signal
import numpy as np
import tensorflow as tf
import librosa
import glob
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000

def load_audio(file_path, target_sr=SAMPLE_RATE):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        audio = librosa.util.normalize(audio)
        return audio, sr

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def plot_single_spectrogram(sample_wav_data):
    # Reshape to add channel dimension (batch_size, time_steps, channels)
    sample_wav_data = sample_wav_data[None, :, None]  # Shape becomes (1, num_samples, 1)
    
    fft_length = 1024
    
    spectrogram = keras.layers.STFTSpectrogram(
        mode="log",
        frame_step=SAMPLE_RATE // 1000,
        frame_length=SAMPLE_RATE * 10 // 1000,
        fft_length=fft_length,
        trainable=False,
    )(sample_wav_data)[0, ...]

    # Get the spectrogram data
    spec_data = spectrogram.numpy().T
        
    # Create the plot with correct frequency scaling
    plt.figure(figsize=(10, 6))
    plt.imshow(spec_data, origin="lower", aspect="auto", extent=[0, spec_data.shape[1], 0, SAMPLE_RATE/2])
    
    # Add frequency ticks (in kHz for better readability)
    y_ticks = np.linspace(0, SAMPLE_RATE/2, 10)
    y_labels = [f"{int(y/1000)}" for y in y_ticks]
    plt.yticks(y_ticks, y_labels)
    
    plt.title("Single Channel Spectrogram")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.show()

plot_single_spectrogram(load_audio("/home/hroudis/strnad-data/BC/-0.452_43.065_2020-07-03_1552_birdnet_mobile_3778446732_recording_0_05.wav")[0])
