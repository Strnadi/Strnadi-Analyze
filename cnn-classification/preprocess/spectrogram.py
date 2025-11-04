import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def mel_spectrogram(samples: np.ndarray, sr: int = 48000, n_fft=1024, hop_length=512, n_mels=128, f_min=3500, f_max=8000, to_db=True):
    """
    Compute a Mel spectrogram (or log-Mel) from a 1D float32 NumPy array (-1..1)
    
    Parameters:
    - samples: 1D numpy array of audio samples (-1..1)
    - sr: sample rate
    - n_fft: FFT window size
    - hop_length: hop length between windows
    - n_mels: number of Mel bands
    - to_db: if True, convert to log scale (dB)
    
    Returns:
    - S: np.ndarray of shape (n_mels, t) containing Mel spectrogram (log if to_db=True)
    """
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=samples,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0  # power=2 for energy, power=1 for amplitude
    )
    
    if to_db:
        S = librosa.power_to_db(S, ref=np.max)
    
    return S



def plot_mel_spect(spect, title="Mel Spectrogram"):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(spect, sr=48000, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
