"""
plot_keras_melspec.py

Example: compute a Mel spectrogram with tf.keras.layers.MelSpectrogram
and plot it with matplotlib using correct time (s) and frequency (Hz) scales.

Author: (example)
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

# optional: librosa for loading audio files
try:
    import librosa
except Exception:
    librosa = None


def hz_to_mel(f_hz):
    # HTK formula used by tf.keras (and many libraries): mel(f) = 2595 * log10(1 + f/700)
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def mel_to_hz(mel):
    # inverse of the HTK formula
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_mel_spectrogram(
    audio,
    *args,
    **kwargs
):
    """Builds a Keras MelSpectrogram layer, runs it on `audio` and returns
    (mel_spectrogram_numpy, params_dict).

    `audio` should be a 1-D numpy array of samples (float32/float64).
    """
    if max_freq is None:
        max_freq = kwargs['sampling_rate'] / 2.0

    # create the layer
    layer = keras.layers.MelSpectrogram(
        *args,
        **kwargs
    )

    # the layer accepts either (samples,) or (batch, samples)
    audio_tf = tf.convert_to_tensor(audio, dtype=tf.float32)
    if audio_tf.ndim == 1:
        out = layer(audio_tf)  # shape (num_mel_bins, time)
    else:
        out = layer(audio_tf)  # shape (batch, num_mel_bins, time)

    mel_spec = out.numpy()
    # if batched, take first example for plotting
    if mel_spec.ndim == 3:
        mel_spec = mel_spec[0]

    return mel_spec


def plot_mel_spectrogram(mel_spec, params, figsize=(10, 4), cmap="magma"):
    """
    mel_spec: 2D numpy array shaped (num_mel_bins, time_frames)
    params: dict returned from compute_mel_spectrogram
    """
    sr = params["sampling_rate"]
    hop = params["sequence_stride"]
    n_mels = params["num_mel_bins"]
    fmin = params["min_freq"]
    fmax = params["max_freq"]
    power_to_db = params["power_to_db"]

    # time axis (seconds)
    n_frames = mel_spec.shape[1]
    frame_times = (np.arange(n_frames) * hop) / float(sr)
    t_start = frame_times[0]
    t_end = frame_times[-1] + (hop / float(sr))  # extent end

    # Y axis: convert mel bin centers to Hz for tick labels
    # Compute mel scale endpoints and linearly spaced mel centers
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_centers = np.linspace(mel_min, mel_max, n_mels)
    freq_centers_hz = mel_to_hz(mel_centers)

    # Use imshow with extent so axes show seconds and Hz
    extent = [t_start, t_end, freq_centers_hz[0], freq_centers_hz[-1]]

    plt.figure(figsize=figsize)
    im = plt.imshow(
        mel_spec,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    # colorbar label
    if power_to_db:
        cbar_label = "Amplitude (dB)"
    else:
        cbar_label = "Amplitude"

    cbar = plt.colorbar(im, format="%+2.0f")
    cbar.set_label(cbar_label)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    # Nice Hz ticks: pick a handful of mel positions and show corresponding Hz
    n_ticks = 6
    mel_tick_vals = np.linspace(mel_min, mel_max, n_ticks)
    hz_tick_vals = mel_to_hz(mel_tick_vals)

    # place ticks at those Hz values
    plt.yticks(hz_tick_vals, [f"{hz:.0f}" for hz in hz_tick_vals])

    # Optionally add a secondary y-axis that shows approximate mel values
    ax = plt.gca()
    secax = ax.secondary_yaxis(
        "right", functions=(hz_to_mel, mel_to_hz)
    )  # converts between Hz <-> Mel (approx)
    secax.set_ylabel("Mel")
    # set mel ticks roughly corresponding to the same positions (in mel units)
    mel_tick_labels = [f"{m:.0f}" for m in mel_tick_vals]
    secax.set_yticks(hz_tick_vals)  # ticks placed at same Hz positions
    secax.set_yticklabels(mel_tick_labels)

    title = "Mel Spectrogram ({} mel bins)".format(n_mels)
    if power_to_db:
        title += " — dB"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def example_from_file(path, target_sr=16000):
    if librosa is None:
        raise ImportError(
            "librosa not installed. Install with `pip install librosa` or pass a numpy array instead."
        )
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def example_sine(duration_s=2.0, sr=16000, freq=440.0):
    t = np.linspace(0.0, duration_s, int(np.round(duration_s * sr)), endpoint=False)
    audio = 0.5 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return audio, sr


if __name__ == "__main__":
    # Choose: load from file (uncomment) OR create synthetic sine wave.
    audio, sr = example_from_file("/home/hroudis/strnad-data/BC/-0.452_43.065_2020-07-03_1552_birdnet_mobile_3778446732_recording_0_05.wav", target_sr=16000)
    # audio, sr = example_sine(duration_s=3.0, sr=16000, freq=440.0)

    # compute mel spectrogram
    mel_spec, params = compute_mel_spectrogram(
        audio,
        sampling_rate=sr,
        fft_length=2048,
        sequence_stride=512,
        num_mel_bins=128,
        min_freq=20.0,
        max_freq=None,  # defaults to sr/2
        power_to_db=True,
        top_db=80.0,
    )

    # plot
    plot_mel_spectrogram(mel_spec, params)
