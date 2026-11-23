"""Keep artificial silence out of the model's input (research log section 17.3).

Real recordings never contain digital silence, but decoded files do: an MP3 decoder's delay at the
start, zero-padded excerpts, clips padded to a fixed length. On a spectrogram such a stretch is a
black block, and a black block after a song reads as "no ending". An explanation map of a wrong
prediction pointed straight at one.

Two repairs, used by the training loader (training/data.py) and by the service
(serving/pipeline.py), so that both see the same thing:

    fill_zero_runs(waveform)              runs of digital zeros -> noise shaped like the background
    background_noise(waveform, n)         n samples of that noise, to pad a short recording

Only numpy is needed here, so the service can import this module without torch.
"""

import numpy as np

#: At most this many STFT frames are used to estimate the background (a 4 s clip has 249).
MAX_REFERENCE_FRAMES = 2000


def fill_zero_runs(waveform, min_run=320):
    """Replace runs of exact digital zeros (`min_run` samples = 10 ms or longer) by background noise.

    Prevalence in the cache: 72 % of Vizina files start with 13 ms of zeros, 29 % of NCS files
    have up to 0.6 s at their ends, some BirdNET excerpts up to 10 s inside.
    """
    is_zero = waveform == 0
    if is_zero.sum() < min_run:
        return waveform

    # Run boundaries: +1 where a run of zeros starts, -1 one past where it ends.
    edges = np.diff(np.concatenate([[0], is_zero.view(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    runs = [(start, end) for start, end in zip(starts, ends) if end - start >= min_run]
    if not runs:
        return waveform

    waveform = waveform.copy()
    is_real_audio = np.ones(len(waveform), bool)
    for start, end in runs:
        is_real_audio[start:end] = False
    real_audio = waveform[is_real_audio]
    for start, end in runs:
        waveform[start:end] = background_noise(real_audio, end - start)
    return waveform


def _stft_magnitudes(signal, n_fft, hop, window, max_frames=None):
    """Magnitude spectra of overlapping windowed frames, shape (frames, n_fft / 2 + 1)."""
    n_frames = 1 + (len(signal) - n_fft) // hop
    stride = signal.strides[0]
    frames = np.lib.stride_tricks.as_strided(signal, (n_frames, n_fft), (stride * hop, stride))
    if max_frames is not None and n_frames > max_frames:  # a long recording: evenly spaced frames
        frames = frames[np.linspace(0, n_frames - 1, max_frames).astype(int)]
    frames = frames * window
    frames = frames[(frames**2).sum(1) > 1e-10]  # drop frames of digital silence
    return np.abs(np.fft.rfft(frames, axis=1))


def background_noise(waveform, n_samples, n_fft=1024, hop=512):
    """`n_samples` of noise with the spectrum of the recording's background.

    The background spectrum is the 25th percentile, per frequency bin, of the STFT magnitude over
    the non-silent frames: low enough to exclude the song even in an excerpt that is mostly song.
    White noise is then shaped to that spectrum. The result is deterministic for a given input.
    """
    rng = np.random.default_rng(len(waveform) * 7 + n_samples)
    if len(waveform) < n_fft:
        level = float(np.std(waveform)) * 0.3 + 1e-5 if len(waveform) else 1e-4
        return rng.normal(0, level, n_samples).astype(np.float32)

    window = np.hanning(n_fft).astype(np.float32)
    magnitudes = _stft_magnitudes(waveform, n_fft, hop, window, MAX_REFERENCE_FRAMES)
    if len(magnitudes) < 4:
        return rng.normal(0, 1e-4, n_samples).astype(np.float32)
    background = np.percentile(magnitudes, 25, axis=0)  # (n_fft / 2 + 1,)

    # White noise, measured the same way, gives the gain needed per frequency bin.
    white = rng.normal(0, 1, max(n_samples, n_fft) + n_fft).astype(np.float32)
    white_level = np.percentile(_stft_magnitudes(white, n_fft, hop, window), 25, axis=0)
    gain = background / (white_level + 1e-9)

    # Apply the gain in the frequency domain of the whole noise signal.
    spectrum = np.fft.rfft(white)
    gain_per_bin = np.interp(np.linspace(0, 1, len(spectrum)), np.linspace(0, 1, len(gain)), gain)
    return np.fft.irfft(spectrum * gain_per_bin, len(white))[:n_samples].astype(np.float32)
