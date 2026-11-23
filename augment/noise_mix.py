"""Mix real background noise (AudioMoth chunks with no Yellowhammer) into a clip at a chosen SNR.

What "signal power" means decides whether the label survives:

  mode='clip'    Whole-clip power. Research runs R1-R6. Wrong: a faint song in a loud clip
                 disappeared under noise-bank birds while the label stayed a dialect.

  mode='band'    Power of the loudest 20 % of frames in 1.5-11 kHz (the phrase) vs noise in the
                 same band. Research fix (R7). Still referenced to the loud first phrase.

  mode='ending'  Power of the LAST 0.7 s of the contiguous song. Fork-review change and the
                 default here. Checked on real clips (spectrogram sheet, 5 clips): it lands on the
                 FIRST terminal element (B) and is only 2-6 dB below the phrase reference, so it
                 is a partial fix. The second, faintest element (the rising C / E, often detached
                 by > 0.3 s) is not found by an energy rule and can still be buried at low SNR.
                 Until element positions are known (model/element_tagger.py), the corrected
                 recipes therefore also raise the SNR range to 6-26 dB
                 (the corrected experiment scripts).

The noise bank itself must be clean: `processing/clean_noise_bank.py` removes chunks containing
songs (the research bank had Yellowhammer songs in ~9-16 % of its chunks).
"""

import numpy as np

SR = 32000  # sample rate, Hz
NFFT = 1024  # analysis frame length, samples (32 ms)
HOP = 640  # step between analysis frames, samples (20 ms)

# Boolean mask over the FFT bins that lie in the song band (1.5-11 kHz); built on first use.
_BAND = None


def _frame_energy(y):
    """Energy of the song band (1.5-11 kHz) in every 32 ms analysis frame of a waveform.

    Args:
        y: waveform, 1-D array.

    Returns:
        1-D array with one value per frame: the summed squared FFT magnitude of the Hann-windowed
        frame inside the band. A clip shorter than one frame gives a single value instead, the
        mean square of its samples (a plain time-domain power, not band-limited).
    """
    global _BAND
    if _BAND is None:
        bin_frequencies_hz = np.fft.rfftfreq(NFFT, 1 / SR)
        _BAND = (bin_frequencies_hz >= 1500) & (bin_frequencies_hz <= 11000)

    n_frames = (len(y) - NFFT) // HOP + 1
    if n_frames < 1:
        mean_square = float((y**2).mean())
        return np.array([mean_square])

    # Cut the waveform into overlapping frames without copying it: row i is a view of
    # y[i * HOP : i * HOP + NFFT].
    y = np.ascontiguousarray(y, np.float32)
    sample_stride = y.strides[0]  # bytes from one sample to the next
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, NFFT),
        strides=(sample_stride * HOP, sample_stride),
    )

    window = np.hanning(NFFT).astype(np.float32)
    windowed_frames = frames * window

    spectrum = np.fft.rfft(windowed_frames, axis=1)  # (n_frames, NFFT // 2 + 1), complex
    band_magnitude = np.abs(spectrum)[:, _BAND]
    band_power = band_magnitude**2
    return band_power.sum(1)


def signal_power(y, mode='ending'):
    """Reference power of the song in a clip, the "S" of the signal-to-noise ratio.

    Args:
        y: waveform of the clip, 1-D float array.
        mode: what counts as the signal (see the module docstring):
            'clip'   mean square of the whole clip;
            'band'   mean band energy of the loudest 20 % of frames;
            'ending' median band energy of the active frames in the last 0.7 s of the song.

    Returns:
        The power as a float (plus 1e-9, so it is never zero). 'clip' is a time-domain mean
        square; the other two are band energies per frame, to be compared only with
        `noise_power` of the same mode.
    """
    if mode == 'clip':
        return float((y**2).mean()) + 1e-9

    frame_energy = _frame_energy(y)
    n_frames = len(frame_energy)

    # 'band' mode, and also the fallback for clips too short (< 10 frames) to look for an ending.
    if mode == 'band' or n_frames < 10:
        n_loudest = max(1, n_frames // 5)
        loudest_frames = np.sort(frame_energy)[-n_loudest:]
        return float(loudest_frames.mean()) + 1e-9

    # 'ending' mode.
    # The song = the run of active frames around the loudest frame, bridging gaps <= 0.3 s (the
    # pause before the terminal elements). "Active" = at least 6 dB over the noise floor and
    # within 20 dB of the peak, so an ending far fainter than the phrase still counts.
    # Reference = median power of the active frames in the song's last 0.7 s.
    noise_floor = np.percentile(frame_energy, 20)
    peak_energy = frame_energy.max()
    # In power, a factor of 4 is +6 dB and a factor of 0.01 is -20 dB.
    activity_threshold = max(4 * noise_floor, 0.01 * peak_energy)
    is_active = frame_energy > activity_threshold
    if not is_active.any():
        return float(frame_energy.mean()) + 1e-9

    max_gap_frames = int(0.3 * SR / HOP)  # 0.3 s = 15 frames
    tail_frames = int(0.7 * SR / HOP)  # 0.7 s = 35 frames
    active_frames = np.nonzero(is_active)[0]  # indices of the active frames, ascending
    loudest_frame = int(np.argmax(frame_energy))

    # Walk forward from the loudest frame over the later active frames. The song ends at the last
    # active frame reached before a silence longer than the allowed gap.
    song_end = loudest_frame
    later_active_frames = active_frames[active_frames > loudest_frame]
    for frame in later_active_frames:
        if frame - song_end > max_gap_frames:
            break
        song_end = frame

    # The active frames within the last 0.7 s of the song.
    in_ending = (active_frames <= song_end) & (active_frames >= song_end - tail_frames)
    ending_frames = active_frames[in_ending]
    return float(np.median(frame_energy[ending_frames])) + 1e-9


def noise_power(nz, mode='ending'):
    """Power of a noise chunk, the "N" of the signal-to-noise ratio.

    Args:
        nz: noise waveform, 1-D float array.
        mode: 'clip' measures the mean square of the samples; any other mode ('band', 'ending')
            measures the mean band energy (1.5-11 kHz) over all frames, the same unit
            `signal_power` uses for that mode.

    Returns:
        The power as a float (plus 1e-9, so it is never zero).
    """
    if mode == 'clip':
        power = float((nz**2).mean())
    else:
        power = float(_frame_energy(nz).mean())
    return power + 1e-9


def mix_noise(y, noise, snr_db, mode='ending'):
    """Add noise to a clip, scaled so that the signal-to-noise ratio is `snr_db`.

    Args:
        y: waveform of the clip, 1-D float32 array.
        noise: noise waveform of the same length as `y`.
        snr_db: wanted signal-to-noise ratio in dB (higher = quieter noise).
        mode: how signal and noise power are measured: 'clip', 'band' or 'ending' (see
            `signal_power`).

    Returns:
        The mixture `y + gain * noise` as a float32 waveform. The clip itself is not rescaled.
    """
    song_power = signal_power(y, mode)
    background_power = noise_power(noise, mode)
    snr_as_power_ratio = 10 ** (snr_db / 10)

    # Wanted: song_power / (noise_gain ** 2 * background_power) = snr_as_power_ratio.
    noise_gain = np.sqrt(song_power / background_power / snr_as_power_ratio)

    scaled_noise = noise.astype(np.float32) * float(noise_gain)
    return (y + scaled_noise).astype(np.float32)


def draw_chunk(bank, n):
    """Pick a random stretch of `n` samples from a random recording of a noise bank.

    Uses the global `np.random` state: first the recording is drawn, then the start position.

    Args:
        bank: sequence of 1-D waveforms (the noise recordings).
        n: wanted length in samples.

    Returns:
        A float32 waveform of exactly `n` samples. A recording shorter than `n` is repeated from
        its start until it is long enough.
    """
    recording_index = np.random.randint(len(bank))
    recording = bank[recording_index]

    # Random start such that a full chunk fits (start 0 when the recording is too short).
    n_possible_starts = max(1, len(recording) - n)
    start = np.random.randint(0, n_possible_starts)
    chunk = np.asarray(recording[start : start + n], np.float32)

    if len(chunk) == n:
        return chunk
    # Too short: np.resize repeats the chunk until it has n samples.
    return np.resize(chunk, n)
