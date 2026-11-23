"""Far-field simulation: what a song sounds like 10-50 m from the bird on a passive recorder.

Distance-scaled high-frequency loss above 3 kHz + exponentially decaying reverberation whose
direct-to-reverberant ratio falls with distance; the SNR to mix noise at comes from playback
measurements (BioDCASE 2025 Task 3: the same 209 songs re-recorded at 6.5-200 m;
`data/farfield_stats.csv`). Profiles beyond ~50 m are excluded: the song is unreadable there, so
the label would become noise.

History: v1 drew a random spectral-tilt sign and boosted highs by up to 30 dB half the time (found
by visual QA); this is the physical v2. No completed research run used either version.

Returns (waveform, snr_db_to_mix_noise_at).
"""

import os
import numpy as np
from scipy.signal import fftconvolve

SR = 32000  # sample rate, Hz
RANKS = 'BCDEFGH'  # distance classes in the playback data, nearest first


def load_profiles(csv_path, max_dist='F', min_n=50):
    """Read the measured far-field profiles and keep the usable ones.

    Args:
        csv_path: path of the statistics table (`data/farfield_stats.csv`), one row per profile
            with at least the columns `dist` (distance class, a letter of `RANKS`), `n` (number
            of measurements) and `snr` (dB); `far_field` also reads `hab` (habitat) when present.
        max_dist: farthest distance class to keep (a letter of `RANKS`).
        min_n: smallest number of measurements a profile needs to be kept.

    Returns:
        List of profiles, each a dict of one table row. Empty when the file does not exist.
    """
    # Imported lazily: pandas is only needed when profiles are actually loaded.
    import pandas as pd

    if not os.path.exists(csv_path):
        return []
    rows = pd.read_csv(csv_path).to_dict('records')

    # The distance classes from the nearest up to and including max_dist, e.g. 'BCDEF'.
    farthest_rank = RANKS.index(max_dist)
    allowed_distances = RANKS[: farthest_rank + 1]

    profiles = []
    for row in rows:
        if row['dist'] in allowed_distances and row['n'] >= min_n:
            profiles.append(row)
    return profiles


def far_field(y, profile):
    """Make a clip sound as if recorded from the distance described by `profile`.

    Two effects are applied, both stronger for farther distance classes: loss of high
    frequencies, then reverberation. Background noise is NOT added here; the function returns the
    SNR at which the caller should mix it. Uses the global `np.random` state.

    Args:
        y: waveform, 1-D float32 array at 32 kHz.
        profile: dict from `load_profiles` with the keys `dist` (distance class, a letter of
            `RANKS`), `snr` (SNR measured at that distance, dB; the centre of the random draw)
            and optionally `hab` (habitat; 'forest' allows longer reverberation).

    Returns:
        (waveform, snr_db): the processed float32 waveform, same length as `y`, and the SNR in dB
        to mix noise at.
    """
    # 1 for the nearest distance class, growing with distance.
    distance_rank = RANKS.index(profile['dist']) + 1

    # --- High-frequency loss: attenuate everything above 3 kHz, more the higher it is. ---
    spectrum = np.fft.rfft(y)
    frequencies_khz = np.fft.rfftfreq(len(y), 1 / SR) / 1000
    # Slope of the loss in dB per kHz above 3 kHz (negative = attenuation only).
    tilt_db_per_khz = -np.random.uniform(0, 0.4 * distance_rank)
    khz_above_3 = np.maximum(frequencies_khz - 3.0, 0)
    gain_db = np.clip(tilt_db_per_khz * khz_above_3, -30, 0)  # never more than 30 dB of loss
    spectrum *= 10 ** (gain_db / 20)
    y = np.fft.irfft(spectrum, n=len(y)).astype(np.float32)

    # --- Reverberation: convolve with a synthetic impulse response (decaying noise). ---
    # Reverberation time in seconds; the forest habitat allows a longer one than the others.
    if profile.get('hab') == 'forest':
        longest_reverb_time = 0.8
    else:
        longest_reverb_time = 0.4
    reverb_time = np.random.uniform(0.1, longest_reverb_time)
    reverb_samples = int(reverb_time * SR)

    # White noise under an exponential envelope that has fallen by 60 dB at the reverberation
    # time (exp(-6.9) = 0.001).
    decay_envelope = np.exp(-6.9 * np.arange(reverb_samples) / reverb_samples)
    impulse_response = np.random.normal(0, 1, reverb_samples) * decay_envelope

    # Direct-to-reverberant ratio in dB: 3 dB less direct sound per distance class.
    direct_to_reverb_db = np.random.uniform(10, 20) - 3 * distance_rank
    # Scale the reverberant tail to unit energy, then to its level below the direct sound.
    tail_energy = (impulse_response**2).sum()
    reverb_gain = 10 ** (-direct_to_reverb_db / 20)
    impulse_response = impulse_response / np.sqrt(tail_energy) * reverb_gain
    # The direct sound: the unchanged signal at delay 0.
    impulse_response[0] += 1.0

    reverberant = fftconvolve(y, impulse_response)
    y = reverberant[: len(y)].astype(np.float32)

    # --- SNR to mix noise at: around the measured mean (std 4 dB), kept within 5-35 dB. ---
    snr_db = np.random.normal(profile['snr'], 4.0)
    snr_db = float(np.clip(snr_db, 5, 35))
    return y, snr_db
