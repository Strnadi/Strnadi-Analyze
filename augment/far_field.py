"""Far-field simulation: what a song sounds like 10-50 m from the bird on a passive recorder.

Distance-scaled high-frequency loss above 3 kHz + exponentially decaying reverberation whose
direct-to-reverberant ratio falls with distance; the SNR to mix noise at comes from playback measurements
(BioDCASE 2025 Task 3: the same 209 songs re-recorded at 6.5-200 m; `data/farfield_stats.csv`).
Profiles beyond ~50 m are excluded: the song is unreadable there, so the label would become noise.

History: v1 drew a random spectral-tilt sign and boosted highs by up to 30 dB half the time (found by visual
QA); this is the physical v2. No completed research run used either version.
Returns (waveform, snr_db_to_mix_noise_at).
"""
import os
import numpy as np
from scipy.signal import fftconvolve

SR = 32000
RANKS = 'BCDEFGH'          # distance classes in the playback data, nearest first


def load_profiles(csv_path, max_dist='F', min_n=50):
    import pandas as pd
    if not os.path.exists(csv_path):
        return []
    rows = pd.read_csv(csv_path).to_dict('records')
    return [r for r in rows if r['dist'] in RANKS[:RANKS.index(max_dist) + 1] and r['n'] >= min_n]


def far_field(y, profile):
    rank = RANKS.index(profile['dist']) + 1
    Y = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / SR) / 1000
    tilt = -np.random.uniform(0, 0.4 * rank)                                     # dB per kHz above 3 kHz
    Y *= 10 ** (np.clip(tilt * np.maximum(f - 3.0, 0), -30, 0) / 20)
    y = np.fft.irfft(Y, n=len(y)).astype(np.float32)
    rt = np.random.uniform(0.1, 0.8 if profile.get('hab') == 'forest' else 0.4); L = int(rt * SR)
    ir = np.random.normal(0, 1, L) * np.exp(-6.9 * np.arange(L) / L)             # -60 dB at rt
    drr = np.random.uniform(10, 20) - 3 * rank                                   # direct-to-reverberant ratio, dB
    ir = ir / np.sqrt((ir ** 2).sum()) * 10 ** (-drr / 20); ir[0] += 1.0
    y = fftconvolve(y, ir)[:len(y)].astype(np.float32)
    return y, float(np.clip(np.random.normal(profile['snr'], 4.0), 5, 35))
