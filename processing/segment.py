"""Offline bootstrap song segmenter (energy/tonality based), used to cut full-song training windows
out of long weakly-labelled recordings.  Not used at inference (the model does SED itself).

song events = hysteresis-thresholded tonal-SNR envelope in 2-10 kHz, gaps < max_gap merged.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

SR = 32000
NFFT, HOP = 1024, 320  # 32 ms window, 10 ms hop
FPS = SR / HOP


def stft_mag(y):
    import numpy.lib.stride_tricks as st

    y = np.asarray(y, np.float32)
    if len(y) < NFFT:
        y = np.pad(y, (0, NFFT - len(y)))
    n = 1 + (len(y) - NFFT) // HOP
    frames = st.as_strided(y, shape=(n, NFFT), strides=(y.strides[0] * HOP, y.strides[0]))
    w = np.hanning(NFFT).astype(np.float32)
    return np.abs(np.fft.rfft(frames * w, axis=1)).astype(np.float32)  # (T, F)


def tonal_snr(y, fmin=2000, fmax=10000, noise_win_s=8.0):
    S = stft_mag(y)
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    band = (f >= fmin) & (f <= fmax)
    L = 20 * np.log10(S[:, band] + 1e-7)
    # per-bin noise floor: running 20th percentile approximated by median of a downsampled grid
    T = L.shape[0]
    step = max(1, int(FPS * 0.25))
    coarse = L[::step]
    k = max(3, int(noise_win_s * FPS / step) | 1)
    floor_c = median_filter(coarse, size=(k, 1), mode='nearest') - 0.0
    floor = np.repeat(floor_c, step, axis=0)[:T]
    snr = np.clip(L - floor, 0, None)
    # tonality: mean of top-3 bins per frame (narrowband song elements)
    top = np.sort(snr, axis=1)[:, -3:].mean(1)
    env = uniform_filter1d(top, size=5)
    # remove the (biased) baseline of the max-statistic: running median over ~10 s
    st = max(1, int(FPS * 0.2))
    kb = max(3, int(10 * FPS / st) | 1)
    base = np.repeat(median_filter(env[::st], size=kb, mode='nearest'), st)[: len(env)]
    return env - base


def segment_songs(y, hi=12.0, lo=6.0, max_gap=0.30, min_dur=0.5, max_dur=8.0, pad=0.0):
    env = tonal_snr(y)
    above_lo = env > lo
    # hysteresis: keep lo-runs that contain at least one hi frame
    d = np.diff(np.concatenate([[0], above_lo.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    segs = [(s, e) for s, e in zip(starts, ends) if env[s:e].max() > hi]
    # merge small gaps
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / FPS < max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        t0, t1 = s / FPS, e / FPS
        if min_dur <= t1 - t0 <= max_dur:
            out.append((max(0, t0 - pad), t1 + pad, float(env[s:e].mean())))
    return out, env
