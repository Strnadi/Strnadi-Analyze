"""Mix real background noise (AudioMoth chunks with no Yellowhammer) into a clip at a chosen SNR.

What "signal power" means decides whether the label survives:
  mode='clip'    whole-clip power. Research runs R1-R6. Wrong: a faint song in a loud clip disappeared under
                 noise-bank birds while the label stayed a dialect.
  mode='band'    power of the loudest 20 % of frames in 1.5-11 kHz (the phrase) vs noise in the same band.
                 Research fix (R7). Still referenced to the loud first phrase.
  mode='ending'  power of the LAST 0.7 s of the contiguous song. Fork-review change and the default here.
                 Checked on real clips (spectrogram sheet, 5 clips): it lands on the FIRST terminal element (B) and
                 is only 2-6 dB below the phrase reference, so it is a partial fix. The second, faintest element
                 (the rising C / E, often detached by > 0.3 s) is not found by an energy rule and can still be
                 buried at low SNR. Until element positions are known (model/element_tagger.py), the corrected
                 recipes therefore also raise the SNR range to 6-26 dB (experiments/recipes.py::CLEAN).

The noise bank itself must be clean: `processing/clean_noise_bank.py` removes chunks containing songs (the
research bank had Yellowhammer songs in ~9-16 % of its chunks).
"""
import numpy as np

SR, NFFT, HOP = 32000, 1024, 640
_BAND = None


def _frame_energy(y):
    global _BAND
    if _BAND is None:
        f = np.fft.rfftfreq(NFFT, 1 / SR); _BAND = (f >= 1500) & (f <= 11000)
    n = (len(y) - NFFT) // HOP + 1
    if n < 1:
        return np.array([float((y ** 2).mean())])
    y = np.ascontiguousarray(y, np.float32)
    fr = np.lib.stride_tricks.as_strided(y, (n, NFFT), (y.strides[0] * HOP, y.strides[0])) * np.hanning(NFFT).astype(np.float32)
    return (np.abs(np.fft.rfft(fr, axis=1))[:, _BAND] ** 2).sum(1)


def signal_power(y, mode='ending'):
    if mode == 'clip':
        return float((y ** 2).mean()) + 1e-9
    e = _frame_energy(y)
    if mode == 'band' or len(e) < 10:
        return float(np.sort(e)[-max(1, len(e) // 5):].mean()) + 1e-9
    # the song = the run of active frames around the loudest frame, bridging gaps <= 0.3 s (the pause before the
    # terminal elements); "active" = at least 6 dB over the noise floor and within 20 dB of the peak, so an ending far
    # fainter than the phrase still counts. Reference = median power of the active frames in the song's last 0.7 s.
    floor = np.percentile(e, 20); peak = e.max()
    active = e > max(4 * floor, 0.01 * peak)
    if not active.any():
        return float(e.mean()) + 1e-9
    gap = int(0.3 * SR / HOP); idx = np.nonzero(active)[0]; k = int(np.argmax(e))
    end = k
    for i in idx[idx > k]:
        if i - end > gap:
            break
        end = i
    tail = idx[(idx <= end) & (idx >= end - int(0.7 * SR / HOP))]
    return float(np.median(e[tail])) + 1e-9


def noise_power(nz, mode='ending'):
    return (float((nz ** 2).mean()) if mode == 'clip' else float(_frame_energy(nz).mean())) + 1e-9


def mix_noise(y, noise, snr_db, mode='ending'):
    gain = np.sqrt(signal_power(y, mode) / noise_power(noise, mode) / 10 ** (snr_db / 10))
    return (y + noise.astype(np.float32) * float(gain)).astype(np.float32)


def draw_chunk(bank, n):
    nz = bank[np.random.randint(len(bank))]
    s = np.random.randint(0, max(1, len(nz) - n)); nz = np.asarray(nz[s:s + n], np.float32)
    return nz if len(nz) == n else np.resize(nz, n)
