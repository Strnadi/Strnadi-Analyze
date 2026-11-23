"""Element-level parsing of Yellowhammer songs + knowledge-based dialect rules.

Mirrors how experts measure dialects (Diblikova et al. 2023, Avian Research 14:100115): find the terminal
phrase elements after the first-phrase trill, measure their start/end/median frequency, bandwidth, duration and the
frequency offset between them, then apply rules:
  BC   : B (flat) followed by a high, descending sweep (C)
  BE   : B followed by a high, ascending element (E, bandwidth > ~500 Hz)
  BhBl : two flat B elements, second lower
  BlBh : two flat B elements, second higher, small bandwidth (else BE per the 2023 lumping)
  XB   : short high X element(s) directly followed by a final flat B
  Unfinished : trill (+ optional single B) with no terminal element
Returns (label or None if unparseable, confidence 0..1, features dict).
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

SR = 32000
NFFT, HOP = 1024, 160           # 31 Hz bins, 5 ms hop
FMIN, FMAX = 2000, 10500


def spectrogram(y):
    y = np.asarray(y, np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(y, NFFT)[::HOP] * np.hanning(NFFT).astype(np.float32)
    S = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    b = (f >= FMIN) & (f <= FMAX)
    L = 20 * np.log10(S[:, b] + 1e-7)
    L = L - np.median(L, axis=0, keepdims=True)       # per-frequency noise floor
    return np.clip(L, 0, None), f[b]


def elements(L, freqs, snr_db=9.0, max_jump_hz=300, min_dur=0.02, gap_frames=3):
    """track the dominant spectral peak per frame and split into continuous elements"""
    Ls = median_filter(L, size=(3, 3))
    pk = Ls.argmax(1); pv = Ls.max(1)
    active = uniform_filter1d((pv > snr_db).astype(float), 3) > 0.5
    pf = freqs[pk]
    els, cur, miss = [], [], 0
    for t in range(len(pv)):
        if active[t]:
            if cur and abs(pf[t] - pf[cur[-1]]) > max_jump_hz:
                els.append(cur); cur = []
            cur.append(t); miss = 0
        else:
            miss += 1
            if cur and miss > gap_frames:
                els.append(cur); cur = []
    if cur: els.append(cur)
    out = []
    dt = HOP / SR
    for e in els:
        e = np.array(e)
        if len(e) * dt < min_dur: continue
        fe = pf[e]; n = max(1, len(e) // 5)
        out.append(dict(t0=e[0] * dt, t1=(e[-1] + 1) * dt, dur=len(e) * dt, f0=float(np.median(fe[:n])), f1=float(np.median(fe[-n:])),
                        fmed=float(np.median(fe)), fmin=float(fe.min()), fmax=float(fe.max()), snr=float(pv[e].mean())))
    # merge fragments of the same tonal element split by noise (close in time and frequency)
    merged = []
    for el in out:
        if merged and el['t0'] - merged[-1]['t1'] < 0.03 and abs(el['f0'] - merged[-1]['f1']) < max_jump_hz:
            m = merged[-1]; w0, w1 = m['dur'], el['dur']
            m.update(t1=el['t1'], dur=m['dur'] + el['dur'], f1=el['f1'], fmed=(m['fmed'] * w0 + el['fmed'] * w1) / (w0 + w1),
                     fmin=min(m['fmin'], el['fmin']), fmax=max(m['fmax'], el['fmax']), snr=(m['snr'] * w0 + el['snr'] * w1) / (w0 + w1))
        else:
            merged.append(dict(el))
    return merged


def classify(els, long_dur=0.12, flat_bw=300):
    """apply dialect rules to the element list; returns (label, confidence, feats)"""
    feats = {'n_el': len(els)}
    if len(els) < 3:
        return None, 0.0, feats
    longs = [i for i, e in enumerate(els) if e['dur'] >= long_dur]
    feats['n_long'] = len(longs)
    if not longs:
        return 'Unfinished', 0.4, feats
    # terminal block = last long element and what follows/precedes it closely
    i2 = longs[-1]; e2 = els[i2]
    prev_long = longs[-2] if len(longs) >= 2 else None
    e1 = els[prev_long] if prev_long is not None and e2['t0'] - els[prev_long]['t1'] < 0.7 else None
    tail_after = [e for e in els[i2 + 1:] if e['dur'] >= 0.06]
    feats.update(e2_f0=e2['f0'], e2_f1=e2['f1'], e2_fmed=e2['fmed'], e2_dur=e2['dur'], e2_bw=e2['f1'] - e2['f0'], n_after=len(tail_after))
    if e1 is not None:
        feats.update(e1_f1=e1['f1'], e1_fmed=e1['fmed'], e1_dur=e1['dur'], e1_bw=e1['f1'] - e1['f0'], df=e2['f0'] - e1['f1'],
                     gap=e2['t0'] - e1['t1'])
    e2_flat = abs(e2['f1'] - e2['f0']) < flat_bw and (e2['fmax'] - e2['fmin']) < 2 * flat_bw
    if e1 is not None:
        e1_flat = abs(e1['f1'] - e1['f0']) < flat_bw
        df = e2['fmed'] - e1['fmed']; bw2 = e2['f1'] - e2['f0']
        if e1_flat and bw2 < -500 and df > 600:
            return 'BC', 0.8, feats
        if e1_flat and bw2 > 400 and df > 300:
            return 'BE', 0.75, feats
        if e1_flat and e2_flat:
            if df < -120: return 'BhBl', min(0.9, 0.5 + abs(df) / 1000), feats
            if df > 120: return ('BlBh' if df < 1500 else 'BE'), min(0.85, 0.5 + abs(df) / 1000), feats
            return 'Unfinished', 0.3, feats
    # single final long element: XB if preceded by short high X elements close in time
    if e2_flat:
        pre = [e for e in els[max(0, i2 - 4):i2] if e2['t0'] - e['t1'] < 0.35 and e['dur'] < long_dur and e['fmed'] > e2['fmed'] + 1000]
        feats['n_x'] = len(pre)
        if pre and not tail_after:
            return 'XB', 0.6, feats
        if not tail_after:
            return 'Unfinished', 0.5, feats
    return None, 0.0, feats


def parse_clip(y):
    L, f = spectrogram(y)
    els = elements(L, f)
    lab, conf, feats = classify(els)
    return lab, conf, feats, els
