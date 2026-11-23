"""Counterfactual song surgery on Vizina training songs (complete BhBl, annotated components).

 swap      : exchange the two terminal elements (Bh<->Bl, gap kept)   -> 'BlBh'
 trunc2    : replace the 2nd terminal element with same-file background -> 'Unfinished'  (B only)
 trunc12   : replace both terminal elements with background            -> 'Unfinished'  (first phrase only)
 orig      : the untouched song window                                 -> 'BhBl'
Element boundaries: annotated component boxes snapped to the tonal elements found by the parser.
Writes 4.4 s int16 windows to audio32k/cf/ and a table data/cf_set.csv usable as train_cv --extra.
"""

from config import SRC, WS
import os, sys, numpy as np, pandas as pd
from processing.parse import spectrogram, elements

SR = 32000
OUT = f'{WS}/audio32k/cf_v4'
os.makedirs(OUT, exist_ok=True)
V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str})
V = V[(V.split == 'train') & (V.label == 'BhBl')]
import csv, glob, re

# reload component boxes per song
ROOT = f'{SRC}/source/Vizina_YH_all_rec/Vizina_annotation/'
comp = {}
for f in glob.glob(ROOT + '*.txt'):
    for r in csv.DictReader(open(f, encoding='utf-8', errors='replace'), delimiter='\t'):
        c = (r.get('Component') or '').strip().lower().replace(' ', '')
        m = re.match(r'd[ia]+l?[ae]*c?t?_?(\d)', c)
        if m:
            comp.setdefault((os.path.basename(f), (r.get('song_code') or '').strip()), {})[
                'd' + m.group(1)
            ] = (float(r['Begin Time (s)']), float(r['End Time (s)']))


def snap(els, box, off):
    """element (in window coords) with max overlap to annotated box (file coords)"""
    b0, b1 = box[0] - off, box[1] - off
    best, bo = None, 0
    for e in els:
        o = min(b1, e['t1']) - max(b0, e['t0'])
        if o > bo and e['dur'] >= 0.1:
            best, bo = e, o
    return best


def xfade_concat(parts, fade=int(0.005 * SR)):
    out = parts[0].copy()
    for p in parts[1:]:
        if len(out) >= fade and len(p) >= fade:
            w = np.linspace(0, 1, fade, dtype=np.float32)
            out[-fade:] = out[-fade:] * (1 - w) + p[:fade] * w
            out = np.concatenate([out, p[fade:]])
        else:
            out = np.concatenate([out, p])
    return out


def match_spectrum(x, ref, nper=512):
    """per-frequency-bin EQ of the filler x to the reference background ref (median STFT power; robust to calls).
    v4: the v3 full-band RMS match against the short inter-element gap left visibly black patches (QA sheet)."""
    from scipy.signal import stft, istft

    _, _, X = stft(x, SR, nperseg=nper)
    _, _, R = stft(ref, SR, nperseg=nper)
    g = np.sqrt((np.median(np.abs(R) ** 2, 1) + 1e-14) / (np.median(np.abs(X) ** 2, 1) + 1e-14))
    g = np.convolve(g, np.ones(5) / 5, 'same')
    g = np.clip(g, 0.05, 20)
    _, y = istft(X * g[:, None], SR, nperseg=nper)
    y = y[: len(x)].astype(np.float32)
    return np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y


def background(a, n, rng, after=None, ref=None):
    """filler = background right after the song, spectrally matched to the window's own pre-song background (ref)"""
    L = len(a)
    s = int(after * SR) if after is not None else int(rng.integers(0, max(1, L - n)))
    if s + n > L:
        s = max(0, L - n)
    x = a[s : s + n].astype(np.float32) / 32768
    return match_spectrum(x, ref) if ref is not None and len(ref) > 2048 else x


rows = []
rng = np.random.default_rng(0)
ok = 0
for r in V.itertuples():
    c = comp.get((r.ann, str(r.song_code)))
    if not c or 'd1' not in c or 'd2' not in c:
        continue
    a = np.load(r.npy, mmap_mode='r')
    w0 = r.t0 - 0.6
    if w0 < 0 or (w0 + 4.4) * SR > len(a):
        continue
    y = np.asarray(a[int(w0 * SR) : int(w0 * SR) + int(4.4 * SR)], np.float32) / 32768
    L, f = spectrogram(y)
    els = elements(L, f)
    e1, e2 = snap(els, c['d1'], w0), snap(els, c['d2'], w0)
    if e1 is None or e2 is None or e2['t0'] <= e1['t1'] or e2['t0'] - e1['t1'] > 0.5:
        continue
    i1a, i1b = int((e1['t0'] - 0.01) * SR), int((e1['t1'] + 0.01) * SR)
    i2a, i2b = int((e2['t0'] - 0.01) * SR), int((e2['t1'] + 0.01) * SR)
    if not (0 < i1a < i1b <= i2a < i2b < len(y)):
        continue
    ok += 1
    base = f'{OUT}/{os.path.basename(r.npy)[:-8]}_{r.song_code}'
    variants = {
        'orig': (
            xfade_concat([y[:i1a], y[i1a:i1b], y[i1b:i2a], y[i2a:i2b], y[i2b:]]),
            'BhBl',
        ),  # null splices at the same junctions
        'swap': (xfade_concat([y[:i1a], y[i2a:i2b], y[i1b:i2a], y[i1a:i1b], y[i2b:]]), 'BlBh'),
        'trunc2': (
            xfade_concat(
                [y[:i2a], background(a, i2b - i2a, rng, r.t1 + 0.15, y[: int(0.5 * SR)]), y[i2b:]]
            ),
            'Unfinished',
        ),
        'trunc12': (
            xfade_concat(
                [y[:i1a], background(a, i2b - i1a, rng, r.t1 + 0.15, y[: int(0.5 * SR)]), y[i2b:]]
            ),
            'Unfinished',
        ),
    }
    for k, (yy, lab) in variants.items():
        yy = yy[: int(4.4 * SR)]
        if len(yy) < int(4.4 * SR):
            yy = np.pad(yy, (0, int(4.4 * SR) - len(yy)))
        p = f'{base}_{k}.npy'
        np.save(p, np.clip(yy * 32767, -32768, 32767).astype(np.int16))
        rows.append(
            dict(
                npy=p,
                t0=0.0,
                t1=4.4,
                label=lab,
                fold=-1,
                aug=False,
                name=f'cf_{k}_{r.song_code}',
                w=1.0,
                sw=1.0,
                source='cf',
                rec=f'vizina:{r.indiv}',
                variant=k,
            )
        )
D = pd.DataFrame(rows)
D['source'] = 'cf'
D.to_csv(f'{WS}/data/cf_v4.csv', index=False)
print('songs used', ok, 'of', len(V))
print(D.groupby(['variant', 'label']).size())
