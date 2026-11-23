"""Evaluate bootstrap segmenter against Vizina component-level annotations (precise song extents)."""
from config import SRC, WS
import sys, os, numpy as np, pandas as pd, itertools
from processing.segment import segment_songs
V = pd.read_csv(f'{WS}/data/vizina_songs.csv')
man = pd.read_csv(f'{WS}/audio32k/manifest.csv') if os.path.exists(f'{WS}/audio32k/manifest.csv') else None
root = f'{WS}/audio32k/vizina/'
V['npy'] = V.audio.fillna('').map(lambda p: root + os.path.relpath(p, f'{SRC}/source/Vizina_YH_all_rec') + '.npy' if p else '')
V = V[V.npy.map(os.path.exists)]
files = V.npy.unique()
print('files with audio', len(files), 'songs', len(V))
params = dict(hi=float(sys.argv[1]) if len(sys.argv) > 1 else 12, lo=float(sys.argv[2]) if len(sys.argv) > 2 else 6, max_gap=float(sys.argv[3]) if len(sys.argv) > 3 else 0.3)
tp = fn = fp = 0; on_err = []; off_err = []; split = 0
for f in files:
    a = np.load(f).astype(np.float32) / 32768
    segs, env = segment_songs(a, **params)
    ann = V[V.npy == f]
    # annotated region of this file (annotators may not annotate all songs): restrict fp counting to [min t0 - 5, max t1 + 5]
    lo_t, hi_t = ann.t0.min() - 2, ann.t1.max() + 2
    used = set()
    for r in ann.itertuples():
        # best overlapping segment
        ov = [(min(r.t1, s[1]) - max(r.t0, s[0]), i) for i, s in enumerate(segs)]
        ov = [o for o in ov if o[0] > 0]
        if not ov: fn += 1; continue
        n_ov = len(ov)
        o, i = max(ov); s = segs[i]; used.add(i)
        if n_ov > 1: split += 1
        tp += 1; on_err.append(s[0] - r.t0); off_err.append(s[1] - r.t1)
    fp += sum(1 for i, s in enumerate(segs) if i not in used and lo_t <= s[0] <= hi_t)
on_err, off_err = np.array(on_err), np.array(off_err)
print(params, f'recall {tp/(tp+fn):.3f}  extra-segs(in annotated span) {fp}  songs split into >1 seg {split}')
print('onset err  (s) median %.3f  |err|<0.15: %.3f  |err|<0.3: %.3f' % (np.median(on_err), np.mean(np.abs(on_err) < .15), np.mean(np.abs(on_err) < .3)))
print('offset err (s) median %.3f  |err|<0.15: %.3f  |err|<0.3: %.3f  seg ends before song end by >0.3: %.3f' % (np.median(off_err), np.mean(np.abs(off_err) < .15), np.mean(np.abs(off_err) < .3), np.mean(off_err < -0.3)))
