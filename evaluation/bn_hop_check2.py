"""BirdNET window placement, pipeline-faithful: songs must first be found by the tonal segmenter (as in
build_song_index / jcu_pipeline); then the event gets a YH score under each window scheme.
Recall on annotated Vizina test songs + false-positive rate on tonal events inside hard-negative windows
(other species detected by BirdNET; biased in favour of grid3, which selected them, so compare the others)."""
from config import WS
import sys, os, json, numpy as np, pandas as pd
from processing.jcu_pipeline import birdnet
from processing.segment import segment_songs
SR = 32000
SCHEMES = {'onset1': [-0.5], 'onset3': [-1.0, -0.5, 0.0], 'onset5': [-1.5, -1.0, -0.5, 0.0, 0.5]}


def scores(y, evs):
    dur = len(y) / SR; out = {k: [] for k in ['grid3', 'grid05', *SCHEMES]}; cost = dict.fromkeys(out, 0)
    for name, hop in [('grid3', 3.0), ('grid05', 0.5)]:
        st = np.arange(0, max(dur - 3, 0) + 1e-6, hop); sc = birdnet(y, list(st)) if len(st) else np.zeros(0); cost[name] = len(st)
        for s0, s1 in evs:
            ov = (st < s1) & (st + 3 > s0); out[name].append(float(sc[ov].max()) if ov.any() else 0.0)
    for name, offs in SCHEMES.items():
        st = [max(0.0, min(s0 + o, dur - 3)) for s0, _ in evs for o in offs]
        sc = birdnet(y, st).reshape(len(evs), len(offs)) if evs else np.zeros((0, 1)); cost[name] = len(st)
        out[name] += list(sc.max(1))
    return out, cost


V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str}); V = V[V.split == 'test']
R = {k: [] for k in ['grid3', 'grid05', *SCHEMES]}; C = dict.fromkeys(R, 0); found = 0
for npy, g in V.groupby('npy'):
    lo, hi = max(0, g.t0.min() - 3), g.t1.max() + 3
    a = np.load(npy, mmap_mode='r'); y = np.asarray(a[int(lo * SR):int(hi * SR)], np.float32) / 32768
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.6, max_dur=6.0)
    evs = []
    for r in g.itertuples():
        best, bo = None, 0
        for s0, s1, _ in segs:
            o = min(r.t1 - lo, s1) - max(r.t0 - lo, s0)
            if o > bo: best, bo = (s0, s1), o
        evs.append(best)
    ok = [e for e in evs if e is not None]; found += len(ok)
    sc, cost = scores(y, ok)
    for k in R:
        it = iter(sc[k]); R[k] += [next(it) if e is not None else 0.0 for e in evs]; C[k] += cost[k]
H = pd.read_csv(f'{WS}/data/hardneg_v1.csv').sample(600, random_state=0); FP = {k: [] for k in R}
for r in H.itertuples():
    a = np.load(r.npy, mmap_mode='r'); lo = max(0, r.t0 - 3); y = np.asarray(a[int(lo * SR):int((r.t1 + 3) * SR)], np.float32) / 32768
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.6, max_dur=6.0)
    evs = [(s0, s1) for s0, s1, _ in segs if s0 >= r.t0 - lo and s1 <= r.t1 - lo + 0.5]
    if not evs: continue
    sc, _ = scores(y, evs)
    for k in FP: FP[k] += list(sc[k])
out = {'songs': len(V), 'segmenter_found': found, 'hardneg_events': len(FP['grid3'])}
for k in R:
    out[k] = {**{f'recall@{t}': round(float((np.array(R[k]) >= t).mean()), 3) for t in (0.2, 0.3, 0.5)},
              **{f'fp@{t}': round(float((np.array(FP[k]) >= t).mean()), 3) for t in (0.2, 0.3)}, 'windows_vizina': C[k]}
print(json.dumps(out, indent=1)); json.dump(out, open(f'{WS}/results/bn_hop_check2.json', 'w'), indent=1)
