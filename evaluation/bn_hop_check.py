"""How much does the BirdNET step matter?  On annotated Vizina test songs, YH score per song under:
  grid3   : 3 s windows, 3 s hop (what the corpus pass birdnet24/ used)    score = max over windows overlapping the song
  grid05  : 3 s windows, 0.5 s hop (old birdnet-segment branch, 2.5 s overlap)
  onset   : one window starting 0.5 s before the tonal-event onset (what jcu_pipeline does)
Reports share of songs with score >= 0.2 / 0.3 / 0.5 (detection recall at the thresholds we filter with) and cost."""
from config import WS
import sys, os, json, numpy as np, pandas as pd
from processing.jcu_pipeline import birdnet
SR = 32000
V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str}); V = V[V.split == 'test']
res = {k: [] for k in ['grid3', 'grid05', 'onset']}; cost = {k: 0 for k in res}
for npy, g in V.groupby('npy'):
    lo, hi = max(0, g.t0.min() - 3), g.t1.max() + 3
    a = np.load(npy, mmap_mode='r'); y = np.asarray(a[int(lo * SR):int(hi * SR)], np.float32) / 32768; dur = len(y) / SR
    for name, hop in [('grid3', 3.0), ('grid05', 0.5)]:
        st = list(np.arange(0, max(dur - 3, 0) + 1e-6, hop)); sc = birdnet(y, st); cost[name] += len(st); st = np.array(st)
        for r in g.itertuples():
            s0, s1 = r.t0 - lo, r.t1 - lo; ov = (st < s1) & (st + 3 > s0)
            res[name].append(float(sc[ov].max()) if ov.any() else 0.0)
    st = [max(0.0, min(r.t0 - lo - 0.5, dur - 3)) for r in g.itertuples()]; sc = birdnet(y, st); cost['onset'] += len(st)
    res['onset'] += list(map(float, sc))
out = {k: {f'recall@{t}': round(float((np.array(v) >= t).mean()), 3) for t in (0.2, 0.3, 0.5)} | {'windows': cost[k]} for k, v in res.items()}
out['n_songs'] = len(V); print(json.dumps(out, indent=1)); json.dump(out, open(f'{WS}/results/bn_hop_check.json', 'w'), indent=1)
