"""Re-score song_index segments that the corpus BirdNET pass (3 s grid, 3 s hop) scored < 0.3, with the chosen
window scheme (3 windows at onset -1.0/-0.5/0.0 s, max).  bn_hop_check2: recall@0.3 0.741 -> 0.816 at FP 0.051 -> 0.071.
Writes song_index.parquet with yh = max(old, new) and yh_grid3 (old value) kept; backup song_index.pre_rescore.parquet.
Run in the TF venv (ai_edge_litert).  Usage: rescore_yh.py [n_procs]"""
from config import WS
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
SR = 32000; OFFS = (-1.0, -0.5, 0.0)


def job(args):
    from processing.jcu_pipeline import birdnet
    npy, segs = args
    a = np.load(npy, mmap_mode='r'); dur = len(a) / SR; out = []
    lo = max(0.0, min(s for s, _ in segs) - 1.5); hi = min(dur, max(s for s, _ in segs) + 3.5)
    y = np.asarray(a[int(lo * SR):int(hi * SR)], np.float32) / 32768; d = len(y) / SR
    st = [max(0.0, min(s0 - lo + o, d - 3)) for s0, _ in segs for o in OFFS]
    sc = birdnet(y, st).reshape(len(segs), len(OFFS)).max(1)
    return [(i, float(v)) for (_, i), v in zip(segs, sc)]


if __name__ == '__main__':
    S = pd.read_parquet(f'{WS}/data/song_index.parquet')
    if not os.path.exists(f'{WS}/data/song_index.pre_rescore.parquet'): S.to_parquet(f'{WS}/data/song_index.pre_rescore.parquet')
    if 'yh_grid3' not in S: S['yh_grid3'] = S.yh
    m = S.yh_grid3.notna() & (S.yh_grid3 < 0.3) & S.source.isin(['ncs', 'vizina', 'audiomoth_bhbl'])
    todo = [(npy, [(s0, i) for s0, i in zip(g.seg_t0, g.index)]) for npy, g in S[m].groupby('npy')]
    print('segments', int(m.sum()), 'files', len(todo), flush=True)
    new = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        for k, res in enumerate(p.imap_unordered(job, todo, chunksize=4)):
            new.update(res)
            if k % 500 == 0: print(k, len(new), flush=True)
    idx = np.array(list(new)); S.loc[idx, 'yh'] = np.maximum(S.loc[idx, 'yh_grid3'].values, np.array([new[i] for i in idx]))
    S.to_parquet(f'{WS}/data/song_index.parquet')
    gained = (S.loc[idx, 'yh'] >= 0.3).sum()
    print('rescored', len(idx), 'now >= 0.3:', int(gained), S.loc[idx].assign(g=S.loc[idx, 'yh'] >= 0.3).groupby('source').g.sum().to_dict())
