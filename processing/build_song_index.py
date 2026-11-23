"""Song index over all unlabelled / weakly-labelled audio.

For every cached file: bootstrap segmenter -> candidate song segments; keep those supported by Yellowhammer
evidence (BirdNET 2.4 window score, AudioMoth BirdNET tables, or the file being a BirdNET YH detection clip).
Each song gets a 4.4 s window span [onset-0.6, onset+3.8] (training draws random 4 s crops whose start lies in
[onset-0.6, onset-0.2], so the whole song incl. terminal phrase stays inside the crop for songs <= 3.8 s).

Weak labels (candidate sets for the partial-label loss):
  NCS file with expert recording dialect D  -> 'D|Unfinished|None'  (BD -> 'BE|BhBl|Unfinished|None')
  Vizina / AudioMoth (BhBl-only site)       -> 'BhBl|Unfinished|None'
  otherwise                                 -> '' (unlabelled; SSL / pseudo-labelling only)
Also emits 'None' windows: >=6 s away from any segment and with BirdNET YH score < 0.02.
"""
from config import WS
import os, re, sys, glob, numpy as np, pandas as pd
from multiprocessing import Pool
from processing.segment import segment_songs


DIA_MAP = {'BC': 'BC', 'BE': 'BE', 'BhBl': 'BhBl', 'BlBh': 'BlBh', 'XlB': 'XB', 'XsB': 'XB', 'BD': 'BE|BhBl'}


def process(job):
    npy, source, rec, weak, fold, split, yh_windows = job
    try:
        a = np.load(npy, mmap_mode='r')
        y = np.asarray(a, np.float32) / 32768
    except Exception:
        return [], []
    dur = len(y) / 32000
    if dur < 1.0: return [], []
    segs, env = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.6, max_dur=6.0)
    songs, nones = [], []
    yw = np.asarray(yh_windows) if yh_windows is not None else None  # (n,2): t, yh  (3 s windows)
    for s0, s1, snr in segs:
        if yw is not None and len(yw):
            ov = (yw[:, 0] < s1) & (yw[:, 0] + 3 > s0)
            yh = float(yw[ov, 1].max()) if ov.any() else 0.0
        else:
            yh = 1.0 if source == 'birdnet_yh' else np.nan
        songs.append((npy, source, rec, s0, s1, snr, yh, weak, fold, split))
    # None windows (only where BirdNET coverage exists)
    if yw is not None and len(yw):
        seg_busy = np.zeros(int(dur) + 2, bool)
        for s0, s1, _ in segs: seg_busy[max(0, int(s0) - 6):int(s1) + 6] = True
        cand = [t for t, p in yw if p < 0.02 and t + 4 < dur and not seg_busy[int(t):int(t) + 5].any()]
        rng = np.random.default_rng(abs(hash(npy)) % 2**32)
        for t in rng.choice(cand, size=min(len(cand), 3), replace=False) if cand else []:
            nones.append((npy, source, rec, float(t), float(t) + 4.0, 0.0, 0.0, 'None', fold, split))
    return songs, nones


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    man = pd.read_csv(f'{WS}/audio32k/manifest.csv'); man = man[man.ok]
    # BirdNET window scores
    bn = pd.concat([pd.read_parquet(p) for p in glob.glob(f'{WS}/birdnet24/*.parquet')]) if glob.glob(f'{WS}/birdnet24/*.parquet') else pd.DataFrame(columns=['npy', 't', 'yh'])
    bn_by = {k: g[['t', 'yh']].values for k, g in bn.groupby('npy')}
    am = pd.read_csv(f'{WS}/data/audiomoth_yh_dets.csv')
    am_by = {k: np.c_[g.t0.values, g.conf.values] for k, g in am.groupby('npy')}
    dia = pd.read_csv(f'{WS}/data/meta/dialects_filtered.csv', keep_default_na=False, na_values=[''])
    dia['F'] = dia.Filename.str.extract(r'^(F\d+)')[0]
    dmap = dict(zip(dia.F, dia.Dialect))
    orig = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    fold_by_grp = orig.groupby('grp').fold.first().to_dict()
    vz = pd.read_csv(f'{WS}/data/vizina_files.csv'); vsplit = dict(zip(vz.npy, vz.split))
    jobs = []
    for r in man.itertuples():
        src = r.source; base = os.path.basename(r.rel)
        weak, fold, split, rec = '', -1, 'train', f'{src}:{r.rel}'
        yw = None
        if src == 'ncs':
            m = re.match(r'(F\d+)', base)
            F = m.group(1) if m else None
            rec = f'ncs:{F or r.rel}'
            d = dmap.get(F) if F else None
            if isinstance(d, str) and d in DIA_MAP: weak = DIA_MAP[d] + '|Unfinished|None'
            fold = int(fold_by_grp.get(F, -1)) if F else -1
            yw = bn_by.get(r.npy, np.zeros((0, 2)))
        elif src == 'vizina':
            weak = 'BhBl|Unfinished|None'; split = vsplit.get(r.npy, 'train'); yw = bn_by.get(r.npy, np.zeros((0, 2)))
        elif src == 'audiomoth_bhbl':
            weak = 'BhBl|Unfinished|None'; yw = am_by.get(r.npy, np.zeros((0, 2)))
            rec = f'am:{base.split("_")[-3] if base.count("_") >= 2 else base}:{r.rel}'
        elif src == 'birdnet_yh':
            rec = f'bnyh:{base}'
        jobs.append((r.npy, src, rec, weak, fold, split, yw))
    print('jobs', len(jobs), flush=True)
    songs, nones = [], []
    with Pool(nproc) as pool:
        for i, (s, n) in enumerate(pool.imap_unordered(process, jobs, chunksize=8)):
            songs += s; nones += n
            if i % 5000 == 0: print(i, len(songs), len(nones), flush=True)
    cols = ['npy', 'source', 'rec', 'seg_t0', 'seg_t1', 'snr', 'yh', 'weak', 'fold', 'split']
    S = pd.DataFrame(songs, columns=cols); N = pd.DataFrame(nones, columns=cols)
    S.to_parquet(f'{WS}/data/song_index.parquet'); N.to_parquet(f'{WS}/data/none_index.parquet')
    print(S.groupby('source').agg(n=('npy', 'size'), recs=('rec', 'nunique'), yh_ge_03=('yh', lambda x: (x >= 0.3).sum())))
    print('None windows', len(N))


if __name__ == '__main__':
    main()
