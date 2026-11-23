"""Re-align the strong NCS clips to the whole song (QA sheets: many strong clips cut the song at the 4 s edge).

The original 4 s clips were cut from the NCS recordings on a 0.5 s grid (BirdNET windows).  For each clip:
 1. locate it in the source recording: normalised correlation on the 0.5 s grid (decimated), refined +-60 ms at 32 kHz;
 2. tonal events around it (segment_songs), merged into songs (gap < 0.6 s, <= 4 s) as in the song-level pipeline;
    the clip's song = the merged event with the largest overlap with the clip;
 3. status: inside (song fully in the clip) / start_cut / end_cut / both_cut / no_song / unlocated;
 4. new window = song onset - 0.6 s, 4.4 s long, in the source recording.  Kept unchanged (original clip): None clips,
    no_song, unlocated, songs > 3.8 s, and Unfinished clips whose song ran past the clip end (the 'unfinished' may
    have been the cut itself; listed for expert audit instead of being moved).
Row order/index identical to orig_index.csv (OOF idx stay valid).  Output: data/orig_aligned.csv + status counts.
Usage: realign_orig.py [n_procs]"""

from config import WS
import os, sys, glob, numpy as np, pandas as pd
from multiprocessing import Pool
from processing.segment import segment_songs

SR = 32000
CL = 4 * SR
DIA = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
SRC = {os.path.basename(p).split('.')[0].upper(): p for p in glob.glob(f'{WS}/audio32k/ncs/*.npy')}


def ncorr(a, c):
    """normalised correlation of c against every position of a (valid)"""
    from scipy.signal import fftconvolve

    num = fftconvolve(a, c[::-1], 'valid')
    e = np.sqrt(np.maximum(fftconvolve(a**2, np.ones(len(c)), 'valid'), 1e-12)) * np.sqrt(
        (c**2).sum() + 1e-12
    )
    return num / e


def locate(a, c):
    d = 8
    a8, c8 = a[::d], c[::d]
    r = ncorr(a8, c8)
    grid = np.arange(
        0, len(r), int(0.5 * SR / d)
    )  # 0.5 s grid first, then anywhere if the grid fails
    k = grid[np.argmax(np.abs(r[grid]))] if len(grid) else 0
    if abs(r[k]) < 0.8:
        k = int(np.argmax(np.abs(r)))
    lo = max(0, k * d - int(0.06 * SR))
    seg = a[lo : lo + len(c) + int(0.12 * SR)]
    r2 = ncorr(seg, c)
    j = int(np.argmax(np.abs(r2)))
    return (lo + j) / SR, float(abs(r2[j]))


def songs(y):
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.4, max_dur=6.0)
    ev = []
    for s0, s1, _ in segs:
        if ev and s0 - ev[-1][1] < 0.6 and s1 - ev[-1][0] <= 4.0:
            ev[-1][1] = s1
        else:
            ev.append([s0, s1])
    return [e for e in ev if e[1] - e[0] >= 0.5]


def job(r):
    out = dict(idx=r['idx'], status='unlocated', off=np.nan, corr=np.nan, s0=np.nan, s1=np.nan)
    F = r['name'].split('.')[0].upper()
    sp = SRC.get(F)
    if sp is None:
        return out
    try:
        a = np.load(sp).astype(np.float32) / 32768
        c = np.load(r['npy']).astype(np.float32) / 32768
    except Exception:
        return out
    off, cor = locate(a, c)
    out.update(off=off, corr=cor)
    if cor < 0.9:
        return out
    lo = max(0.0, off - 4.0)
    y = a[int(lo * SR) : int((off + 8.0) * SR)]
    best, bo = None, 0.0
    for s0, s1 in songs(y):
        s0, s1 = s0 + lo, s1 + lo
        o = min(s1, off + 4) - max(s0, off)
        if o > bo:
            best, bo = (s0, s1), o
    if best is None or bo < 0.3:
        out['status'] = 'no_song'
        return out
    s0, s1 = best
    out.update(s0=s0, s1=s1, src=sp, src_len=len(a) / SR)
    sc, ec = s0 < off + 0.05, s1 > off + 3.95
    out['status'] = {
        (False, False): 'inside',
        (True, False): 'start_cut',
        (False, True): 'end_cut',
        (True, True): 'both_cut',
    }[(sc, ec)]
    return out


if __name__ == '__main__':
    d = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    todo = [
        dict(idx=i, name=r['name'], npy=r['npy'])
        for i, r in d.iterrows()
        if r['src'] == 'ncs' and not r['aug']
    ]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        res = pd.DataFrame(p.map(job, todo, chunksize=8)).set_index('idx')
    A = d.copy()
    A['align_status'] = 'orig_kept'
    A['src_off'] = np.nan
    A.loc[res.index, 'align_status'] = res.status
    A.loc[res.index, 'src_off'] = res.off
    A.loc[res.index, 'align_corr'] = res['corr']
    A['song_s0'] = np.nan
    A['song_s1'] = np.nan
    A.loc[res.index, 'song_s0'] = res.s0
    A.loc[res.index, 'song_s1'] = res.s1
    moved = 0
    for i, r in res.iterrows():
        lab = A.at[i, 'label']
        if r.status not in ('inside', 'start_cut', 'end_cut', 'both_cut') or lab == 'None':
            continue
        if r.s1 - r.s0 > 3.8:
            A.at[i, 'align_status'] = r.status + '_toolong'
            continue
        if lab == 'Unfinished' and r.status in ('end_cut', 'both_cut'):
            A.at[i, 'align_status'] = r.status + '_unf_kept'
            continue  # keep the original clip + label (eval needs a strong label)
        t0 = max(0.0, min(r.s0 - 0.6, r.src_len - 4.4))
        A.at[i, 'npy'] = r.src
        A.at[i, 't0'] = t0
        A.at[i, 't1'] = t0 + 4.4
        moved += 1
    # augmented copies (aug=True) follow their parent clip only if the training code uses them; leave untouched
    A.to_csv(f'{WS}/data/orig_aligned.csv', index=False)
    print('moved windows', moved)
    print(pd.crosstab(A.align_status, A.label))
    print('corr quantiles', res['corr'].quantile([0.01, 0.05, 0.5]).round(3).to_dict())
