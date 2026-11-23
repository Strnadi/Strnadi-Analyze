"""JCU AudioMoth songs (compact stores) -> weak training rows + held-out passive-domain test set.

7 of 35 recorders (hash-selected) are held out entirely (test); the rest give weak rows 'BhBl|Unfinished|None'
(site has only BhBl) and 'None' windows.  Test metric (label-free): wrong-dialect rate = share of songs predicted as
BC/BE/BlBh/XB (always an error at this site) and BhBl rate.
Usage: build_jcu_set.py <out_train.csv> <out_test.csv>
"""
from config import WS
import sys, glob, zlib, numpy as np, pandas as pd

fs = glob.glob(f'{WS}/data/jcu_index/*.parquet')
J = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
devs = sorted(J.dev.unique()); USED_ELSEWHERE = {'AUD27', 'AUD32'}  # local AudioMoth sample (weak train, noise bank, SSL)
all35 = [f'AUD{i}' for i in range(1, 36)]
test_devs = set(sorted([d for d in all35 if d not in USED_ELSEWHERE], key=lambda d: zlib.crc32(d.encode()))[:7])
J['day'] = J.src_file.str.extract(r'_(\d{8})_')[0]
J['rec'] = 'jcu:' + J.dev + '/' + J.day
songs = J[(J.kind == 'song') & (J.yh >= 0.3) & (J.seg_t1 - J.seg_t0 <= 3.8) & (J.seg_t0 >= 0.6)].copy();  # full song inside 4.4 s window
songs['t0'] = songs.seg_t0 - 0.6; songs['t1'] = songs.t0 + 4.4; songs['label'] = 'BhBl|Unfinished|None'
songs['song_t0'] = songs.seg_t0; songs['song_t1'] = songs.seg_t1
nones = J[J.kind == 'none'].copy(); nones['t0'] = nones.seg_t0; nones['t1'] = nones.seg_t1; nones['label'] = 'None'
E = pd.concat([songs, nones], ignore_index=True)
E['source'] = 'jcu'; E['aug'] = False; E['fold'] = -1; E['name'] = E.rec + '@' + E.src_t.round(2).astype(str)
E['w'] = np.where(E.label == 'None', 0.5, 1.0)
E['sw'] = 1 / np.sqrt(E.groupby('rec').npy.transform('size'))
E['split'] = np.where(E.dev.isin(test_devs), 'test', 'train')
cols = ['npy', 't0', 't1', 'label', 'fold', 'aug', 'name', 'w', 'sw', 'source', 'rec', 'dev', 'yh', 'split', 'song_t0', 'song_t1']
tr = E[E.split == 'train']; te = E[(E.split == 'test') & (E.label != 'None')]
# keep weak JCU at a modest share of draws: scale sw so JCU ~ 15 % of weak draws
tr = tr.assign(sw=tr.sw * 0.15 * 50000 / max(tr.sw.sum(), 1e-9) / 1.0)
tr[cols].to_csv(sys.argv[1], index=False); te[cols].to_csv(sys.argv[2], index=False)
print('devices', len(devs), 'test devices', sorted(test_devs)); print(E.groupby(['split', 'label']).size())
