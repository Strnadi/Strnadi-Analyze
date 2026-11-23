"""Vizina external set: annotated songs (BhBl complete / Unfinished) split by *individual*.
~30% of individuals -> 'test' (never used for training); every Vizina file whose name mentions a test individual
is marked test as well (so unannotated recordings of test birds are excluded from weak training)."""
from config import SRC, WS
import os, re, zlib, glob, numpy as np, pandas as pd

V = pd.read_csv(f'{WS}/data/vizina_songs.csv', dtype={'indiv': str, 'song_code': str})
V['indiv'] = V.indiv.str.replace(r'\(\?\)', '', regex=True).str.strip()
V = V[V.indiv.str.fullmatch(r'\d{2}\.\d{2}', na=False)]  # canonical ring codes only ('?', 'unknown' dropped)
root = f'{WS}/audio32k/vizina/'
V['npy'] = V.audio.fillna('').map(lambda p: root + os.path.relpath(p, f'{SRC}/source/Vizina_YH_all_rec') + '.npy' if p else '')
V = V[V.npy.map(lambda p: bool(p) and os.path.exists(p))].copy()
inds = sorted(V.indiv.astype(str).unique())
test_inds = {i for i in inds if zlib.crc32(i.encode()) % 10 < 3}
V['split'] = np.where(V.indiv.astype(str).isin(test_inds), 'test', 'train')
V['win_t0'] = V.t0 - 0.4; V['win_t1'] = V.win_t0 + 4.0
V.to_csv(f'{WS}/data/vizina_eval.csv', index=False)
# file-level split for all Vizina recordings (indiv codes appear at the start of the filename, e.g. "12.19, 03.20 ECB ...")
man = pd.read_csv(f'{WS}/audio32k/manifest.csv')
vf = man[man.source == 'vizina'].copy()
def file_inds(rel):
    head = os.path.basename(rel).split('ECB')[0]
    return set(re.findall(r'\d{2}\.\d{2}', head))
vf['inds'] = vf.rel.map(file_inds)
# files naming a test male -> test; files with no parseable male code -> 'exclude' (could contain test males)
vf['split'] = vf.inds.map(lambda s: 'test' if s & test_inds else ('exclude' if not s else 'train'))
vf[['npy', 'rel', 'dur', 'split']].assign(inds=vf.inds.map(lambda s: ';'.join(sorted(s)))).to_csv(f'{WS}/data/vizina_files.csv', index=False)
print('individuals', len(inds), 'test', len(test_inds))
print(pd.crosstab(V.label, V.split))
print('vizina files by split', vf.split.value_counts().to_dict(), 'hours', vf.groupby('split').dur.sum().div(3600).round(1).to_dict())
