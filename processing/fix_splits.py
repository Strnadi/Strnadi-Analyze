"""Single source of truth for splits/folds in song_index / none_index (post-review fix V1-V4).
- NCS rows with fold -1 get crc32('ncs:F')%5 (same hash as build_weak_set) -> pseudo-labelling, None windows, SSL all consistent
- Vizina rows get split from vizina_files.csv (train / test / exclude)
"""

from config import WS
import zlib, shutil, pandas as pd

vf = pd.read_csv(f'{WS}/data/vizina_files.csv')
vsplit = dict(zip(vf.npy, vf.split))
for f in ['song_index', 'none_index']:
    p = f'{WS}/data/{f}.parquet'
    shutil.copy(p, p.replace('.parquet', '.pre_review.parquet'))
    S = pd.read_parquet(p)
    m = (S.source == 'ncs') & (S.fold < 0)
    S.loc[m, 'fold'] = S.loc[m, 'rec'].map(lambda r: zlib.crc32(r.encode()) % 5)
    v = S.source == 'vizina'
    S.loc[v, 'split'] = S.loc[v, 'npy'].map(vsplit).fillna('exclude')
    S.to_parquet(p)
    print(
        f,
        'ncs fold -1 left:',
        int(((S.source == 'ncs') & (S.fold < 0)).sum()),
        '| vizina split',
        S[v].split.value_counts().to_dict(),
    )
