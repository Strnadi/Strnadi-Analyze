"""Unlabelled song pool for semi-supervised training (FixMatch): all detected songs, fold-tagged (NCS hashed folds),
Vizina train males only, capped per recording.  JCU train recorders appended when jcu_train.csv exists."""
from config import WS
import os, numpy as np, pandas as pd

S = pd.read_parquet(f'{WS}/data/song_index.parquet')
S = S[((S.yh >= 0.3) | S.yh.isna()) & ((S.seg_t1 - S.seg_t0) <= 3.8) & (S.seg_t0 >= 0.6)]
S = S[(S.source != 'vizina') | (S.split == 'train')]
S = S.sample(frac=1, random_state=0).groupby('rec').head(20)
S = S.assign(t0=S.seg_t0 - 0.6, t1=S.seg_t0 + 3.8, name=S.rec + '@' + S.seg_t0.round(2).astype(str))[['npy', 't0', 't1', 'fold', 'rec', 'source', 'name']]
if os.path.exists(f'{WS}/data/jcu_train.csv'):
    J = pd.read_csv(f'{WS}/data/jcu_train.csv'); J = J[J.label != 'None'].sample(frac=1, random_state=0).groupby('rec').head(20)
    S = pd.concat([S, J[['npy', 't0', 't1', 'fold', 'rec', 'source', 'name']]], ignore_index=True)
S.to_csv(f'{WS}/data/unlabeled_v2.csv', index=False); print(S.source.value_counts().to_dict(), len(S))
