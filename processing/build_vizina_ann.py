"""Vizina annotated songs of TRAINING males as labelled rows (review V9).
complete (dialect_1 + dialect_2)          -> 'BhBl' (hard)
single terminal B annotated (dialect 'B') -> 'Unfinished' (hard)
first phrase only (no dialect annotated)  -> 'BhBl|Unfinished' (partial: annotator may have stopped)"""
from config import WS
import pandas as pd, numpy as np

V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str}); V = V[V.split == 'train'].copy()
da = V.dialect_any.fillna('').astype(str)
V['label'] = np.where(V.label == 'BhBl', 'BhBl', np.where(da.str.contains('B'), 'Unfinished', 'BhBl|Unfinished'))
V['t0'] = V.t0 - 0.6; V['t1'] = V.t0 + 4.4; V = V[V.t0 >= 0]
V['fold'] = -1; V['aug'] = False; V['name'] = 'vizann:' + V.indiv + '@' + V.t0.round(2).astype(str); V['w'] = 1.0; V['sw'] = 1.0
V['source'] = 'vizina_ann'; V['rec'] = 'vind:' + V.indiv
V[['npy', 't0', 't1', 'label', 'fold', 'aug', 'name', 'w', 'sw', 'source', 'rec']].to_csv(f'{WS}/data/vizann_v1.csv', index=False)
print(V.label.value_counts().to_dict())
