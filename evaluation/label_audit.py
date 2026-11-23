"""Confident-learning audit of the human labels (Northcutt et al. 2021, 'confident joint').

OOF probabilities (averaged over the given runs) -> per-class thresholds t_c = mean p(c | labelled c).
A clip labelled y is flagged if p(y) < t_y and some other class c has p(c) >= t_c; suspicion = max_c p(c) - p(y).
Also lists clips whose recording's expert dialect (Diblikova 2023) contradicts the clip's dialect label.
Usage: label_audit.py <run1,run2,...> <out.csv>
"""
from config import SRC
import sys, os, glob, numpy as np, pandas as pd
from config import CLASSES, C2I, WS
runs = sys.argv[1].split(','); out = sys.argv[2]
data = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
acc = {}
for r in runs:
    for f in glob.glob(f'{WS}/runs/{r}/fold*/oof.npz'):
        z = np.load(f, allow_pickle=True)
        for i, p in zip(z['idx'], z['p']): acc.setdefault(int(i), []).append(p)
idx = np.array(sorted(acc)); P = np.stack([np.mean(acc[i], 0) for i in idx])
d = data.loc[idx].copy(); y = d.label.map(C2I).values
t = np.array([P[y == c, c].mean() if (y == c).any() else 0.5 for c in range(len(CLASSES))])
py = P[np.arange(len(P)), y]
other = P.copy(); other[np.arange(len(P)), y] = -1
oc = other.argmax(1); flag = (py < t[y]) & (other[np.arange(len(P)), oc] >= t[oc])
d['p_label'] = py; d['suggested'] = np.array(CLASSES)[oc]; d['p_suggested'] = other[np.arange(len(P)), oc]
d['suspicion'] = d.p_suggested - d.p_label; d['flag'] = flag
T = pd.read_excel(f'{SRC}/external/diblikova2023/Supplementary_Table_S1.xlsx'); T['F'] = T.filename.str.extract(r'^(F\d+)')[0]
newd = dict(zip(T.F, T.new_dialect)); d['expert_rec_dialect'] = d.grp.map(newd)
d['expert_conflict'] = d.expert_rec_dialect.isin(['BE', 'BlBh']) & d.label.isin(['BC', 'BE', 'BhBl', 'BlBh', 'XB']) & (d.label != d.expert_rec_dialect)
d.sort_values('suspicion', ascending=False).to_csv(out, index=False)
print('clips', len(d), 'flagged', int(flag.sum()), f'({flag.mean():.1%})', 'expert conflicts', int(d.expert_conflict.sum()))
print('thresholds', dict(zip(CLASSES, t.round(2))))
print(pd.crosstab(d[d.flag].label, d[d.flag].suggested))
