"""Post-hoc reliability analysis of a CV run's out-of-fold predictions (cross-fitted: fold k calibrated on the others).
  - temperature scaling: ECE (15 bins) before/after
  - split-conformal prediction sets (LAC score) at 90 % and 95 %: empirical coverage, mean set size, singleton rate,
    and for dialect clips how often the set is a single dialect
  - selective prediction: accuracy vs. fraction auto-accepted (max prob threshold); fraction acceptable at 95/98 % acc.
Usage: calibrate.py <run[,seed_run...]>"""
import sys, os, glob, json, numpy as np, pandas as pd
from config import CLASSES, C2I, WS
from scipy.optimize import minimize_scalar
runs = sys.argv[1].split(',')
acc = {}
for r in runs:
    for f in glob.glob(f'{WS}/runs/{r}/fold*/oof.npz'):
        z = np.load(f, allow_pickle=True); k = int(f.split('fold')[1].split('/')[0])
        for i, p in zip(z['idx'], z['p']): acc.setdefault(int(i), (k, []))[1].append(p)
data = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
idx = np.array(sorted(acc)); fold = np.array([acc[i][0] for i in idx]); P = np.stack([np.mean(acc[i][1], 0) for i in idx])
y = data.loc[idx].label.map(C2I).values; L = np.log(np.clip(P, 1e-8, 1))


def ece(p, y, nb=15):
    c = p.max(1); ok = p.argmax(1) == y; e = 0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(e)


def soft(L, T):
    z = L / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


Pc = np.zeros_like(P); sets = {0.9: [], 0.95: []}; res = {'ece_raw': ece(P, y)}
for k in np.unique(fold):
    tr, te = fold != k, fold == k
    T = minimize_scalar(lambda t: -np.log(soft(L[tr], t)[np.arange(tr.sum()), y[tr]] + 1e-12).mean(), bounds=(0.3, 5), method='bounded').x
    Pc[te] = soft(L[te], T); Ptr = soft(L[tr], T)
    scores = 1 - Ptr[np.arange(tr.sum()), y[tr]]
    for a in sets:
        q = np.quantile(scores, np.ceil((tr.sum() + 1) * a) / tr.sum(), method='higher')
        sets[a].append((np.where(te)[0], Pc[te] >= 1 - q))
res['ece_temp'] = ece(Pc, y)
dia = np.isin(y, [C2I[c] for c in ['BC', 'BE', 'BhBl', 'BlBh', 'XB']])
for a, lst in sets.items():
    S = np.zeros_like(P, bool)
    for ii, s in lst: S[ii] = s
    cov = S[np.arange(len(y)), y].mean(); size = S.sum(1)
    res[f'conformal_{int(a * 100)}'] = dict(coverage=float(cov), mean_set_size=float(size.mean()), singleton=float((size == 1).mean()),
                                            dialect_clips_singleton=float((size[dia] == 1).mean()),
                                            typical_sets=pd.Series([' | '.join(np.array(CLASSES)[s]) for s in S[dia]]).value_counts().head(6).to_dict())
c = Pc.max(1); ok = Pc.argmax(1) == y; order = np.argsort(-c); cum = np.cumsum(ok[order]) / np.arange(1, len(c) + 1)
for target in [0.95, 0.98]:
    good = np.where(cum >= target)[0]; res[f'auto_accept_at_{int(target * 100)}acc'] = float((good.max() + 1) / len(c)) if len(good) else 0.0
print(json.dumps(res, indent=1)); json.dump(res, open(f'{WS}/results/calib_{runs[0]}.json', 'w'), indent=1)
