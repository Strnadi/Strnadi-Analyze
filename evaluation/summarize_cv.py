"""Aggregate out-of-fold predictions of a CV run: overall + per-source metrics, confusion matrix."""
import sys, os, glob, json, numpy as np, pandas as pd
from config import CLASSES, C2I, WS
from training.metrics import metrics

for name in sys.argv[1:]:
    files = sorted(glob.glob(f'{WS}/runs/{name}/fold*/oof.npz'))
    if not files: print(name, 'no folds'); continue
    data = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    P, Y, I = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True); P.append(z['p']); Y.append(z['y']); I.append(z['idx'])
    P, Y, I = np.concatenate(P), np.concatenate(Y), np.concatenate(I)
    m = metrics(Y, P)
    src = data.loc[I, 'src'].values
    m['by_source'] = {s: {k: v for k, v in metrics(Y[src == s], P[src == s]).items() if k in ('acc', 'macro_f1')} for s in np.unique(src)}
    m['folds'] = len(files)
    os.makedirs(f'{WS}/results', exist_ok=True); json.dump(m, open(f'{WS}/results/cv_{name}.json', 'w'), indent=1)
    print(f"{name:28s} folds {len(files)}  macroF1 {m['macro_f1']:.3f}  acc {m['acc']:.3f}  balacc {m['bal_acc']:.3f}  dialectF1 {m['dialect_f1_on_dialect_clips']:.3f}")
    print('   recall', {c: round(v, 2) for c, v in m['per_class_recall'].items()})
    print('   by source', m['by_source'])
    if '-cm' in os.environ.get('SHOW', ''):
        print(pd.DataFrame(m['cm'], index=CLASSES, columns=CLASSES))
