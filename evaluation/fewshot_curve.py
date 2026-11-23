"""Few-shot learning curves for frozen embeddings (BioAcu per-project head scenario).
For n in shots: sample n training clips per class (from training folds, one clip per recording group where
possible), fit logistic / prototype heads, evaluate macro-F1 on the held-out fold; 5 folds x 3 seeds."""
import sys, os, json, numpy as np, pandas as pd
from config import CLASSES, C2I, WS
from training.metrics import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

emb_file, tag = sys.argv[1], sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else 'emb'
df = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
Z = np.load(emb_file); E = Z[key].astype(np.float32)
if E.ndim == 3: E = E.mean(1)
y = df.label.map(C2I).values
res = {}
for n in [4, 8, 16, 32, 64, 'all']:
    for head in ['logreg', 'proto']:
        scores = []
        for k in range(5):
            for seed in range(3 if n != 'all' else 1):
                rng = np.random.default_rng(seed)
                tr_pool = df[(df.fold != k) & ~df.aug]
                if n == 'all': tr = tr_pool.index.values
                else:
                    tr = []
                    for c in CLASSES:
                        g = tr_pool[tr_pool.label == c].groupby('grp').apply(lambda d: d.index[rng.integers(len(d))])
                        tr += list(rng.choice(g.values, size=min(n, len(g)), replace=False))
                    tr = np.array(tr)
                va = df[(df.fold == k) & ~df.aug].index.values
                if head == 'logreg':
                    sc = StandardScaler().fit(E[tr]); clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced').fit(sc.transform(E[tr]), y[tr])
                    P = clf.predict_proba(sc.transform(E[va]))
                else:
                    En = E / np.linalg.norm(E, axis=1, keepdims=True)
                    mu = np.stack([En[tr][y[tr] == c].mean(0) for c in range(7)]); mu /= np.linalg.norm(mu, axis=1, keepdims=True)
                    s = En[va] @ mu.T * 20; P = np.exp(s - s.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
                scores.append(metrics(y[va], P)['macro_f1'])
        res[f'{head}_{n}'] = (float(np.mean(scores)), float(np.std(scores)))
        print(tag, head, n, 'macroF1 %.3f ± %.3f' % res[f'{head}_{n}'], flush=True)
json.dump(res, open(f'{WS}/results/fewshot_{tag}.json', 'w'), indent=1)
