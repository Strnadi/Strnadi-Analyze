from config import WS
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from training.metrics import metrics
from config import C2I
df = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=['']); df = df[~df.aug].reset_index(drop=True)
F = pd.read_parquet(f'{WS}/results/parse_feats_v0.parquet').drop(columns=['label'])
X = F.values.astype(float); y = df.label.map(C2I).values; P = np.zeros((len(df), 7))
for k in range(5):
    tr = df.fold.values != k; va = ~tr
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=0).fit(X[tr], y[tr]); P[va] = clf.predict_proba(X[va])
m = metrics(y, P); print('parse-features GBM: macroF1 %.3f acc %.3f dialectF1 %.3f' % (m['macro_f1'], m['acc'], m['dialect_f1_on_dialect_clips'])); print({c: round(v, 2) for c, v in m['per_class_recall'].items()})
np.save(f'{WS}/results/gbm_parse_oof.npy', P)
