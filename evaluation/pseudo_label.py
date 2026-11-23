"""Noisy-student pseudo-labelling of the song index with a fold teacher.

For fold k, teacher = runs/<teacher>/fold<k>/model.pt (never trained on fold k's recordings).
Pool = all songs (min_yh filter) except Vizina test individuals and NCS recordings of fold k.
Prediction = mean over 3 time-shifted 4 s crops (TTA).  For songs with a candidate set (weak label) the teacher's
distribution is renormalised within the set (+eps elsewhere); then sharpened (T) and kept if max prob >= thr.
Output: data/pl_<teacher>_f<k>.csv  (columns compatible with train_cv --extra, label = 'soft' + p_* columns)
"""

import os, sys, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.checkpoint import load_logits_model as load_torch_model

teacher, k = sys.argv[1], int(sys.argv[2])
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
T = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
max_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
torch.set_num_threads(int(os.environ.get('THREADS', 8)))
S = pd.read_parquet(f'{WS}/data/song_index.parquet')
S = S[
    (S.split != 'test')
    & ((S.yh >= 0.3) | S.yh.isna())
    & ((S.seg_t1 - S.seg_t0) <= 3.8)
    & (S.seg_t0 >= 0.6)
]
S = S[~((S.source == 'ncs') & (S.fold == k))]
# cap songs per recording to keep long files from dominating
S = S.sample(frac=1, random_state=0).groupby('rec').head(40)
if len(S) > max_rows:
    S = S.sample(max_rows, random_state=0)
S = S.reset_index(drop=True)
model = load_torch_model(f'{WS}/runs/{teacher}/fold{k}/model.pt')
P = np.zeros((len(S), len(CLASSES)), np.float32)
offs = [-0.5, -0.4, -0.25]
B = 64
with torch.no_grad():
    for i in range(0, len(S), B):
        rows = S.iloc[i : i + B]
        acc = 0
        for o in offs:
            X = np.stack(
                [load_seg(r.npy, r.seg_t0 + o, r.seg_t0 + o + 4, 128000) for r in rows.itertuples()]
            )
            X = X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5
            acc = acc + torch.softmax(model(torch.from_numpy(X.astype(np.float32))), 1).numpy()
        P[i : i + B] = acc / len(offs)
        if i % (B * 50) == 0:
            print(i, len(S), flush=True)
np.save(f'{WS}/data/pl_raw_{teacher}_f{k}.npy', P)
Q = P.copy()
for j, w in enumerate(S.weak.values):
    if w:
        m = np.full(len(CLASSES), 0.05 if S.source.values[j] == 'ncs' else 0.0, np.float32)
        for c in w.split('|'):
            m[C2I[c]] = 1
        Q[j] = P[j] * m
Q = Q / Q.sum(1, keepdims=True)
Q = Q ** (1 / T)
Q = Q / Q.sum(1, keepdims=True)
keep = Q.max(1) >= thr
out = S[keep].copy()
out['t0'] = out.seg_t0 - 0.6
out['t1'] = out.t0 + 4.4
out['label'] = 'soft'
out['aug'] = False
out['name'] = out.rec + '@' + out.t0.round(2).astype(str)
for c, i in C2I.items():
    out[f'p_{c}'] = Q[keep, i]
out['hard'] = [CLASSES[i] for i in Q[keep].argmax(1)]
out['w'] = 1.0
out['fold'] = -1
n = out.groupby('rec').npy.transform('size')
out['sw'] = 1 / np.sqrt(n)
out.drop(columns=['weak']).to_csv(f'{WS}/data/pl_{teacher}_f{k}.csv', index=False)
print('pool', len(S), 'kept', int(keep.sum()))
print(pd.crosstab(out.source, out.hard))
