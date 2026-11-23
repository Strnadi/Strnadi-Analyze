"""Label-free error rate on held-out sets whose songs have a candidate set (e.g. JCU test recorders: BhBl|Unfinished|None).
Uses the fold ensemble of a run (the set must be held out from all folds).  Reports: rate of predictions outside the
candidate set (always an error), prediction distribution, and the same restricted to confident songs (BirdNET yh >= 0.5).
Usage: eval_constrained.py <run> <table.csv> [max_rows=4000]"""

import sys, os, json, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.inference import TorchEnsemble

run, table = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
df = pd.read_csv(table, keep_default_na=False, na_values=[''])
if len(df) > mx:
    df = df.sample(mx, random_state=0)
model = TorchEnsemble(f'{WS}/runs/{run}')
P = []
for i in range(0, len(df), 64):
    X = np.stack(
        [load_seg(r.npy, r.t0 + 0.2, r.t0 + 4.2, 128000) for r in df.iloc[i : i + 64].itertuples()]
    )
    X = X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5
    P.append(model(X.astype(np.float32)))
P = np.concatenate(P)
pred = np.array(CLASSES)[P.argmax(1)]
allowed = df.label.str.split('|').values
out_set = np.array([p not in a for p, a in zip(pred, allowed)])
conf = (df.yh.values >= 0.5) if 'yh' in df else np.ones(len(df), bool)
res = dict(
    run=run,
    table=os.path.basename(table),
    n=int(len(df)),
    outside_rate=float(out_set.mean()),
    outside_rate_confident=float(out_set[conf].mean()),
    pred_dist={c: float((pred == c).mean()) for c in CLASSES},
)
print(json.dumps(res))
json.dump(
    res, open(f'{WS}/results/constrained_{os.path.basename(table)[:-4]}_{run}.json', 'w'), indent=1
)
