"""Bird/recording-level out-of-fold evaluation on NCS recordings (weak_v2 hashed folds, expert overrides).

For recordings in fold f, every detected song is scored with runs/<run>/fold<f>/model.pt (window onset-0.4 s .. +3.6 s).
Recording decision = argmax over dialects of the summed dialect probabilities of songs whose best dialect prob >= tau;
recordings with no such song are 'undetermined' (counted in coverage, not accuracy).
Usage: eval_recordings.py <run> [tau=0.5] [max_songs_per_rec=30]
"""

import sys, os, json, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.checkpoint import load_logits_model as load_torch_model

run = sys.argv[1]
tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 30
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
W = pd.read_csv(f'{WS}/data/weak_v2.csv', keep_default_na=False, na_values=[''])
W = W[(W.source == 'ncs') & W.label.str.contains('|', regex=False)].copy()
W['rec_label'] = W.label.str.split('|').str[0]
W = W[
    ~W.label.str.startswith('BE|BhBl') & ~W.label.str.startswith('BE|BlBh')
]  # ambiguous BD/transitional excluded
W = W.sample(frac=1, random_state=0).groupby('rec').head(cap)
DIA = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
di = [C2I[c] for c in DIA]
res_rows = []
for f, g in W.groupby('fold'):
    pt = f'{WS}/runs/{run}/fold{int(f)}/model.pt'
    if not os.path.exists(pt):
        continue
    m = load_torch_model(pt)
    P = []
    for i in range(0, len(g), 64):
        X = np.stack(
            [
                load_seg(r.npy, r.t0 + 0.2, r.t0 + 4.2, 128000)
                for r in g.iloc[i : i + 64].itertuples()
            ]
        )
        X = X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5
        with torch.no_grad():
            P.append(torch.softmax(m(torch.from_numpy(X.astype(np.float32))), 1).numpy())
    g = g.assign(**{c: v for c, v in zip(CLASSES, np.concatenate(P).T)})
    for rec, d in g.groupby('rec'):
        pd_ = d[DIA].values
        conf = pd_.max(1) >= tau
        pred = DIA[int(pd_[conf].sum(0).argmax())] if conf.any() else 'undetermined'
        res_rows.append(
            dict(
                rec=rec,
                fold=f,
                label=d.rec_label.iloc[0],
                pred=pred,
                n_songs=len(d),
                n_conf=int(conf.sum()),
            )
        )
R = pd.DataFrame(res_rows)
det = R[R.pred != 'undetermined']
out = dict(
    run=run,
    n_recs=len(R),
    coverage=float(len(det) / max(1, len(R))),
    acc=float((det.pred == det.label).mean()),
    per_class={
        c: dict(
            n=int((det.label == c).sum()),
            recall=float((det[det.label == c].pred == c).mean())
            if (det.label == c).any()
            else None,
        )
        for c in DIA
    },
    confusion=pd.crosstab(det.label, det.pred).to_dict(),
)
print(json.dumps({k: v for k, v in out.items() if k != 'confusion'}))
print(pd.crosstab(R.label, R.pred))
json.dump(out, open(f'{WS}/results/recs_{run}.json', 'w'), indent=1)
R.to_csv(f'{WS}/results/recs_{run}.csv', index=False)
