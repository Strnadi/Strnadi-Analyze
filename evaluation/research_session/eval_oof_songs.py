"""Out-of-fold evaluation on song tables whose rows carry a CV fold (expert_eval.csv, ...).
Each song is scored by runs/<run>/fold<f>/model.pt with f = the song's fold (that model never trained on the recording).
Usage: eval_oof_songs.py <run> <table.csv>"""

import sys, os, json, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.checkpoint import load_logits_model as load_torch_model

run, table = sys.argv[1], sys.argv[2]
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
df = pd.read_csv(table)
out = []
for f, g in df.groupby('fold'):
    pt = f'{WS}/runs/{run}/fold{int(f)}/model.pt'
    if not os.path.exists(pt):
        continue
    m = load_torch_model(pt)
    for o in [-0.25, 0.0, 0.25]:
        X = np.stack([load_seg(r.npy, r.t0 + o, r.t1 + o, 128000) for r in g.itertuples()])
        X = X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5
        with torch.no_grad():
            P = torch.cat(
                [
                    torch.softmax(m(torch.from_numpy(X[i : i + 64].astype(np.float32))), 1)
                    for i in range(0, len(X), 64)
                ]
            ).numpy()
        out.append(pd.DataFrame(P, columns=CLASSES, index=g.index).assign(off=o))
if not out:
    print('no fold models')
    sys.exit()
P = pd.concat(out).groupby(level=0)[CLASSES].max()  # slide-max over offsets
d = df.loc[P.index]
pred = P.values.argmax(1)
y = d.label.map(C2I).values
res = {'n': int(len(d)), 'acc': float((pred == y).mean())}
for c in sorted(d.label.unique()):
    m = (d.label == c).values
    res[c] = {
        'recall': float((pred[m] == C2I[c]).mean()),
        'pred': {CLASSES[k]: int((pred[m] == k).sum()) for k in range(7) if (pred[m] == k).sum()},
    }
# dialect-only decision (ignore None/Unfinished): BE vs BlBh among dialect classes
dia = [C2I[c] for c in ['BC', 'BE', 'BhBl', 'BlBh', 'XB']]
pd_ = np.array(dia)[P.values[:, dia].argmax(1)]
res['dialect_only_acc'] = float((pd_ == y).mean())
print(run, os.path.basename(table), json.dumps(res))
json.dump(res, open(f'{WS}/results/oof_{os.path.basename(table)[:-4]}_{run}.json', 'w'), indent=1)
