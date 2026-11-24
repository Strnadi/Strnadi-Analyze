"""Grouped-CV evaluation of frozen foundation embeddings with different heads.

heads: logreg (pooled emb), proto (cosine nearest class-mean), mlp (pooled), temporal (1D conv + attention over the
16-step Perch spatial embedding -> tests whether 'overlap' is due to pooling away temporal order).
"""

import sys, os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from config import CLASSES, C2I, WS
from training.metrics import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(int(os.environ.get('THREADS', 8)))
model_name = sys.argv[1] if len(sys.argv) > 1 else 'perch'
df = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
Z = np.load(f'{WS}/emb/orig_{model_name}.npz')
E = Z['emb'].astype(np.float32)
S = Z['spatial'].astype(np.float32) if 'spatial' in Z else None
y = df.label.map(C2I).values


class TempHead(nn.Module):
    def __init__(self, din, d=128, nc=7):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d), nn.GELU(), nn.Dropout(0.2))
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.att = nn.Linear(d, 1)
        self.out = nn.Linear(2 * d, nc)

    def forward(self, x):  # (B,T,D)
        h = self.inp(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        a = torch.softmax(self.att(h), 1)
        mu = (h * a).sum(1)
        sd = ((h - mu[:, None]) ** 2 * a).sum(1).clamp_min(1e-6).sqrt()
        return self.out(torch.cat([mu, sd], 1))


class MLPHead(nn.Module):
    def __init__(self, din, d=256, nc=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, d), nn.GELU(), nn.Dropout(0.3), nn.Linear(d, nc)
        )

    def forward(self, x):
        return self.net(x)


def fit_torch(head, Xtr, ytr, Xva, epochs=60, lr=1e-3):
    cnt = np.bincount(ytr, minlength=7) + 1
    w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            loss = nn.functional.cross_entropy(
                head(Xtr_t[b]), ytr_t[b], weight=w, label_smoothing=0.05
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(torch.tensor(Xva)), 1).numpy()


res = {}
heads = ['logreg', 'proto', 'mlp'] + (['temporal'] if S is not None else [])
for h in heads:
    P = np.zeros((len(df), 7))
    mask = np.zeros(len(df), bool)
    for k in range(5):
        tr = (df.fold != k).values
        va = ((df.fold == k) & (~df.aug)).values
        if h == 'logreg':
            sc = StandardScaler().fit(E[tr])
            clf = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced')
            clf.fit(sc.transform(E[tr]), y[tr])
            P[va] = clf.predict_proba(sc.transform(E[va]))
        elif h == 'proto':
            En = E / np.linalg.norm(E, axis=1, keepdims=True)
            mu = np.stack([En[tr & (y == c)].mean(0) for c in range(7)])
            mu /= np.linalg.norm(mu, axis=1, keepdims=True)
            sim = En[va] @ mu.T * 20
            P[va] = np.exp(sim - sim.max(1, keepdims=True))
            P[va] /= P[va].sum(1, keepdims=True)
        elif h == 'mlp':
            torch.manual_seed(k)
            P[va] = fit_torch(MLPHead(E.shape[1]), E[tr], y[tr], E[va])
        elif h == 'temporal':
            torch.manual_seed(k)
            P[va] = fit_torch(TempHead(S.shape[2]), S[tr], y[tr], S[va], epochs=40)
        mask |= va
    m = metrics(y[mask], P[mask])
    res[h] = m
    print(
        f'{model_name:8s} {h:9s} macroF1 {m["macro_f1"]:.3f} acc {m["acc"]:.3f} balacc {m["bal_acc"]:.3f} dialectF1 {m["dialect_f1_on_dialect_clips"]:.3f}  recall:',
        {c: round(v, 2) for c, v in m['per_class_recall'].items()},
        flush=True,
    )
os.makedirs(f'{WS}/results', exist_ok=True)
json.dump(res, open(f'{WS}/results/frozen_{model_name}.json', 'w'), indent=1)
