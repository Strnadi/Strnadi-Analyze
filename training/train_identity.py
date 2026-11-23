"""Bird identity embedding for diarization ('which bird sang which song'), trained on Vizina males.

Songs: song_index Vizina rows whose file names a single male (ring code); split by male (held-out males never seen).
Batches: P males x K songs (K from different files when possible).  Loss: sub-center cosine margin (ArcFace-style, CosFace
margin) over male IDs + SupCon on the same embeddings.
Eval (held-out males): verification EER over song pairs; synthetic diarization: for each trial take songs of 2-3 random
held-out males, cluster embeddings (agglomerative, cosine, threshold chosen on training males) -> purity, n-birds accuracy.
Usage: train_identity.py <name> <threads> [steps=6000]
"""
import os, sys, json, time, math, zlib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from model import build_model
from training.data import load_seg
from config import WS
from augment.random_eq import random_eq as rand_eq
from losses.supcon import supcon
name, threads = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
torch.set_num_threads(threads); rng = np.random.default_rng(0)
S = pd.read_parquet(f'{WS}/data/song_index.parquet'); S = S[(S.source == 'vizina') & (S.yh >= 0.3) & ((S.seg_t1 - S.seg_t0) <= 3.8) & (S.seg_t0 >= 0.6)]
vf = pd.read_csv(f'{WS}/data/vizina_files.csv', dtype={'inds': str}); one = vf[vf.inds.fillna('').str.count(';') == 0].dropna(subset=['inds'])
S = S.assign(male=S.npy.map(dict(zip(one.npy, one.inds)))).dropna(subset=['male'])
cnt = S.groupby('male').size(); S = S[S.male.isin(cnt[cnt >= 20].index)]
males = sorted(S.male.unique()); test_m = {m for m in males if zlib.crc32(m.encode()) % 4 == 0}
tr = S[~S.male.isin(test_m)].reset_index(drop=True); te = S[S.male.isin(test_m)].reset_index(drop=True)
mid = {m: i for i, m in enumerate(sorted(tr.male.unique()))}
print('males train', len(mid), 'test', len(test_m), 'songs', len(tr), len(te), flush=True)
noise = list(np.load(f'{WS}/data/noise_bank_audiomoth.npy', allow_pickle=True))
N = 128000


def get(row, aug):
    t0 = row.seg_t0 - np.random.uniform(0.2, 0.6) if aug else row.seg_t0 - 0.4
    y = load_seg(row.npy, t0, t0 + 4, N)
    if aug:
        if np.random.rand() < 0.6: y = rand_eq(y)
        if np.random.rand() < 0.6:
            nz = noise[np.random.randint(len(noise))].astype(np.float32); s = np.random.randint(0, len(nz) - N); nz = nz[s:s + N]
            snr = np.random.uniform(0, 20); y = y + nz * np.sqrt(((y ** 2).mean() + 1e-9) / ((nz ** 2).mean() + 1e-9) / 10 ** (snr / 10))
    return (y / max(np.abs(y).max(), 1e-6) * 0.5).astype(np.float32)


model = build_model('strnadnet', n_classes=len(mid), head='cosine', head_kw=dict(k=2, s=24.0))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
by_m = tr.groupby('male').indices; P_, K_ = 12, 4; t0 = time.time()
for step in range(steps):
    ms = rng.choice(list(by_m), P_, replace=False); rows = []
    for m in ms: rows += list(rng.choice(by_m[m], K_, replace=len(by_m[m]) < K_))
    X = torch.from_numpy(np.stack([get(tr.iloc[i], True) for i in rows])); y = torch.tensor([mid[tr.male.iloc[i]] for i in rows])
    out = model(X)
    logits = out['logits'] - 24.0 * 0.25 * F.one_hot(y, len(mid)).float()
    loss = F.cross_entropy(logits, y) + 0.5 * supcon(out['emb'], y, 0.1)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sched.step()
    if step % 100 == 0: print(step, round(loss.item(), 3), round(time.time() - t0), flush=True)
model.eval()


def embed(df):
    E = []
    with torch.no_grad():
        for i in range(0, len(df), 64):
            X = torch.from_numpy(np.stack([get(r, False) for r in df.iloc[i:i + 64].itertuples()]))
            E.append(F.normalize(model(X)['emb'], dim=1).numpy())
    return np.concatenate(E)


te_s = te.sample(frac=1, random_state=0).groupby('male').head(60).reset_index(drop=True)
Et = embed(te_s); lab = te_s.male.to_numpy(dtype=object)
sim = Et @ Et.T; iu = np.triu_indices(len(Et), 1); same = (lab[:, None] == lab[None])[iu]; sc = sim[iu]
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(same, sc); eer_i = np.nanargmin(np.abs(fpr - (1 - tpr))); eer = float((fpr[eer_i] + 1 - tpr[eer_i]) / 2)
# synthetic diarization on held-out males
from sklearn.cluster import AgglomerativeClustering
tr_s = tr.sample(frac=1, random_state=0).groupby('male').head(30).reset_index(drop=True)
Etr = embed(tr_s); best = None
for th in np.linspace(0.2, 0.9, 15):  # tune distance threshold on training males
    acc = []
    for trial in range(40):
        ms = rng.choice(tr_s.male.unique(), rng.integers(1, 4), replace=False); idx = np.concatenate([rng.choice(np.where(tr_s.male == m)[0], 8) for m in ms])
        cl = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric='cosine', linkage='average').fit_predict(Etr[idx])
        acc.append(len(set(cl)) == len(ms))
    if best is None or np.mean(acc) > best[1]: best = (th, np.mean(acc))
res_n, pur = [], []
for trial in range(200):
    ms = rng.choice(te_s.male.unique(), rng.integers(1, 4), replace=False); idx = np.concatenate([rng.choice(np.where(te_s.male == m)[0], 8) for m in ms])
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=best[0], metric='cosine', linkage='average').fit_predict(Et[idx])
    res_n.append(len(set(cl)) == len(ms)); l = lab[idx]
    pur.append(sum(pd.Series(l[cl == c]).value_counts().iloc[0] for c in set(cl)) / len(idx))
res = dict(name=name, males_train=len(mid), males_test=len(test_m), verification_EER=eer, diar_nbirds_acc=float(np.mean(res_n)), diar_purity=float(np.mean(pur)), threshold=float(best[0]))
print(json.dumps(res)); os.makedirs(f'{WS}/identity', exist_ok=True)
json.dump(res, open(f'{WS}/results/identity_{name}.json', 'w'), indent=1); torch.save(dict(state=model.state_dict(), n_males=len(mid)), f'{WS}/identity/{name}.pt')
