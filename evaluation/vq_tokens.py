"""VQ tokenizer of song frames (HuBERT-style acoustic units) + label-free dialect discovery.

fit   : k-means codebook (K units) on frame vectors sampled from training songs (NCS-free corpus).
        stage 0 = normalised log-mel frames (pooled to 40 ms, 128 bins);  stage 1 = frame features of an encoder
        (e.g. the HuBERT-trained ssl model) -> iterate like HuBERT.
discover: every labelled clip -> token sequence -> run-length encoded 'element' sequence -> ending signature
        (bag of the last song elements) -> cluster signatures (k-means / HDBSCAN) and compare to dialect labels (ARI,
        NMI, purity); also a kNN classifier on signatures (how much dialect info do raw tokens carry, label-free).
Usage: vq_tokens.py fit <out.npz> <K> [encoder.pt]      vq_tokens.py discover <codebook.npz> [encoder.pt]
"""
import os, sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from training.data import load_seg
from config import WS, CLASSES, C2I
from layers.registry import build_frontend
from model import build_model
from training.checkpoint import load_encoder_state
torch.set_num_threads(int(os.environ.get('THREADS', 4)))
FE = build_frontend()
N = 128000


def frames(X, enc=None):
    """X (B,N) waveform -> (B,100,D) frame vectors"""
    with torch.no_grad():
        S = FE(torch.from_numpy(X))  # (B,1,128,400)
        if enc is None:
            f = F.avg_pool2d(S, (1, 4)).squeeze(1).transpose(1, 2)  # (B,100,128)
        else:
            f = enc.frames(S).transpose(1, 2)  # (B,100,d)
        f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-5)
    return f.numpy()


def load_enc(pt):
    if not pt: return None
    return load_encoder_state(build_model('strnadnet'), torch.load(pt, map_location='cpu', weights_only=False)['state']).eval()


def clip_batch(rows, t0col='t0'):
    X = np.stack([load_seg(r.npy, getattr(r, t0col), getattr(r, t0col) + 4, N) for r in rows.itertuples()])
    return (X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5).astype(np.float32)


def fit(out, K, enc_pt=''):
    from sklearn.cluster import MiniBatchKMeans
    enc = load_enc(enc_pt)
    S = pd.read_parquet(f'{WS}/data/song_index.parquet')
    S = S[S.source.isin(['vizina', 'audiomoth_bhbl', 'birdnet_yh']) & ((S.split == 'train') | (S.source != 'vizina')) & ((S.yh >= 0.3) | S.yh.isna()) & (S.seg_t0 >= 0.6)]
    S = S.sample(6000, random_state=0).assign(t0=lambda d: d.seg_t0 - 0.4)
    V = []
    for i in range(0, len(S), 64):
        f = frames(clip_batch(S.iloc[i:i + 64]), enc); V.append(f.reshape(-1, f.shape[-1])[::3])
    V = np.concatenate(V); print('frames', V.shape, flush=True)
    km = MiniBatchKMeans(K, batch_size=8192, n_init=3, random_state=0).fit(V)
    np.savez(out, centroids=km.cluster_centers_.astype(np.float32), encoder=enc_pt)
    print('saved', out, 'inertia', km.inertia_)


def tokens(f, C):
    d = (f ** 2).sum(-1, keepdims=True) - 2 * f @ C.T + (C ** 2).sum(-1)[None, None]
    return d.argmin(-1)  # (B,100)


def discover(cb, enc_pt=''):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    C = np.load(cb)['centroids']; K = len(C); enc = load_enc(enc_pt)
    d = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=['']); d = d[~d.aug].reset_index(drop=True)
    T = []
    for i in range(0, len(d), 64):
        T.append(tokens(frames(clip_batch(d.iloc[i:i + 64]), enc), C))
    T = np.concatenate(T)
    # background unit = most frequent token overall; element runs = run-length encoding without background
    bg = np.bincount(T.ravel(), minlength=K).argmax()
    sig = np.zeros((len(T), K), np.float32)
    for j, t in enumerate(T):
        runs = [t[0]] + [t[k] for k in range(1, len(t)) if t[k] != t[k - 1]]
        el = [u for u in runs if u != bg][-6:]  # last ~6 element units = the ending
        for w, u in enumerate(el): sig[j, u] += 1 + w / 6  # later units weigh more
    sig /= np.maximum(sig.sum(1, keepdims=True), 1e-6)
    y = d.label.map(C2I).values; dia = np.isin(y, [C2I[c] for c in ['BC', 'BE', 'BhBl', 'BlBh', 'XB']])
    res = {'K': K}
    for nc in [5, 10, 20]:
        cl = KMeans(nc, n_init=5, random_state=0).fit_predict(sig[dia])
        res[f'kmeans{nc}'] = dict(ARI=float(adjusted_rand_score(y[dia], cl)), NMI=float(normalized_mutual_info_score(y[dia], cl)),
                                  purity=float(sum(np.bincount(y[dia][cl == c]).max() for c in set(cl)) / dia.sum()))
    tr = (d.fold != 0).values; va = ~tr
    knn = KNeighborsClassifier(15, metric='cosine').fit(sig[tr & dia], y[tr & dia]); p = knn.predict(sig[va & dia])
    from sklearn.metrics import f1_score
    res['knn_dialect_macroF1_fold0'] = float(f1_score(y[va & dia], p, average='macro'))
    print(json.dumps(res)); json.dump(res, open(f'{WS}/results/vq_discover_{os.path.basename(cb)[:-4]}.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'fit': fit(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else '')
    else: discover(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
