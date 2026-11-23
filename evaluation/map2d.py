"""2-D maps of clip embeddings for the expert 'draw a circle around the dialect' tool.

Methods: PCA, LDA (supervised linear), UMAP, supervised UMAP (fit on training folds only, transform held-out),
t-SNE (no out-of-sample transform; shown for reference only).
Metrics on held-out fold (grouped): kNN (k=15) accuracy & macro-F1 in 2-D, silhouette of dialect clusters.
Also HDBSCAN on the high-D embedding: ARI / NMI vs dialect labels (label-free discovery check).
Usage: map2d.py <emb.npz|emb.npy> <tag> [key]
"""
import sys, os, json, numpy as np, pandas as pd
from config import CLASSES, C2I, WS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

f, tag = sys.argv[1], sys.argv[2]; key = sys.argv[3] if len(sys.argv) > 3 else 'emb'
df = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
E = np.load(f)[key] if f.endswith('.npz') else np.load(f)
E = E.astype(np.float32)
if E.ndim == 3: E = E.mean(1)
keep = (~df.aug).values
df, E = df[keep].reset_index(drop=True), E[keep]
y = df.label.map(C2I).values
dial = np.isin(y, [C2I[c] for c in ['BC', 'BE', 'BhBl', 'BlBh', 'XB']])
k = 0; tr = (df.fold != k).values; va = ~tr
Es = StandardScaler().fit(E[tr]).transform(E)
res = {}; maps = {}
import umap
methods = {
    'pca': lambda: PCA(2).fit(Es[tr]),
    'lda': lambda: LinearDiscriminantAnalysis(n_components=2).fit(Es[tr], y[tr]),
    'umap': lambda: umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, n_jobs=2).fit(Es[tr]),
    'umap_sup': lambda: umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, n_jobs=2, target_weight=0.5).fit(Es[tr], y=y[tr]),
}
for name, fit in methods.items():
    m = fit(); Z = m.transform(Es)
    knn = KNeighborsClassifier(15).fit(Z[tr], y[tr]); p = knn.predict(Z[va])
    res[name] = dict(knn_acc=float((p == y[va]).mean()), knn_macro_f1=float(f1_score(y[va], p, average='macro')),
                     knn_dialect_f1=float(f1_score(y[va][dial[va]], p[dial[va]], average='macro')),
                     silhouette_dialects=float(silhouette_score(Z[va][dial[va]], y[va][dial[va]])))
    maps[name] = Z; print(tag, name, {a: round(b, 3) for a, b in res[name].items()}, flush=True)
# label-free discovery: HDBSCAN on PCA-50 of dialect clips
from sklearn.cluster import HDBSCAN
Zp = PCA(50).fit_transform(Es[dial])
for mcs in [15, 30]:
    cl = HDBSCAN(min_cluster_size=mcs).fit_predict(Zp)
    res[f'hdbscan_{mcs}'] = dict(n_clusters=int(cl.max() + 1), noise=float((cl < 0).mean()), ARI=float(adjusted_rand_score(y[dial], cl)), NMI=float(normalized_mutual_info_score(y[dial], cl)))
    print(tag, f'hdbscan_{mcs}', res[f'hdbscan_{mcs}'], flush=True)
json.dump(res, open(f'{WS}/results/map2d_{tag}.json', 'w'), indent=1)
fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 6))
cols = {'BC': '#d4c100', 'BE': '#24581a', 'BhBl': '#61c1da', 'BlBh': '#19268e', 'XB': '#fe0a04', 'Unfinished': '#b0b0b0', 'None': '#000000'}
for ax, (name, Z) in zip(axes, maps.items()):
    for c in ['Unfinished', 'None', 'BC', 'BE', 'XB', 'BhBl', 'BlBh']:
        m = (y == C2I[c]) & va
        ax.scatter(Z[m, 0], Z[m, 1], s=6 if c in ('Unfinished', 'None') else 12, c=cols[c], label=c, alpha=0.7)
    ax.set_title(f'{tag} {name} (held-out fold)'); ax.legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.savefig(f'{WS}/figs/map2d_{tag}.png', dpi=60)
