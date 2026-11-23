"""Bird-level evaluation against expert re-evaluated dialects (Diblikova et al. 2023, Table S1; BE / BlBh / transitional).
The per-song times in Table S1 are mostly a 6.000 s placeholder (384/430), so song windows are NOT taken from it:
every detected song of the recording (song_index, ncs, complete inside 4.4 s) is scored by the out-of-fold model of
that recording (3 onset-aligned windows, mean), and the bird's dialect = argmax of the mean dialect distribution over
songs judged complete (p(BhBl..XB) mass > 0.5; fallback: all songs).
Usage: eval_expert_birds.py <run[,run2...]> [min_yh=0.3]"""

from config import SRC
import sys, os, json, zlib, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.checkpoint import load_logits_model as load_torch_model

torch.set_num_threads(int(os.environ.get('THREADS', 2)))
runs = sys.argv[1].split(',')
min_yh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
DIA = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
di = [C2I[c] for c in DIA]
T = pd.read_excel(f'{SRC}/external/diblikova2023/Supplementary_Table_S1.xlsx')
T['F'] = T.filename.str.extract(r'^(F\d+)')[0]
T = T.dropna(subset=['F']).drop_duplicates('F')
T['rec'] = 'ncs:' + T.F
W = pd.read_csv(f'{WS}/data/weak_v3.csv', keep_default_na=False, na_values=[''])
wf = W[W.source == 'ncs'].groupby('rec').fold.first().to_dict()
orig = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
of = orig.groupby('grp').fold.first().to_dict()
T['fold'] = [int(of.get(f, wf.get(r, zlib.crc32(r.encode()) % 5))) for f, r in zip(T.F, T.rec)]
bad = [(r, wf[r], f) for r, f in zip(T.rec, T.fold) if r in wf and wf[r] != f]
assert not bad, f'fold mismatch weak vs eval: {bad[:5]}'
S = pd.read_parquet(f'{WS}/data/song_index.parquet')
S = S[
    (S.source == 'ncs')
    & S.rec.isin(T.rec)
    & (S.yh >= min_yh)
    & ((S.seg_t1 - S.seg_t0) <= 3.8)
    & (S.seg_t0 >= 0.6)
]
print('expert birds', len(T), 'with songs', S.rec.nunique(), 'songs', len(S), flush=True)
res_all = {}
for run in runs:
    rows = []
    for k, g in S.assign(fold=S.rec.map(dict(zip(T.rec, T.fold)))).groupby('fold'):
        pt = f'{WS}/runs/{run}/fold{int(k)}/model.pt'
        if not os.path.exists(pt):
            continue
        m = load_torch_model(pt)
        P = 0
        for o in (-0.6, -0.4, -0.2):
            X = np.stack(
                [load_seg(r.npy, r.seg_t0 + o, r.seg_t0 + o + 4, 128000) for r in g.itertuples()]
            )
            X = (X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5).astype(np.float32)
            with torch.no_grad():
                P = (
                    P
                    + torch.cat(
                        [
                            torch.softmax(m(torch.from_numpy(X[i : i + 64])), 1)
                            for i in range(0, len(X), 64)
                        ]
                    ).numpy()
                    / 3
                )
        rows.append(pd.DataFrame(P, columns=CLASSES, index=g.index).assign(rec=g.rec.values))
    D = pd.concat(rows)
    out = []
    for rec, g in D.groupby('rec'):
        p = g[DIA].values
        comp = p.sum(1) > 0.5
        q = p[comp] if comp.any() else p
        q = q / q.sum(1, keepdims=True)
        out.append(
            dict(
                rec=rec,
                pred=DIA[int(q.mean(0).argmax())],
                n_songs=len(g),
                n_complete=int(comp.sum()),
                p_BE=float(q.mean(0)[1]),
                p_BlBh=float(q.mean(0)[3]),
            )
        )
    O = pd.DataFrame(out).merge(T[['rec', 'new_dialect', 'dialect']], on='rec')
    r = {'birds': len(O)}
    for c in ['BE', 'BlBh', 'transitional']:
        m_ = O.new_dialect == c
        r[c] = {'n': int(m_.sum()), 'pred': O[m_].pred.value_counts().to_dict()}
        if c != 'transitional':
            r[c]['recall'] = float((O[m_].pred == c).mean())
    clear = O.new_dialect.isin(['BE', 'BlBh'])
    r['acc_BE_BlBh_birds'] = float((O[clear].pred == O[clear].new_dialect).mean())
    # binary BE vs BlBh decision (ignores BC/BhBl/XB confusions)
    bp = np.where(O.p_BE >= O.p_BlBh, 'BE', 'BlBh')
    r['acc_binary'] = float((bp[clear.values] == O[clear].new_dialect.values).mean())
    # expert vs original NCS file label (how much the experts changed)
    r['old_label_agrees'] = float((O[clear].dialect == O[clear].new_dialect).mean())
    print(run, json.dumps(r), flush=True)
    res_all[run] = r
    O.to_csv(f'{WS}/results/expert_birds_{run}.csv', index=False)
json.dump(res_all, open(f'{WS}/results/expert_birds_{"+".join(runs)}.json', 'w'), indent=1)
