"""Paired recording-group bootstrap comparison of two CV runs on their common out-of-fold clips.

Each side may be a comma-separated list of runs (seeds) whose OOF probabilities are averaged.

Corrections over the research version (base-session audit):
- it refuses to compare runs that were evaluated on different clip tables or with different evaluation windows
  (R7 uses re-aligned windows; comparing it with R2 as-is would mix the window change with the model change);
- labels and groups come from the table the runs were evaluated on, not always from `orig_index.csv`;
- it prints what the interval does NOT cover: with one seed per side the bootstrap resamples recordings for fixed
  models, so training-seed variance (roughly +-0.01-0.02 macro-F1 here) is not in the interval. Pass several
  seeds per side to include it.

Usage: python -m evaluation.compare_runs <runA[,runA_s1]> <runB[,...]> [n_boot=2000]
"""
import glob
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from config import CLASSES, C2I, WS, DIALECTS, DIALECT_IDX, UNFINISHED_IDX
from training.defaults import ws


def oof(spec):
    acc, meta = {}, set()
    for run in spec.split(','):
        files = glob.glob(f'{WS}/runs/{run}/fold*/oof.npz')
        if not files:
            sys.exit(f'no finished folds for run {run}')
        for f in files:
            z = np.load(f, allow_pickle=True)
            # research-session runs stored neither: they used data/orig_index.csv (or orig_aligned.csv from R7 on) and the 'start' window
            meta.add((str(z['strong']) if 'strong' in z else json.load(open(f'{WS}/runs/{run}/args.json')).get('strong', 'data/orig_index.csv').replace(f'{WS}/', ''),
                      str(z['eval_crop']) if 'eval_crop' in z else 'start'))
            for i, p in zip(z['idx'], z['p']):
                acc.setdefault((run, int(i)), p)
    by_clip = {}
    for (run, i), p in acc.items():
        by_clip.setdefault(i, []).append(p)
    n_runs = len(spec.split(','))
    return {i: np.mean(v, 0) for i, v in by_clip.items() if len(v) == n_runs}, meta, n_runs


def main():
    A, meta_a, na = oof(sys.argv[1]); B, meta_b, nb_ = oof(sys.argv[2]); n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    if len(meta_a | meta_b) != 1:
        sys.exit(f'refusing to compare: runs were evaluated on different tables / windows: {sorted(meta_a)} vs {sorted(meta_b)}.\n'
                 f'Re-score one side on the other\'s table first (evaluation/rescore_oof.py).')
    table = next(iter(meta_a))[0]
    data = pd.read_csv(ws(table), keep_default_na=False, na_values=[''])
    idx = np.array(sorted(set(A) & set(B))); d = data.loc[idx]; y = d.label.map(C2I).values; g = d.grp.values
    PA, PB = np.stack([A[i] for i in idx]), np.stack([B[i] for i in idx]); pa, pb = PA.argmax(1), PB.argmax(1)

    def stats(sel):
        ys = y[sel]; out = {'macro_f1': f1_score(ys, pa[sel], average='macro', labels=range(7)) - f1_score(ys, pb[sel], average='macro', labels=range(7))}
        for c in range(7):
            m = ys == c
            out[CLASSES[c]] = ((pa[sel][m] == c).mean() - (pb[sel][m] == c).mean()) if m.any() else np.nan
        for name in DIALECTS:                              # the decision that actually fails: complete vs Unfinished
            m = (ys == C2I[name]) | (ys == UNFINISHED_IDX)
            if (ys[m] == C2I[name]).any() and (ys[m] == UNFINISHED_IDX).any():
                out[f'auc_complete_{name}'] = (roc_auc_score(ys[m] == C2I[name], PA[sel][m][:, DIALECT_IDX].sum(1))
                                               - roc_auc_score(ys[m] == C2I[name], PB[sel][m][:, DIALECT_IDX].sum(1)))
        return out

    point = stats(np.arange(len(idx)))
    groups = np.unique(g); members = {k: np.where(g == k)[0] for k in groups}; rng = np.random.default_rng(0)
    boots = pd.DataFrame([stats(np.concatenate([members[k] for k in rng.choice(groups, len(groups))])) for _ in range(n_boot)])
    res = {k: dict(diff=float(point[k]), ci95=[float(boots[k].quantile(0.025)), float(boots[k].quantile(0.975))], p_A_better=float((boots[k] > 0).mean()))
           for k in point}
    print(f'{sys.argv[1]}  vs  {sys.argv[2]}   table {table}; common OOF clips {len(idx)}; groups {len(groups)}')
    for k, v in res.items():
        print(f'  {k:18s} A-B {v["diff"]:+.3f}  95% CI [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]  P(A>B)={v["p_A_better"]:.2f}')
    if na == 1 or nb_ == 1:
        print('  NOTE: one seed per side: the interval covers test-set sampling only, not training-seed variance.')
    json.dump(res, open(f'{WS}/results/compare_{sys.argv[1].replace(",", "+")}__{sys.argv[2].replace(",", "+")}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
