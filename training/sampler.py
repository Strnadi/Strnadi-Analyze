"""Sampling weights: class balance inside each data pool, then a fixed share of draws per pool.

Without pools, the 2,000 "Unfinished" clips or one heavily sampled site (Vizina / JCU are BhBl-only) dominate,
and synthetic counterfactuals swamp the 79 real BlBh clips (independent-review finding V5).
"""
import numpy as np


def pool_sampling_weights(tr, shares, balance=0.5):
    """tr: training DataFrame with columns label, pool (+ optional hard, sw). Returns (weights, realised shares)."""
    main = tr.apply(lambda r: r['hard'] if r.label == 'soft' else r.label.split('|')[0], axis=1) if 'hard' in tr \
        else tr.label.map(lambda s: s.split('|')[0])
    row_w = tr['sw'].fillna(1.0).values if 'sw' in tr else np.ones(len(tr))
    pools = list(tr.pool.unique()); extra = [p for p in pools if p != 'strong']
    default = 0.5 / len(extra) if extra else 0.0
    share = {p: shares.get(p, default) for p in extra}
    share['strong'] = max(0.05, 1 - sum(share.values()))
    w = np.zeros(len(tr))
    for p in pools:
        m = (tr.pool == p).values
        counts = main[m].value_counts()
        wp = main[m].map(lambda c: counts[c] ** (-balance)).values * row_w[m]
        w[m] = wp / wp.sum() * share[p]
    return w, {p: round(float(w[(tr.pool == p).values].sum()), 3) for p in pools}
