"""Leakage-free CV folds for the labelled clip set.

Group = source recording (NCS F-id) or BirdNET-app device; folds are stratified on each group's rarest label.
Corrections over the research version:

--dedupe      byte-identical recordings filed under different F-ids are merged into one group first (the audit
              found 107 such groups; 81 spanned folds, some with conflicting labels). Duplicates are detected by
              hashing the decoded audio of each source recording.
--group_map   CSV with columns `grp,key`: recordings sharing a key (observer, site, session, ...) are kept in the
              same fold. Recording-grouped folds do not separate observers (knowing the observer predicts the
              dialect with 0.81 accuracy), so an observer- or site-grouped run is the honest robustness check.
              (No complete recording -> observer table exists yet: strnadi.cz has ObsOwner per observation and
              the F-id <-> observation mapping is only known for ~630 recordings.)

Usage: python -m processing.make_folds [--table data/orig_index.csv] [--out same] [--dedupe] [--group_map map.csv]
"""

import argparse
import glob
import hashlib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from config import WS
from config import workspace_path

RANK = {'BlBh': 0, 'BhBl': 1, 'None': 2, 'BE': 3, 'BC': 4, 'XB': 5, 'Unfinished': 6}


def audio_hash(npy, chunk=1 << 20):
    a = np.load(npy, mmap_mode='r')
    h = hashlib.md5(str(len(a)).encode())
    h.update(np.ascontiguousarray(a[:chunk]).tobytes())
    mid = len(a) // 2
    h.update(np.ascontiguousarray(a[mid : mid + chunk]).tobytes())
    return h.hexdigest()


def duplicate_groups():
    """F-id -> representative F-id for byte-identical decoded NCS recordings."""
    by_hash = {}
    for p in sorted(glob.glob(f'{WS}/audio32k/ncs/*.npy')):
        by_hash.setdefault(audio_hash(p), []).append(os.path.basename(p).split('.')[0].upper())
    rep = {f: fs[0] for fs in by_hash.values() if len(fs) > 1 for f in fs}
    print(
        f'duplicate recordings: {len(rep)} files in {sum(len(v) > 1 for v in by_hash.values())} groups',
        flush=True,
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='data/orig_index.csv')
    ap.add_argument('--out')
    ap.add_argument('--dedupe', action='store_true')
    ap.add_argument('--group_map')
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(workspace_path(a.table), keep_default_na=False, na_values=[''])
    df['grp'] = np.where(
        df.src == 'birdnet_mobile', df.group.str.replace(r'_\d+$', '', regex=True), df.group
    )
    key = df.grp.copy()
    if a.dedupe:
        rep = duplicate_groups()
        key = key.map(lambda g: rep.get(str(g).split('.')[0].upper(), g))
    if a.group_map:
        gm = pd.read_csv(a.group_map)
        key = key.map(dict(zip(gm.grp, gm.key))).fillna(key)
    df['fold_group'] = key
    aug = df.aug if 'aug' in df else pd.Series(False, index=df.index)
    base = df[~aug]
    glab = base.groupby('fold_group').label.agg(
        lambda s: min(s, key=lambda x: RANK.get(x.split('|')[0], 9))
    )
    best = None
    for seed in range(20):  # keep the most balanced of 20 stratified splits
        sgk = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=seed)
        g, y = glab.index.values, glab.values
        fold = {}
        for k, (_, te) in enumerate(sgk.split(g, y, g)):
            fold.update({g[i]: k for i in te})
        ct = pd.crosstab(base.label, base.fold_group.map(fold))
        score = (ct.max(1) - ct.min(1)).div(ct.mean(1)).sum()
        if best is None or score < best[0]:
            best = (score, fold, ct)
    df['fold'] = df.fold_group.map(best[1])
    df.to_csv(workspace_path(a.out or a.table), index=False)
    print(best[2])
    print('groups', df.fold_group.nunique(), 'rows without a fold', int(df.fold.isna().sum()))


if __name__ == '__main__':
    main()
