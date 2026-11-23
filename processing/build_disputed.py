"""Mark the labelled clips whose label the models consistently and confidently contradict, and give them a
candidate-set training label {label, Unfinished} (experiment `disputed_as_candidates`).

A dialect-labelled clip is "disputed" when the out-of-fold probability of its own label, averaged over several
independently trained runs, is below `--p_own` and the averaged prediction is "Unfinished". Out-of-fold means
no model that votes on a clip was trained on it. Both reviews looked at spectrograms of these clips: most show a
first phrase and ONE terminal element, which the project's own rule calls Unfinished.

Writes a copy of the strong table with two extra columns: `disputed` and `train_label` (blank = use `label`).
Evaluation keeps using `label`. Also writes the list for expert re-labelling, most confident disagreements first.

This is a stop-gap until the clips are re-labelled by experts; selecting on model agreement is circular, so never
report metrics on a test set filtered this way.

Usage: python -m processing.build_disputed base_v3 r1_s0 r2_s0 [--table data/orig_aligned.csv] [--p_own 0.1]
"""
import argparse
import glob
import numpy as np
import pandas as pd
from config import WS, CLASSES, C2I, DIALECTS, UNFINISHED_IDX
from training.defaults import ws


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--table', default='data/orig_aligned.csv')
    ap.add_argument('--out'); ap.add_argument('--p_own', type=float, default=0.1)
    a = ap.parse_args()
    df = pd.read_csv(ws(a.table), keep_default_na=False, na_values=['']); votes = {}
    for run in a.runs:
        for f in glob.glob(f'{WS}/runs/{run}/fold*/oof.npz'):
            z = np.load(f, allow_pickle=True)
            for i, p in zip(z['idx'], z['p']):
                votes.setdefault(int(i), []).append(p)
    idx = np.array(sorted(i for i, v in votes.items() if len(v) == len(a.runs)))
    P = np.stack([np.mean(votes[i], 0) for i in idx]); y = df.loc[idx, 'label'].map(C2I).values
    own = P[np.arange(len(idx)), y]
    disputed = df.loc[idx, 'label'].isin(DIALECTS).values & (own < a.p_own) & (P.argmax(1) == UNFINISHED_IDX)
    df['disputed'] = False; df['train_label'] = np.nan
    df.loc[idx[disputed], 'disputed'] = True
    df.loc[idx[disputed], 'train_label'] = df.loc[idx[disputed], 'label'] + '|Unfinished'
    out = ws(a.out or a.table.replace('.csv', '_disputed.csv')); df.to_csv(out, index=False)
    audit = df.loc[idx[disputed], ['name', 'label', 'npy', 't0']].assign(p_own=own[disputed], p_unfinished=P[disputed, UNFINISHED_IDX])
    audit.sort_values('p_own').to_csv(f'{WS}/results/disputed_for_experts.csv', index=False)
    print(f'{int(disputed.sum())} disputed dialect clips of {len(idx)} scored ->', out)
    print(df.loc[idx[disputed]].label.value_counts().to_dict())


if __name__ == '__main__':
    main()
