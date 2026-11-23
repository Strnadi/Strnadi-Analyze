"""Bank of other-species vocalisations for background mixing (augment/species_mix.py, experiment r8).

Takes hard-negative windows (BirdNET: another species >= 0.5, Yellowhammer < 0.02) from Vizina TRAINING
recordings only, so the bank shares no audio with any CV fold of the citizen data or with the held-out males.

Usage: python -m processing.build_species_bank [--n 1500] [--out data/species_bank.npy]
"""

import argparse
import numpy as np
import pandas as pd
from config import workspace_path
from training.data import load_seg

N = 128000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--out', default='data/species_bank.npy')
    a = ap.parse_args()
    H = pd.read_csv(workspace_path('data/hardneg_v1.csv'), keep_default_na=False, na_values=[''])
    H = H[H.source == 'vizina']
    H = H.sample(min(a.n, len(H)), random_state=0)
    bank = np.empty(len(H), dtype=object)
    bank[:] = [load_seg(r.npy, r.t0, r.t0 + 4, N).astype(np.float16) for r in H.itertuples()]
    np.save(workspace_path(a.out), bank, allow_pickle=True)
    print(len(bank), 'windows;', H.other.value_counts().head(10).to_dict())


if __name__ == '__main__':
    main()
