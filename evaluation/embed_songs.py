"""Out-of-fold embeddings and class probabilities for every detected song of every NCS recording, grouped by
recording: the input of the bird-level model (training/bird_level.py) and of bird-level aggregation.

Each song is scored by the fold model that never saw its recording (folds from the weak table). Per song: the
128-d clip embedding and the 7 class probabilities (averaged over 3 onset-aligned windows).
Writes results/song_emb_<run>.npz with arrays emb (S, D), prob (S, 7), rec (S,), fold (S,), and the
recording-level label per song (`bag_label`: the single dialect in the song's candidate set, or '').

Usage: python -m evaluation.embed_songs <run> [--table data/weak_v3.csv]
"""

import argparse
import os
import numpy as np
import torch
from config import WS, DIALECTS
from training.checkpoint import load_model
from training.data import read_table
from training.inference import windows

OFFS = (
    0.0,
    0.2,
    0.4,
)  # weak windows start 0.6 s before onset: these are onset -0.6 / -0.4 / -0.2 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--table', default='data/weak_v3.csv')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', 4)))
    W = read_table(a.table)
    W = W[(W.source == 'ncs') & W.label.str.contains('|', regex=False)].reset_index(drop=True)
    emb, prob = np.zeros((len(W), 128), np.float32), np.zeros((len(W), 7), np.float32)
    for k in sorted(W.fold.unique()):
        pt = f'{WS}/runs/{a.run}/fold{int(k)}/model.pt'
        if not os.path.exists(pt):
            raise SystemExit(f'missing {pt}')
        m = load_model(pt)
        rows = np.where(W.fold.values == k)[0]
        for i in range(0, len(rows), 64):
            sel = rows[i : i + 64]
            X = torch.from_numpy(windows(W.iloc[sel], OFFS))
            with torch.no_grad():
                out = m(X)
            emb[sel] = out['emb'].view(len(sel), len(OFFS), -1).mean(1).numpy()
            prob[sel] = (
                torch.softmax(out['logits'], 1).view(len(sel), len(OFFS), -1).mean(1).numpy()
            )
        print('fold', int(k), len(rows), 'songs', flush=True)
    bag = W.label.map(
        lambda s: (
            ([c for c in s.split('|') if c in DIALECTS] + [''])[0]
            if sum(c in DIALECTS for c in s.split('|')) == 1
            else ''
        )
    )
    np.savez(
        f'{WS}/results/song_emb_{a.run}.npz',
        emb=emb,
        prob=prob,
        rec=W.rec.values.astype(str),
        fold=W.fold.values.astype(int),
        bag_label=bag.values.astype(str),
    )
    print(
        'saved',
        f'{WS}/results/song_emb_{a.run}.npz',
        len(W),
        'songs',
        W.rec.nunique(),
        'recordings',
    )


if __name__ == '__main__':
    main()
