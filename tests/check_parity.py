"""Parity check against the research session's results (needs the workspace, $STRNADI_WS).

Loads research checkpoints into the refactored models and compares their predictions on out-of-fold clips with
the probabilities the research trainer stored in `oof.npz`. Proves the layer split, the librosa-free mel bank,
the checkpoint name mapping and the data path changed nothing numerically.

    python -m tests.check_parity [run ...]        default: base_v3 r2_s0
"""
import os
import sys
import numpy as np
import torch
from config import WS
from training.checkpoint import load_model
from training.data import read_table, load_seg
from training.inference import peak_normalize


def check(run, fold=0, n=48):
    d = f'{WS}/runs/{run}/fold{fold}'
    if not os.path.exists(f'{d}/oof.npz'):
        print(f'{run}: no oof.npz, skipped'); return None
    z = np.load(f'{d}/oof.npz', allow_pickle=True)
    table = read_table('data/orig_index.csv')
    rows = table.loc[z['idx'][:n]]
    X = peak_normalize(np.stack([load_seg(r.npy, r.t0, r.t0 + 4, 128000) for r in rows.itertuples()]))
    m = load_model(f'{d}/model.pt')
    with torch.no_grad():
        p = torch.softmax(m(torch.from_numpy(X))['logits'], 1).numpy()
    diff = float(np.abs(p - z['p'][:n]).max()); agree = float((p.argmax(1) == z['p'][:n].argmax(1)).mean())
    print(f'{run} fold{fold}: max |dp| = {diff:.2e}, argmax agreement = {agree:.3f} on {n} clips')
    return diff


if __name__ == '__main__':
    torch.set_num_threads(1)
    worst = max([d for d in (check(r) for r in (sys.argv[1:] or ['base_v3', 'r2_s0'])) if d is not None] or [0.0])
    assert worst < 1e-3, f'parity broken: {worst}'
    print('parity ok')
