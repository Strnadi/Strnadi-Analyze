"""Re-score the fold models of a finished run on another clip table (same rows, different windows), so two runs
can be compared like for like (e.g. R2 on the re-aligned windows that R7 is evaluated on).

Writes runs/<run>@<table stem>/fold<k>/oof.npz, which evaluation/compare_runs.py accepts as a run name.

Usage: python -m evaluation.rescore_oof <run> <table.csv> [eval_crop=start]
"""
import glob
import os
import sys
import numpy as np
import torch
from config import WS, C2I
from training.checkpoint import load_model
from training.data import read_table, ClipDataset


def main():
    run, table = sys.argv[1], sys.argv[2]; crop = sys.argv[3] if len(sys.argv) > 3 else 'start'
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    data = read_table(table); stem = os.path.basename(table).replace('.csv', '')
    for pt in sorted(glob.glob(f'{WS}/runs/{run}/fold*/model.pt')):
        k = int(os.path.basename(os.path.dirname(pt)).replace('fold', ''))
        aug = data['aug'] if 'aug' in data else False
        va = data[(data.fold == k) & ~aug]
        m = load_model(pt)
        dl = torch.utils.data.DataLoader(ClipDataset(va, 4.0, False, dict(eval_crop=crop)), batch_size=64)
        with torch.no_grad():
            p = np.concatenate([torch.softmax(m(b['wave'])['logits'], 1).numpy() for b in dl])
        out = f'{WS}/runs/{run}@{stem}/fold{k}'; os.makedirs(out, exist_ok=True)
        np.savez(f'{out}/oof.npz', p=p, y=va.label.map(C2I).values, idx=va.index.values, name=va.name.values, strong=table, eval_crop=crop)
        print('fold', k, 'acc', float((p.argmax(1) == va.label.map(C2I).values).mean()), flush=True)


if __name__ == '__main__':
    main()
