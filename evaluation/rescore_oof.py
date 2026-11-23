"""Re-score the fold models of a finished run on another clip table (same rows, different windows).

Two runs can only be compared like for like when they were evaluated on the same windows: for
example R2 has to be re-scored on the re-aligned windows that R7 is evaluated on. The result is
written as a run of its own, `runs/<run>@<table stem>/fold<k>/oof.npz`, which
`evaluation/compare_runs.py` accepts as a run name.

Usage: python -m evaluation.rescore_oof <run> <table.csv> [eval_window=start]
"""

import glob
import os
import sys

import numpy as np
import torch

from config import C2I, WS
from training.checkpoint import load_model
from training.data import read_table
from training.settings import TrainSettings
from training.validation import predict_clips


def main():
    run = sys.argv[1]
    table_path = sys.argv[2]
    eval_window = sys.argv[3] if len(sys.argv) > 3 else 'start'

    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    settings = TrainSettings(eval_window=eval_window)

    table = read_table(table_path)
    if 'aug' in table:
        table = table[~table['aug']]
    table_stem = os.path.basename(table_path).replace('.csv', '')

    for checkpoint in sorted(glob.glob(f'{WS}/runs/{run}/fold*/model.pt')):
        fold = int(os.path.basename(os.path.dirname(checkpoint)).replace('fold', ''))
        held_out = table[table.fold == fold]
        labels = held_out.label.map(C2I).values

        model = load_model(checkpoint)
        probabilities = predict_clips(model, held_out, settings)

        out_dir = f'{WS}/runs/{run}@{table_stem}/fold{fold}'
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            f'{out_dir}/oof.npz',
            p=probabilities,
            y=labels,
            idx=held_out.index.values,
            name=held_out.name.values,
            strong=table_path,
            eval_crop=eval_window,
        )
        accuracy = float((probabilities.argmax(1) == labels).mean())
        print(f'fold {fold}: accuracy {accuracy:.3f}', flush=True)


if __name__ == '__main__':
    main()
