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
import pandas as pd
import torch

from config import C2I, CLASSES, DIALECT_IDX, WS
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

    report_label_flips(run, table, table_stem)


def report_label_flips(run, table, table_stem):
    """List Unfinished clips that the model calls a dialect on the new windows, but not on the old.

    When a clip is re-cut around the whole song, a terminal element that the old clip cut off can
    come into view. The label "Unfinished" was right for the old clip and is wrong for the new
    window. The research session found 23 such clips among 1,693 moved Unfinished clips (research
    log section 17.7). They belong on the expert audit list, not in a training set.
    """
    old, new, rows = [], [], []
    for old_path in sorted(glob.glob(f'{WS}/runs/{run}/fold*/oof.npz')):
        fold_dir = os.path.basename(os.path.dirname(old_path))
        new_path = f'{WS}/runs/{run}@{table_stem}/{fold_dir}/oof.npz'
        if not os.path.exists(new_path):
            continue
        old_fold = np.load(old_path, allow_pickle=True)
        new_fold = np.load(new_path, allow_pickle=True)
        if not np.array_equal(old_fold['idx'], new_fold['idx']):
            print(f'{fold_dir}: the two tables do not have the same rows; no flip report')
            return
        old.append(old_fold['p'])
        new.append(new_fold['p'])
        rows.append(np.asarray(new_fold['idx'], int))
    if not rows:
        return
    old, new, rows = np.concatenate(old), np.concatenate(new), np.concatenate(rows)

    labels = table.loc[rows, 'label'].map(C2I).values
    was_unfinished = old.argmax(1) == C2I['Unfinished']
    now_dialect = np.isin(new.argmax(1), DIALECT_IDX)
    flips = (labels == C2I['Unfinished']) & was_unfinished & now_dialect
    confident = int((flips & (new.max(1) > 0.8)).sum())
    print(
        f'Unfinished clips predicted Unfinished on the old windows and a dialect on the new ones: '
        f'{int(flips.sum())} ({confident} with confidence above 0.8)'
    )

    os.makedirs(f'{WS}/results', exist_ok=True)
    out_path = f'{WS}/results/unf_flips_{run}_on_{table_stem}.csv'
    pd.DataFrame(
        {
            'idx': rows[flips],
            'name': table.loc[rows[flips], 'name'].values,
            'new_pred': [CLASSES[i] for i in new[flips].argmax(1)],
            'conf': new[flips].max(1),
        }
    ).to_csv(out_path, index=False)
    print('written', out_path)


if __name__ == '__main__':
    main()
