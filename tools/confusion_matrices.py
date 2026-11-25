"""Confusion matrices of every finished run on the test folds, against the final labels.

    python -m tools.confusion_matrices [--out figs/confusion_matrices] [run ...]

Every labelled clip is predicted once, by the fold model that never saw its recording (the test
fold of the recording-grouped cross-validation). Whatever label table a run was trained and
scored on, its predictions are matched by clip name to the current table (`data/clips_audited.csv`)
and scored against those labels, so all matrices are comparable. Runs are skipped when they have
fewer than five finished folds, when they cover less than 95 % of the table (older research runs
on other clip sets), or when they are not 7-class classifiers (song detectors, bag models).

Writes into `--out` (relative to the workspace):

    <run>.png        counts, coloured by the share of each row (label), with accuracy and macro-F1
    <run>.csv        the counts (rows = label, columns = predicted)
    all.pdf          every matrix, one page each, in the order of the summary
    summary.csv      run, clips, accuracy, macro-F1, errors, recall per class

Without run names every run under `runs/` is tried. The three trainings of the combined recipe
are also drawn averaged (`final_combined_3_seeds_averaged`).
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, f1_score

from config import CLASSES, WS, workspace_path
from evaluation.report import load_run_predictions
from training.data import read_table

#: Runs that are not 7-class clip classifiers, or are only parts of another procedure.
SKIPPED_PREFIXES = ('detector_', 'recording_mil', 'tune_reference_', 'smoke_', 'gpu_check')
#: Seeds averaged into one extra matrix.
AVERAGED = {
    'final_combined_3_seeds_averaged': ['final_combined', 'final_combined_s1', 'final_combined_s2']
}


def final_predictions(run, final_labels):
    """The run's test-fold predictions re-labelled from the current table (None if unusable)."""
    try:
        predictions = load_run_predictions(run)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if predictions.cv_fold.nunique() < 5:
        return None
    predictions = predictions.drop_duplicates('name').set_index('name')
    names = predictions.index.intersection(final_labels.index)
    if len(names) < 0.95 * len(final_labels):
        return None
    predictions = predictions.loc[names].copy()
    predictions['label'] = final_labels.loc[names]
    return predictions


def scores(labels, predicted):
    counts = confusion_matrix(labels, predicted, labels=CLASSES)
    recall = {
        name: counts[index, index] / max(counts[index].sum(), 1)
        for index, name in enumerate(CLASSES)
    }
    return {
        'clips': int(counts.sum()),
        'accuracy': float(np.trace(counts) / counts.sum()),
        'macro_f1': float(f1_score(labels, predicted, labels=CLASSES, average='macro')),
        'errors': int(counts.sum() - np.trace(counts)),
        **{f'recall_{name}': float(value) for name, value in recall.items()},
    }, counts


def draw(counts, title):
    """Counts in the cells, colour = share of the row (how the clips of one label were answered)."""
    row_share = counts / counts.sum(1, keepdims=True).clip(1)
    figure, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(row_share, cmap='Blues', vmin=0, vmax=1)
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            ax.text(
                column,
                row,
                counts[row, column],
                ha='center',
                va='center',
                fontsize=9,
                color='white' if row_share[row, column] > 0.5 else 'black',
            )
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha='right')
    ax.set_yticks(
        range(len(CLASSES)), [f'{name} ({counts[i].sum()})' for i, name in enumerate(CLASSES)]
    )
    ax.set_xlabel('predicted')
    ax.set_ylabel('label (clips)')
    ax.set_title(title, fontsize=9, loc='left')
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='*')
    parser.add_argument('--out', default='figs/confusion_matrices')
    arguments = parser.parse_args()

    out = workspace_path(arguments.out)
    os.makedirs(out, exist_ok=True)
    final_labels = (
        read_table('data/clips_audited.csv').drop_duplicates('name').set_index('name').label
    )
    runs = arguments.runs or sorted(
        os.path.basename(path)
        for path in glob.glob(f'{WS}/runs/*')
        if not os.path.basename(path).startswith(SKIPPED_PREFIXES)
    )

    rows, figures, skipped, by_run = [], [], [], {}
    for run in runs:
        predictions = final_predictions(run, final_labels)
        if predictions is None:
            skipped.append(run)
            continue
        by_run[run] = predictions
        result, counts = scores(predictions.label, predictions.predicted)
        rows.append({'run': run, **result})
        figures.append((run, counts, result))

    for name, members in AVERAGED.items():
        if not all(member in by_run for member in members):
            continue
        names = by_run[members[0]].index
        columns = [f'p_{label}' for label in CLASSES]
        probabilities = np.mean(
            [by_run[member].loc[names, columns].values for member in members], 0
        )
        predicted = np.array(CLASSES)[probabilities.argmax(1)]
        result, counts = scores(final_labels.loc[names].values, predicted)
        rows.append({'run': name, **result})
        figures.append((name, counts, result))

    summary = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    summary.to_csv(f'{out}/summary.csv', index=False, float_format='%.4f')
    order = {run: position for position, run in enumerate(summary.run)}
    with PdfPages(f'{out}/all.pdf') as pdf:
        for run, counts, result in sorted(figures, key=lambda item: order[item[0]]):
            title = (
                f'{run}\ntest folds, {result["clips"]} clips: accuracy {100 * result["accuracy"]:.1f} %, '
                f'macro-F1 {result["macro_f1"]:.3f}, {result["errors"]} errors'
            )
            figure = draw(counts, title)
            figure.savefig(f'{out}/{run}.png', dpi=120)
            pdf.savefig(figure)
            plt.close(figure)
            pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(f'{out}/{run}.csv')
    print(f'{len(figures)} confusion matrices -> {out}')
    print(f'skipped ({len(skipped)}): {" ".join(skipped)}')
    print(summary[['run', 'clips', 'accuracy', 'macro_f1', 'errors']].to_string(index=False))


if __name__ == '__main__':
    main()
