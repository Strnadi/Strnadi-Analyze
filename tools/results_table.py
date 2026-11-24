"""One line per finished run: size, accuracy, macro-F1 and recall per class (out of fold).

    python -m tools.results_table [labelled table, default data/clips_audited.csv]

Only runs evaluated on the given labelled table are listed, so every line is comparable.
`--markdown` prints a Markdown table; `--write-doc` puts that table into docs/RESULTS.md, between
the two `runs table` marker lines, so the document can be refreshed after more runs finish.
"""

import glob
import json
import os
import sys

import numpy as np
import torch

from config import CLASSES, WS
from evaluation.report import load_run_predictions
from training.metrics import metrics


def run_size(run_dir):
    """Number of parameters of a run's model, read from one of its checkpoints."""
    checkpoints = sorted(glob.glob(f'{run_dir}/fold*/model.pt'))
    if not checkpoints:
        return None
    state = torch.load(checkpoints[0], map_location='cpu', weights_only=False)['state']
    return sum(value.numel() for name, value in state.items() if 'num_batches_tracked' not in name)


def describe_run(run_dir, labelled_table):
    """Metrics of one run, or None when it was evaluated on another table or has no folds."""
    settings_path = f'{run_dir}/settings.json'
    if not os.path.exists(settings_path):
        return None
    with open(settings_path) as file:
        if json.load(file)['labelled_table'] != labelled_table:
            return None
    try:
        predictions = load_run_predictions(os.path.basename(run_dir))
    except (FileNotFoundError, KeyError):  # no finished fold, or scored on an older table
        return None

    labels = predictions.label.map(CLASSES.index).values
    probabilities = predictions[[f'p_{name}' for name in CLASSES]].values
    result = metrics(labels, probabilities)
    per_fold = [
        metrics(
            labels[(predictions.cv_fold == fold).values],
            probabilities[(predictions.cv_fold == fold).values],
        )['acc']
        for fold in sorted(predictions.cv_fold.unique())
    ]
    return {
        'run': os.path.basename(run_dir),
        'folds': len(per_fold),
        'parameters': run_size(run_dir),
        'accuracy': result['acc'],
        'accuracy_fold_sd': float(np.std(per_fold)),
        'macro_f1': result['macro_f1'],
        'recall': result['per_class_recall'],
    }


def main():
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith('--')]
    labelled_table = arguments[0] if arguments else 'data/clips_audited.csv'
    write_doc = '--write-doc' in sys.argv
    markdown = '--markdown' in sys.argv or write_doc
    lines = []

    rows = [describe_run(run_dir, labelled_table) for run_dir in sorted(glob.glob(f'{WS}/runs/*'))]
    rows = sorted([row for row in rows if row], key=lambda row: -row['accuracy'])

    header = ['run', 'folds', 'params (M)', 'accuracy', 'fold sd', 'macro-F1'] + CLASSES
    separator = ' | ' if markdown else '  '
    if markdown:
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('|' + '---|' * len(header))
    else:
        lines.append(f'{header[0]:38s}' + separator.join(f'{name:>10s}' for name in header[1:]))
    for row in rows:
        cells = [
            f'{row["folds"]}',
            f'{row["parameters"] / 1e6:.2f}',
            f'{row["accuracy"]:.3f}',
            f'{row["accuracy_fold_sd"]:.3f}',
            f'{row["macro_f1"]:.3f}',
        ] + [f'{row["recall"][name]:.2f}' for name in CLASSES]
        if markdown:
            lines.append(f'| {row["run"]} | ' + ' | '.join(cells) + ' |')
        else:
            lines.append(f'{row["run"]:38s}' + separator.join(f'{cell:>10s}' for cell in cells))

    if not write_doc:
        print('\n'.join(lines))
        return

    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    document_path = os.path.join(repository, 'docs', 'RESULTS.md')
    with open(document_path) as file:
        document = file.read()
    begin, end = '<!-- runs table: begin -->', '<!-- runs table: end -->'
    before, rest = document.split(begin)
    after = rest.split(end)[1]
    with open(document_path, 'w') as file:
        file.write(before + begin + '\n' + '\n'.join(lines) + '\n' + end + after)
    print(f'{len(rows)} runs written to docs/RESULTS.md')


if __name__ == '__main__':
    main()
