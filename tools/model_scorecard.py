"""One table to choose a model from: accuracy, robustness, size and speed of every finished run.

    python -m tools.model_scorecard [--write-doc]

For every 5-fold run scored on `data/clips_audited.csv`:

    accuracy, macro-F1       out of fold (evaluation/report.py)
    weakest class            the lowest recall among the seven classes
    noise 5 dB               accuracy with pink noise added at 5 dB (evaluation/stress_test.py)
    params, ms per window    size, and time of one 4 s window on one CPU thread
                             (evaluation/inference_cost.py); run this on an idle machine
    s per min                seconds of one CPU thread to scan one minute of audio as the service does

Marks: `=` accuracy within 0.005 of the best run (two seeds of one recipe differ by that much);
`*` on the efficient frontier: no other run is both at least as accurate and faster.

`--write-doc` puts the table into docs/RESULTS.md between the `scorecard` marker lines.
"""

import glob
import json
import os
import sys

import torch

from config import CLASSES, WS
from evaluation.inference_cost import WINDOWS_PER_MINUTE, cost_of
from evaluation.stress_test import stress_test
from tools.results_table import describe_run

NOISE_OF_ONE_SEED = 0.005


def scorecard_row(run_dir, device):
    """All numbers of one run, or None when it is unfinished or scored on another table."""
    row = describe_run(run_dir, 'data/clips_audited.csv')
    if row is None or row['folds'] != 5:
        return None
    run = row['run']
    try:
        row['noise_5db'] = stress_test(run, device)['noise 5 dB']
        cost = cost_of(run)
    except Exception as error:  # a model this tool cannot load or time
        print(f'{run}: skipped ({type(error).__name__}: {error})', file=sys.stderr)
        return None
    row['window_ms'] = cost['frontend_ms'] + cost['network_ms']
    row['weakest_class'] = min(row['recall'], key=row['recall'].get)
    row['weakest_recall'] = row['recall'][row['weakest_class']]
    return row


def mark_rows(rows):
    """Add the `=` (within noise of the best) and `*` (efficient frontier) marks."""
    best = max(row['accuracy'] for row in rows)
    for row in rows:
        within_noise = row['accuracy'] >= best - NOISE_OF_ONE_SEED
        dominated = any(
            other['accuracy'] >= row['accuracy']
            and other['window_ms'] < row['window_ms']
            and other is not row
            and (
                other['accuracy'] > row['accuracy'] or other['window_ms'] < row['window_ms'] * 0.95
            )
            for other in rows
        )
        row['marks'] = ('=' if within_noise else '') + ('' if dominated else '*')


def main():
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = [scorecard_row(run_dir, device) for run_dir in sorted(glob.glob(f'{WS}/runs/*'))]
    rows = sorted([row for row in rows if row], key=lambda row: row['window_ms'])
    mark_rows(rows)
    with open(f'{WS}/results/model_scorecard.json', 'w') as file:
        json.dump(rows, file, indent=1)

    header = (
        '| run | marks | params (M) | accuracy | macro-F1 | weakest class | noise 5 dB | '
        'ms per window | s per min of audio |'
    )
    lines = [header, '|' + '---|' * 9]
    for row in rows:
        lines.append(
            f'| {row["run"]} | {row["marks"]} | {row["parameters"] / 1e6:.2f} | {row["accuracy"]:.3f} '
            f'| {row["macro_f1"]:.3f} | {row["weakest_class"]} {row["weakest_recall"]:.2f} '
            f'| {row["noise_5db"]:.3f} | {row["window_ms"]:.1f} '
            f'| {row["window_ms"] * WINDOWS_PER_MINUTE / 1000:.1f} |'
        )
    print('\n'.join(lines))

    if '--write-doc' in sys.argv:
        repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repository, 'docs', 'RESULTS.md')
        with open(path) as file:
            document = file.read()
        begin, end = '<!-- scorecard: begin -->', '<!-- scorecard: end -->'
        before, rest = document.split(begin)
        with open(path, 'w') as file:
            file.write(before + begin + '\n' + '\n'.join(lines) + '\n' + end + rest.split(end)[1])


if __name__ == '__main__':
    assert len(CLASSES) == 7
    main()
