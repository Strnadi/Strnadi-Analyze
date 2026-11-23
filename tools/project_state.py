"""Print the state of every run in the workspace: folds finished and the headline numbers.

    python -m tools.project_state

Read-only. Useful after moving to a new machine, or to see where a queue of jobs has got to.
"""

import glob
import json
import os

from config import WS
from evaluation.report import _labels_and_probabilities, load_run_predictions
from training.metrics import metrics


def describe_cv_run(run):
    predictions = load_run_predictions(run)
    labels, probabilities = _labels_and_probabilities(predictions)
    scores = metrics(labels, probabilities)
    recall = scores['per_class_recall']
    auc = scores['complete_vs_unfinished_auc']
    folds = ','.join(str(fold) for fold in sorted(predictions.cv_fold.unique()))
    print(
        f'{run:28s} folds {folds:9s} clips {len(predictions):5d}  '
        f'macro-F1 {scores["macro_f1"]:.3f}  acc {scores["acc"]:.3f}  '
        f'recall BhBl {recall["BhBl"]:.2f} BlBh {recall["BlBh"]:.2f} None {recall["None"]:.2f}  '
        f'dialect|complete {scores["dialect_acc_given_complete"]:.3f}  '
        f'AUC complete BhBl {auc.get("BhBl", float("nan")):.2f} BlBh {auc.get("BlBh", float("nan")):.2f}'
    )


def main():
    print(f'workspace: {WS}\n')
    print('cross-validation runs (out-of-fold predictions of the finished folds)')
    for run_dir in sorted(glob.glob(f'{WS}/runs/*')):
        run = os.path.basename(run_dir)
        if not glob.glob(f'{run_dir}/fold*/oof.npz'):
            print(f'{run:28s} no finished fold')
            continue
        try:
            describe_cv_run(run)
        except Exception as error:  # a run evaluated on a table this checkout cannot read
            print(f'{run:28s} could not be summarised: {error}')

    print(
        '\nlabel-free pretraining (linear probe on the frozen embedding: macro-F1, dialect macro-F1)'
    )
    for log_path in sorted(glob.glob(f'{WS}/ssl/*/log.json')):
        with open(log_path) as file:
            log = json.load(file)
        name = os.path.basename(os.path.dirname(log_path))
        finished = os.path.exists(os.path.join(os.path.dirname(log_path), 'done'))
        last = log[-1] if log else {}
        print(
            f'{name:28s} step {last.get("step", "?")}  probe {last.get("probe")}  '
            f'{"finished" if finished else "running / interrupted"}'
        )


if __name__ == '__main__':
    main()
