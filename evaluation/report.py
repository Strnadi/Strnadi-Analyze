"""Evaluation of a finished run: the functions every experiment script ends with.

    predictions = load_run_predictions(RUN)   # one row per held-out clip, with probabilities
    summarize_run(RUN)                        # metrics table, printed and saved as JSON
    plot_run(RUN)                             # learning curves + confusion matrix
    plot_error_examples(RUN)                  # spectrograms of the most confident errors

All of them work on the out-of-fold predictions (`oof.npz`) that `training.fit.cross_validate`
writes: every labelled clip is scored by the fold model that never saw its recording.
Figures are saved under `$STRNADI_WS/figs/` and returned, so a notebook displays them.

`validate` (= `training.validation.predict_clips`) scores any model on any table.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from config import CLASSES, WS, dry_run, effective_run_name
from training.data import load_seg, read_table
from training.metrics import metrics
from training.validation import predict_clips as validate  # noqa: F401  (re-exported)


# ------------------------------------------------------------------------------- predictions
def load_run_predictions(run):
    """Collect the out-of-fold predictions of a run.

    Returns:
        DataFrame with the labelled-table columns of every scored clip plus
        `p_<class>` (probabilities), `predicted`, `confidence`, `correct` and `cv_fold`.
    """
    run_dir = f'{WS}/runs/{effective_run_name(run)}'
    fold_files = sorted(glob.glob(f'{run_dir}/fold*/oof.npz'))
    if not fold_files:
        raise FileNotFoundError(f'no finished folds under {run_dir}')

    parts = []
    table = None
    for path in fold_files:
        fold_result = np.load(path, allow_pickle=True)
        if table is None:
            table = read_table(_labelled_table_of(run_dir, fold_result))

        clips = table.loc[fold_result['idx']].copy()
        probabilities = fold_result['p']
        for index, name in enumerate(CLASSES):
            clips[f'p_{name}'] = probabilities[:, index]
        clips['predicted'] = [CLASSES[i] for i in probabilities.argmax(1)]
        clips['confidence'] = probabilities.max(1)
        clips['cv_fold'] = int(os.path.basename(os.path.dirname(path)).replace('fold', ''))
        parts.append(clips)

    predictions = pd.concat(parts)
    predictions['correct'] = predictions.predicted == predictions.label
    return predictions


def _labelled_table_of(run_dir, fold_result):
    """The clip table a run was evaluated on (research-session runs stored it in args.json)."""
    if 'strong' in fold_result:
        return str(fold_result['strong'])
    research_args = f'{run_dir}/args.json'
    if os.path.exists(research_args):
        with open(research_args) as file:
            return json.load(file).get('strong', 'data/orig_index.csv')
    return 'data/orig_index.csv'


def _labels_and_probabilities(predictions):
    labels = predictions.label.map(CLASSES.index).values
    probabilities = predictions[[f'p_{name}' for name in CLASSES]].values
    return labels, probabilities


# ----------------------------------------------------------------------------------- summary
def summarize_run(run, save=True):
    """Print the metrics of a run and return them as a dict.

    Reported: accuracy and macro-F1, recall per class, the decomposition into "is the song
    complete?" and "which dialect, given complete?", recall per data source, and macro-F1 per fold.
    """
    if dry_run():
        return None

    predictions = load_run_predictions(run)
    labels, probabilities = _labels_and_probabilities(predictions)
    sources = predictions.src.values if 'src' in predictions else None
    result = metrics(labels, probabilities, sources)

    per_fold = {}
    for fold, fold_predictions in predictions.groupby('cv_fold'):
        fold_labels, fold_probabilities = _labels_and_probabilities(fold_predictions)
        per_fold[int(fold)] = metrics(fold_labels, fold_probabilities)['macro_f1']
    result['macro_f1_per_fold'] = per_fold
    result['n_clips'] = len(predictions)

    name = effective_run_name(run)
    print(f'\n=== {name}: {len(predictions)} held-out clips, {len(per_fold)} folds')
    print(
        f'accuracy {result["acc"]:.3f}   macro-F1 {result["macro_f1"]:.3f}   '
        f'balanced accuracy {result["bal_acc"]:.3f}'
    )
    print('macro-F1 per fold: ' + '  '.join(f'{k}: {v:.3f}' for k, v in per_fold.items()))

    print('\nrecall per class')
    for class_name, recall in result['per_class_recall'].items():
        n_clips = int((predictions.label == class_name).sum())
        if recall is not None:
            print(f'  {class_name:11s} {recall:.2f}   ({n_clips} clips)')

    print('\nwhere the errors are')
    print(
        f'  dialect accuracy, given label and prediction are both a complete song: '
        f'{_fmt(result["dialect_acc_given_complete"])}'
    )
    print(f'  complete songs recognised as complete (recall): {_fmt(result["complete_recall"])}')
    print(
        f'  predicted-complete songs that are complete (precision): '
        f'{_fmt(result["complete_precision"])}'
    )
    print(
        '  complete-vs-Unfinished AUC per dialect: '
        + '  '.join(f'{k} {v:.2f}' for k, v in result['complete_vs_unfinished_auc'].items())
    )

    if 'recall_by_source' in result:
        print('\nrecall per data source (large gaps between sources suggest a source shortcut)')
        for source, recalls in result['recall_by_source'].items():
            print(f'  {source:15s} ' + '  '.join(f'{k} {v:.2f}' for k, v in recalls.items()))

    if save:
        os.makedirs(f'{WS}/results', exist_ok=True)
        with open(f'{WS}/results/cv_{name}.json', 'w') as file:
            json.dump(result, file, indent=1)
    return result


def _fmt(value):
    return 'n/a' if value is None else f'{value:.3f}'


# ----------------------------------------------------------------- label-free models: probing
def probe_checkpoint(path, held_out_fold=0):
    """How much dialect information a (label-free) model's embedding carries.

    Embeds every labelled clip with the frozen model, fits logistic regression on the clips outside
    `held_out_fold` and scores it on that fold. Prints and returns macro-F1 and dialect macro-F1.
    """
    if dry_run():
        return None
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from config import C2I, workspace_path
    from training.checkpoint import load_model

    model = load_model(workspace_path(path))
    labelled = read_table('data/orig_index.csv')
    if 'aug' in labelled:
        labelled = labelled[~labelled['aug']]

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(labelled), 64):
            chunk = labelled.iloc[start : start + 64]
            waves = np.stack([load_seg(r.npy, r.t0, r.t0 + 4, 128000) for r in chunk.itertuples()])
            waves = waves / np.maximum(np.abs(waves).max(1, keepdims=True), 1e-6) * 0.5
            embeddings.append(model(torch.from_numpy(waves.astype(np.float32)))['emb'].numpy())
    embeddings = np.concatenate(embeddings)
    labels = labelled.label.map(C2I).values

    train = (labelled.fold != held_out_fold).values
    test = ~train
    scaler = StandardScaler().fit(embeddings[train])
    probe = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
    probe.fit(scaler.transform(embeddings[train]), labels[train])
    scores = metrics(labels[test], probe.predict_proba(scaler.transform(embeddings[test])))

    result = {
        'macro_f1': scores['macro_f1'],
        'dialect_f1': scores['dialect_f1_on_dialect_clips'],
        'complete_vs_unfinished_auc': scores['complete_vs_unfinished_auc'],
    }
    print(f'linear probe on the frozen embedding of {path} (fold {held_out_fold} held out):')
    print(f'  macro-F1 {result["macro_f1"]:.3f}   dialect macro-F1 {result["dialect_f1"]:.3f}')
    return result


# -------------------------------------------------------------------------------------- plots
def plot_confusion_matrix(run, ax=None):
    """Confusion matrix of the out-of-fold predictions: counts, coloured by the share of each row."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    predictions = load_run_predictions(run)
    counts = confusion_matrix(predictions.label, predictions.predicted, labels=CLASSES)
    row_share = counts / counts.sum(1, keepdims=True).clip(1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(row_share, cmap='Blues', vmin=0, vmax=1)
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            text_colour = 'white' if row_share[row, column] > 0.5 else 'black'
            ax.text(
                column,
                row,
                counts[row, column],
                ha='center',
                va='center',
                fontsize=8,
                color=text_colour,
            )
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('predicted')
    ax.set_ylabel('label')
    ax.set_title('held-out clips: counts (colour = share of the row)')
    return ax


def plot_learning_curves(run, axes=None):
    """Training loss and macro-F1 per epoch, one line per fold.

    Solid lines are the inner validation set (which decides the kept checkpoint); dotted lines are
    the test fold, shown for diagnosis only.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis, score_axis = axes

    log_files = sorted(glob.glob(f'{WS}/runs/{effective_run_name(run)}/fold*/log.json'))
    for path in log_files:
        with open(path) as file:
            log = json.load(file)
        fold = os.path.basename(os.path.dirname(path))
        epochs = [entry['ep'] for entry in log]

        loss_axis.plot(epochs, [entry['loss'] for entry in log], label=fold)
        if 'inner_macro_f1' in log[0]:
            inner = [entry.get('inner_macro_f1', np.nan) for entry in log]
            score_axis.plot(epochs, inner, '-', label=f'{fold} inner validation')
        if 'val_macro_f1' in log[0]:
            outer = [entry.get('val_macro_f1', np.nan) for entry in log]
            score_axis.plot(epochs, outer, ':', label=f'{fold} test fold')

    loss_axis.set_title('training loss')
    loss_axis.set_xlabel('epoch')
    loss_axis.legend(fontsize=7)
    score_axis.set_title('macro-F1 (solid: inner validation, dotted: test fold)')
    score_axis.set_xlabel('epoch')
    score_axis.legend(fontsize=6, ncol=2)
    return axes


def plot_run(run):
    """Learning curves and confusion matrix in one figure, saved to figs/run_<run>.png."""
    if dry_run():
        return None
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    plot_learning_curves(run, axes[:2])
    plot_confusion_matrix(run, axes[2])
    figure.tight_layout()
    return _save(figure, f'run_{effective_run_name(run)}.png')


def plot_error_examples(run, n=24, columns=6):
    """Spectrograms of the most confident wrong predictions, as the model sees them.

    Looking at these is not optional: metrics have repeatedly hidden problems here (clips starting
    mid-song, labels that describe the bird and not the window, synthetic artefacts). Saved to
    figs/errors_<run>.png.
    """
    if dry_run():
        return None
    import matplotlib.pyplot as plt
    import torch

    from layers.registry import build_frontend

    predictions = load_run_predictions(run)
    errors = predictions[~predictions.correct].sort_values('confidence', ascending=False).head(n)
    if errors.empty:
        print('no errors to show')
        return None

    frontend = build_frontend()
    rows = int(np.ceil(len(errors) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.2 * rows))
    axes = np.atleast_2d(axes)

    for position, clip in enumerate(errors.itertuples()):
        waveform = load_seg(clip.npy, clip.t0, clip.t0 + 4, 128000)
        waveform = waveform / max(np.abs(waveform).max(), 1e-6) * 0.5
        with torch.no_grad():
            spectrogram = frontend(torch.from_numpy(waveform)[None])[0, 0].numpy()

        ax = axes[position // columns, position % columns]
        ax.imshow(spectrogram, origin='lower', aspect='auto', vmin=0, vmax=2.5, cmap='magma')
        ax.set_title(f'{clip.label} -> {clip.predicted} ({clip.confidence:.2f})', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for position in range(len(errors), rows * columns):
        axes[position // columns, position % columns].axis('off')

    figure.suptitle('most confident errors: label -> prediction (confidence)', fontsize=10)
    figure.tight_layout()
    return _save(figure, f'errors_{effective_run_name(run)}.png')


def report_run(run):
    """Everything at once: the summary, the curves + confusion matrix, and the error sheet."""
    result = summarize_run(run)
    plot_run(run)
    plot_error_examples(run)
    return result


def _save(figure, file_name):
    os.makedirs(f'{WS}/figs', exist_ok=True)
    path = f'{WS}/figs/{file_name}'
    figure.savefig(path, dpi=110)
    print('saved', path)
    return figure


if __name__ == '__main__':
    import sys

    report_run(sys.argv[1])
