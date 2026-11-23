"""Classification metrics.

Macro-F1 over the 7 classes is kept for continuity with earlier results, but it is not the headline:

- given a complete song, the dialect is already right ~97 % of the time;
- almost all error is "is the terminal phrase there?" (complete vs Unfinished), worst for BhBl;
- None vs Unfinished has no scientific value, yet it moves macro-F1 a lot.

So `metrics()` also reports the decomposition both reviews asked for: the complete-vs-Unfinished
AUC per dialect, dialect accuracy given a complete song, and recall per data source (which exposes
source shortcuts such as "BirdNET-app clips are never Unfinished"). For the scientists, report
bird-level accuracy (evaluation/eval_recordings.py) as the headline.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config import C2I, CLASSES, DIALECT_IDX, DIALECTS, UNFINISHED_IDX


def metrics(y, p, src=None):
    """Compute all metrics for integer labels `y` (N,) and class probabilities `p` (N, 7).

    Args:
        y: true class indices.
        p: predicted class probabilities.
        src: optional array with the data source of each clip, for per-source recall.
    """
    predicted = p.argmax(1)
    class_indices = range(len(CLASSES))
    result = {
        'acc': float((predicted == y).mean()),
        'macro_f1': float(f1_score(y, predicted, average='macro', labels=class_indices)),
        'bal_acc': float(balanced_accuracy_score(y, predicted)),
    }

    # ---- per class
    recall = {}
    for index, name in enumerate(CLASSES):
        of_class = y == index
        recall[name] = float((predicted[of_class] == index).mean()) if of_class.any() else None
    result['per_class_recall'] = recall
    f1_per_class = f1_score(y, predicted, average=None, labels=class_indices)
    result['per_class_f1'] = dict(zip(CLASSES, map(float, f1_per_class)))
    result['cm'] = confusion_matrix(y, predicted, labels=class_indices).tolist()

    # ---- dialect clips only
    is_dialect = np.isin(y, DIALECT_IDX)
    predicted_dialect = np.isin(predicted, DIALECT_IDX)
    if is_dialect.any():
        result['dialect_f1_on_dialect_clips'] = float(
            f1_score(y[is_dialect], predicted[is_dialect], average='macro', labels=DIALECT_IDX)
        )
        result['complete_recall'] = float(predicted_dialect[is_dialect].mean())
    else:
        result['dialect_f1_on_dialect_clips'] = None
        result['complete_recall'] = None

    # ---- decomposition: "is the song complete?" and "which dialect, given complete?"
    both_complete = is_dialect & predicted_dialect
    if both_complete.any():
        result['dialect_acc_given_complete'] = float(
            (predicted[both_complete] == y[both_complete]).mean()
        )
    else:
        result['dialect_acc_given_complete'] = None
    result['complete_precision'] = (
        float(is_dialect[predicted_dialect].mean()) if predicted_dialect.any() else None
    )

    probability_complete = p[:, DIALECT_IDX].sum(1)
    auc = {}
    for name in DIALECTS:
        # Clips of this dialect against the Unfinished clips.
        subset = (y == C2I[name]) | (y == UNFINISHED_IDX)
        is_this_dialect = y[subset] == C2I[name]
        if is_this_dialect.any() and (~is_this_dialect).any():
            auc[name] = float(roc_auc_score(is_this_dialect, probability_complete[subset]))
    result['complete_vs_unfinished_auc'] = auc

    # ---- per data source
    if src is not None:
        by_source = {}
        for source in np.unique(src):
            source_recall = {}
            for index, name in enumerate(CLASSES):
                selected = (src == source) & (y == index)
                if selected.any():
                    source_recall[name] = float((predicted[selected] == index).mean())
            by_source[str(source)] = source_recall
        result['recall_by_source'] = by_source

    return result
