"""Metrics. Macro-F1 over the 7 classes is kept for continuity with earlier results, but it is not the headline:

- given a complete song, dialect accuracy is already ~97 %;
- almost all error is "is the terminal phrase there?" (complete vs Unfinished), worst for BhBl (AUC ~0.8);
- None vs Unfinished has no scientific value, yet moves macro-F1 a lot.
So `metrics()` also reports the decomposition both reviews asked for: completeness AUC per dialect, dialect
accuracy given complete, and the 3-way song state. Report bird-level accuracy
(evaluation/eval_recordings.py) as the headline for the scientists.
"""
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from config import CLASSES, C2I, DIALECTS, DIALECT_IDX, UNFINISHED_IDX


def metrics(y, p, src=None):
    pred = p.argmax(1); n = len(CLASSES)
    out = dict(acc=float((pred == y).mean()), macro_f1=float(f1_score(y, pred, average='macro', labels=range(n))),
               bal_acc=float(balanced_accuracy_score(y, pred)))
    is_dia = np.isin(y, DIALECT_IDX)
    out['dialect_f1_on_dialect_clips'] = float(f1_score(y[is_dia], pred[is_dia], average='macro', labels=DIALECT_IDX)) if is_dia.any() else None
    out['per_class_recall'] = {c: float((pred[y == i] == i).mean()) if (y == i).any() else None for i, c in enumerate(CLASSES)}
    out['per_class_f1'] = dict(zip(CLASSES, map(float, f1_score(y, pred, average=None, labels=range(n)))))
    out['cm'] = confusion_matrix(y, pred, labels=range(n)).tolist()
    # --- decomposition
    p_complete = p[:, DIALECT_IDX].sum(1); pred_dia = np.isin(pred, DIALECT_IDX); both = is_dia & pred_dia
    out['dialect_acc_given_complete'] = float((pred[both] == y[both]).mean()) if both.any() else None
    out['complete_recall'] = float(pred_dia[is_dia].mean()) if is_dia.any() else None
    out['complete_precision'] = float(is_dia[pred_dia].mean()) if pred_dia.any() else None
    auc = {}
    for c in DIALECTS:
        m = (y == C2I[c]) | (y == UNFINISHED_IDX)
        if (y[m] == C2I[c]).any() and (y[m] == UNFINISHED_IDX).any():
            auc[c] = float(roc_auc_score(y[m] == C2I[c], p_complete[m]))
    out['complete_vs_unfinished_auc'] = auc
    if src is not None:   # per-source recall exposes source shortcuts (e.g. BirdNET-app clips are never Unfinished/None)
        out['recall_by_source'] = {str(s): {c: float((pred[(src == s) & (y == i)] == i).mean()) for i, c in enumerate(CLASSES) if ((src == s) & (y == i)).any()}
                                   for s in np.unique(src)}
    return out
