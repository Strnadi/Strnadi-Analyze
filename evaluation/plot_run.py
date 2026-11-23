"""Learning curves and confusion matrix of a run (replaces the legacy `history-plot.py` and
`confusion-matrix.py`, which read Keras history objects and a randomly split test set).

Curves: training loss, inner-validation macro-F1 (what checkpoints are chosen on) and outer-fold macro-F1
(diagnostic only) per epoch and fold. Confusion matrix: all out-of-fold clips, row-normalised, with counts.

Usage: python -m evaluation.plot_run <run>        -> figs/run_<run>.png
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import WS, CLASSES


def main():
    run = sys.argv[1]; fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for f in sorted(glob.glob(f'{WS}/runs/{run}/fold*/log.json')):
        L = json.load(open(f)); k = os.path.basename(os.path.dirname(f)); ep = [r['ep'] for r in L]
        ax[0].plot(ep, [r['loss'] for r in L], label=k)
        for key, style in (('inner_macro_f1', '-'), ('val_macro_f1', ':')):
            if key in L[0]:
                ax[1].plot(ep, [r.get(key, np.nan) for r in L], style, label=f'{k} {"inner" if style == "-" else "outer"}')
    ax[0].set_title('training loss'); ax[0].set_xlabel('epoch'); ax[0].legend(fontsize=7)
    ax[1].set_title('macro-F1: inner validation (solid), outer fold (dotted, diagnostic)'); ax[1].set_xlabel('epoch'); ax[1].legend(fontsize=6, ncol=2)
    Y, P = [], []
    for f in glob.glob(f'{WS}/runs/{run}/fold*/oof.npz'):
        z = np.load(f, allow_pickle=True); Y.append(z['y']); P.append(z['p'].argmax(1))
    if Y:
        cm = confusion_matrix(np.concatenate(Y), np.concatenate(P), labels=range(len(CLASSES))); rn = cm / cm.sum(1, keepdims=True).clip(1)
        ax[2].imshow(rn, cmap='Blues', vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax[2].text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if rn[i, j] > 0.5 else 'black')
        ax[2].set_xticks(range(len(CLASSES)), CLASSES, rotation=45); ax[2].set_yticks(range(len(CLASSES)), CLASSES)
        ax[2].set_xlabel('predicted'); ax[2].set_ylabel('label'); ax[2].set_title('out-of-fold confusion (counts; colour = row share)')
    os.makedirs(f'{WS}/figs', exist_ok=True); fig.tight_layout(); fig.savefig(f'{WS}/figs/run_{run}.png', dpi=110)
    print(f'{WS}/figs/run_{run}.png')


if __name__ == '__main__':
    main()
