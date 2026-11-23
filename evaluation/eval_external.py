"""Evaluate a model on an external song table (e.g. Vizina held-out individuals).

model spec:  tflite:/path/model.tflite   (input 4 s @ 48 kHz, 7 outputs, production format)
             torch:/path/run_dir          (averages fold*/model.pt ensemble) or torch:/path/model.pt
protocols:   aligned = one 4 s window starting 0.4 s before annotated song onset
             slide   = 5 windows (onset-1.0 .. onset+0.0 s), class score = max over windows
"""

import sys, os, glob, json, numpy as np, pandas as pd, torch
from scipy.signal import resample_poly
from config import CLASSES, C2I, WS
from training.data import load_seg
from training.checkpoint import (
    load_logits_model as load_torch_model,
)  # tensor-returning view, as these scripts expect
from training.inference import TorchEnsemble

OFFS = [-1.0, -0.75, -0.5, -0.25, 0.0]


def windows(df, offs):
    X = []
    for r in df.itertuples():
        for o in offs:
            X.append(load_seg(r.npy, r.t0 + o, r.t0 + o + 4, 128000))
    X = np.stack(X)
    pk = np.abs(X).max(1, keepdims=True)
    X = X / np.maximum(pk, 1e-6) * 0.5
    return X.astype(np.float32)


class TFLiteModel:
    def __init__(self, path, threads=8):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=path, num_threads=threads)
        self.B = 16
        self.it.resize_tensor_input(0, [self.B, 192000])
        self.it.allocate_tensors()

    def __call__(self, X):
        X48 = resample_poly(X, 3, 2, axis=1).astype(np.float32)
        out = []
        for i in range(0, len(X48), self.B):
            xb = X48[i : i + self.B]
            n = len(xb)
            if n < self.B:
                xb = np.concatenate([xb, np.zeros((self.B - n, 192000), np.float32)])
            self.it.set_tensor(self.it.get_input_details()[0]['index'], xb)
            self.it.invoke()
            out.append(self.it.get_tensor(self.it.get_output_details()[0]['index'])[:n])
        return np.concatenate(out)


def report(y, P, name):
    pred = P.argmax(1)
    out = {'n': int(len(y))}
    for c in sorted(set(y)):
        m = y == c
        out[CLASSES[c]] = {
            'recall': float((pred[m] == c).mean()),
            'pred_dist': {
                CLASSES[k]: int((pred[m] == k).sum())
                for k in range(len(CLASSES))
                if (pred[m] == k).sum()
            },
        }
    out['acc'] = float((pred == y).mean())
    print(name, json.dumps(out))
    return out


if __name__ == '__main__':
    spec, table, split = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'test')
    torch.set_num_threads(int(os.environ.get('THREADS', 8)))
    df = pd.read_csv(table, dtype={'indiv': str})
    if split != 'all':
        df = df[df.split == split]
    y = df.label.map(C2I).values
    model = (
        TFLiteModel(spec.split(':', 1)[1])
        if spec.startswith('tflite:')
        else TorchEnsemble(spec.split(':', 1)[1])
    )
    X = windows(df, OFFS)
    P = model(X).reshape(len(df), len(OFFS), -1)
    res = {
        'aligned': report(y, P[:, 2], 'aligned'),
    }
    # slide: per class max over windows; None only if no YH class wins anywhere
    Pm = P.max(1)
    res['slide'] = report(y, Pm, 'slide-max')
    tag = os.path.basename(spec.rstrip('/')).replace('.tflite', '')
    os.makedirs(f'{WS}/results', exist_ok=True)
    json.dump(
        res,
        open(
            f'{WS}/results/ext_{os.path.basename(table).replace(".csv", "")}_{split}_{tag}.json',
            'w',
        ),
        indent=1,
    )
