"""Batch inference helpers shared by evaluation, pseudo-labelling, export and the service."""
import numpy as np
import torch
from config import SR
from .checkpoint import load_model, run_checkpoints
from .data import load_seg


def peak_normalize(X, peak=0.5):
    return (X / np.maximum(np.abs(X).max(-1, keepdims=True), 1e-6) * peak).astype(np.float32)


def windows(df, offsets=(0.0,), win=4.0):
    """(len(df) * len(offsets), n) peak-normalised windows starting at t0 + offset."""
    n = int(win * SR)
    return peak_normalize(np.stack([load_seg(r.npy, r.t0 + o, r.t0 + o + win, n) for r in df.itertuples() for o in offsets]))


class TorchEnsemble:
    """Average class probabilities over the fold models of a run directory (or one checkpoint file)."""

    def __init__(self, spec, name='model.pt'):
        self.models = [load_model(p) for p in run_checkpoints(spec, name)]
        if not self.models:
            raise FileNotFoundError(f'no checkpoints under {spec}')

    @torch.no_grad()
    def __call__(self, X, bs=64, key='logits'):
        total = 0
        for m in self.models:
            outs = [m(torch.from_numpy(X[i:i + bs]))[key] for i in range(0, len(X), bs)]
            y = torch.cat(outs)
            total = total + (torch.softmax(y, 1) if key == 'logits' else y).numpy()
        return total / len(self.models)

    def embed(self, X, bs=64):
        return self(X, bs, key='emb')
