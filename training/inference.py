"""Batch inference helpers shared by evaluation, pseudo-labelling, export and the service."""

import numpy as np
import torch

from config import SR

from .checkpoint import load_model, run_checkpoints
from .data import load_seg


def peak_normalize(batch, peak=0.5):
    """Scale every waveform of a (N, samples) array to the same peak."""
    peaks = np.abs(batch).max(-1, keepdims=True)
    return (batch / np.maximum(peaks, 1e-6) * peak).astype(np.float32)


def windows(table, offsets=(0.0,), win=4.0):
    """Cut peak-normalised windows for every table row and every start offset.

    Returns:
        array of shape (len(table) * len(offsets), samples), rows in table order,
        offsets varying fastest.
    """
    n_samples = int(win * SR)
    cut = []
    for row in table.itertuples():
        for offset in offsets:
            start = row.t0 + offset
            cut.append(load_seg(row.npy, start, start + win, n_samples))
    return peak_normalize(np.stack(cut))


class TorchEnsemble:
    """The fold models of a run directory (or one checkpoint), averaged.

    Calling it on a (N, samples) array returns averaged class probabilities (N, 7).
    """

    def __init__(self, run_or_file, name='model.pt', device='cpu'):
        self.device = torch.device(device)
        self.models = [
            load_model(path).to(self.device) for path in run_checkpoints(run_or_file, name)
        ]
        if not self.models:
            raise FileNotFoundError(f'no checkpoints under {run_or_file}')

    @torch.no_grad()
    def __call__(self, batch, bs=64, key='logits'):
        total = 0
        for model in self.models:
            outputs = []
            for start in range(0, len(batch), bs):
                chunk = torch.from_numpy(batch[start : start + bs]).to(self.device)
                outputs.append(model(chunk)[key])
            outputs = torch.cat(outputs)
            if key == 'logits':
                outputs = torch.softmax(outputs, 1)
            total = total + outputs.float().cpu().numpy()
        return total / len(self.models)

    def embed(self, batch, bs=64):
        """Averaged clip embeddings (N, D)."""
        return self(batch, bs, key='emb')
