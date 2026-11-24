"""Validation: run a model over the rows of a table and return class probabilities."""

import numpy as np
import torch

from .data import ClipDataset


def load_eval_waveforms(table, settings, input_store=None):
    """Read the evaluation window of every row of `table` once: tensor (N, samples), table order.

    Each row gives one window (see `TrainSettings.eval_window`), peak-normalised, never augmented.
    With `input_store` the rows are cached embeddings instead of audio.
    """
    if len(table) == 0:
        return torch.zeros(0, settings.window_samples)
    dataset = ClipDataset(table, settings, train=False, input_store=input_store)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=min(8, settings.data_workers)
    )
    return torch.cat([batch['wave'] for batch in loader])


@torch.no_grad()
def predict_waveforms(model, waveforms, device='cpu', mixed_precision=False, batch_size=128):
    """Class probabilities (N, 7) for a tensor of evaluation windows."""
    model.eval()
    probabilities = []
    for start in range(0, len(waveforms), batch_size):
        batch = waveforms[start : start + batch_size].to(device)
        takes_waveform = getattr(model, 'wave_input', False) or getattr(model, 'frontend', None) is None
        if not takes_waveform and batch.dim() == 2:
            # spectrogram in full precision, network in mixed precision (as in training)
            spectrogram = model.frontend(batch)
            with torch.autocast('cuda', enabled=mixed_precision):
                logits = model(spectrogram)['logits']
        else:
            logits = model(batch)['logits']
        probabilities.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probabilities)


def predict_clips(model, table, settings, input_store=None, device='cpu'):
    """Class probabilities (N, 7) for every row of `table`, in table order."""
    waveforms = load_eval_waveforms(table, settings, input_store)
    return predict_waveforms(model.to(device), waveforms, device)


def mean_nll(labels, probabilities):
    """Mean negative log-likelihood of the true classes: the validation loss."""
    own = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(own, 1e-7, 1.0)).mean())
