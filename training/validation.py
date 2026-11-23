"""Validation: run a model over the rows of a table and return class probabilities."""

import numpy as np
import torch

from .data import ClipDataset


@torch.no_grad()
def predict_clips(model, table, settings, input_store=None, batch_size=64):
    """Class probabilities (N, 7) for every row of `table`, in table order.

    Each row is scored on one evaluation window (see `TrainSettings.eval_window`), without
    augmentation. `input_store` is passed when the model is fed cached embeddings instead of audio.
    """
    model.eval()
    dataset = ClipDataset(table, settings, train=False, input_store=input_store)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=min(2, settings.data_workers)
    )

    probabilities = []
    for batch in loader:
        logits = model(batch['wave'])['logits']
        probabilities.append(torch.softmax(logits, 1).numpy())
    return np.concatenate(probabilities)
