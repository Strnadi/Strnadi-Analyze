"""Concept-bottleneck head (Koh et al. 2020): predict the quantities the ornithologists measure,
then classify from those alone.

Measurements (Diblikova et al. 2023): f1 = end frequency of B, f2 / f3 = start / end frequency of
the terminal element, df = f2 - f1, bw = f3 - f2, t2 = terminal-element duration, plus two presence
logits (first terminal element present, second present). The class logits are a small MLP on the
measurement vector only, so every decision can be read back as "B ends at x Hz, the last element
rises by y Hz".
Train the measurements with `losses/measurement.py` on the songs that have expert values; the class
loss trains everything end to end.
Origin: research backlog idea "physics-informed multitask" (never built) + fork review
(architecture idea 7).
"""

import torch
import torch.nn as nn

MEASUREMENTS = ['f1', 'f2', 'f3', 'df', 'bw', 't2', 'has_first', 'has_second']


class MeasurementHead(nn.Module):
    def __init__(self, d, n_classes, hidden=32):
        super().__init__()
        n_measurements = len(MEASUREMENTS)
        self.measure = nn.Linear(d, n_measurements)
        self.classify = nn.Sequential(
            nn.Linear(n_measurements, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, e):
        """Embedding e (B, D) -> dict: `logits` (B, n_classes), `measurements` (B, 8) in the order
        of MEASUREMENTS (the two presence entries as logits)."""
        measurements = self.measure(e)  # (B, 8)
        continuous = measurements[:, :6]  # f1, f2, f3, df, bw, t2
        presence = torch.sigmoid(measurements[:, 6:])  # has_first, has_second as probabilities
        concepts = torch.cat([continuous, presence], 1)  # (B, 8), the classifier's only input
        return {'logits': self.classify(concepts), 'measurements': measurements}
