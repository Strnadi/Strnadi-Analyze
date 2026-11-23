"""Frozen Perch 2.0 pooled embedding (1536-d) -> 512 (GELU) -> classes. Ported from Keras.

`precomputed=True` (default): the input is the embedding itself, (B, 1536), as cached by
`processing/extract_foundation_emb.py`; nothing is downloaded. `precomputed=False` runs the torch port of
Perch inside the model on 5 s waveforms.

Measured (research session, grouped 5-fold): logistic 0.662, MLP 0.691 macro-F1. Pooled bird-ID embeddings
blur Yellowhammer dialects (Ghani et al. 2023) because pooling discards element order; `perch_temporal.py`
recovers +5 points. Origin: legacy `model/perch_based.py`, `model/_perch.py`, `experiments/perch.py`.
"""
import torch.nn as nn
from layers.mlp_head import MLPHead


class PerchMLP(nn.Module):
    wave_input = True

    def __init__(self, n_classes=7, precomputed=True, hidden=512, drop=0.3):
        super().__init__()
        self.frontend = None
        if not precomputed:
            from layers.perch_encoder import PerchEncoder
            self.encoder = PerchEncoder(trainable=False)
        else:
            self.encoder = None
        self.cls = MLPHead(1536, n_classes, hidden=(hidden,), drop=drop, act=nn.GELU, bn_in=False)

    def forward(self, x):
        e = x if self.encoder is None else self.encoder(x)['embedding']
        out = self.cls(e); out['emb'] = e
        return out
