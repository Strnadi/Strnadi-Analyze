"""Frozen Perch 2.0 pooled embedding (1536-d) -> 512 (GELU) -> classes. Ported from Keras.

`precomputed=True` (default): the input is the embedding itself, (B, 1536), as cached by
`processing/extract_foundation_emb.py`; nothing is downloaded. `precomputed=False` runs the torch
port of Perch inside the model on 5 s waveforms.

Measured (research session, grouped 5-fold): logistic 0.662, MLP 0.691 macro-F1. Pooled bird-ID
embeddings blur Yellowhammer dialects (Ghani et al. 2023) because pooling discards element order;
`perch_temporal.py` recovers +5 points.

Origin: legacy `model/perch_based.py`, `model/_perch.py`, `experiments/perch.py`.
"""

import torch.nn as nn

from layers.mlp_head import MLPHead
from model.base import Model


class PerchMLP(Model):
    """Perch clip embedding -> one hidden layer -> classes.

    Args:
        n_classes: number of output classes.
        precomputed: True (default) = the input is the cached 1536-d Perch embedding and no Perch
            network is built; False = the input is a waveform and the frozen torch port of Perch
            runs inside the model (downloaded on first use).
        hidden: width of the hidden layer.
        drop: dropout after the hidden layer.

    `forward` returns a dict: `logits` (B, n_classes) and `emb` (B, 1536), the Perch embedding.
    """

    wave_input = True  # the trainer passes the input as is (no spectrogram augmentation)

    def __init__(self, n_classes=7, precomputed=True, hidden=512, drop=0.3):
        super().__init__()

        # --- no spectrogram front-end of our own: Perch brings its own (or the input is cached)
        self.frontend = None

        # --- encoder: frozen Perch 2.0, only when the embeddings are not precomputed
        if not precomputed:
            from layers.perch_encoder import PerchEncoder  # needs a network download

            self.encoder = PerchEncoder(trainable=False)
        else:
            self.encoder = None

        # --- classifier head: 1536 -> hidden (GELU, dropout) -> classes, no input BatchNorm
        self.cls = MLPHead(1536, n_classes, hidden=(hidden,), drop=drop, act=nn.GELU, bn_in=False)

    def forward(self, x):
        if self.encoder is None:
            embedding = x  # (B, 1536)   precomputed Perch embedding
        else:
            embedding = self.encoder(x)['embedding']  # waveform (B, N) -> (B, 1536)
        outputs = self.cls(embedding)  # dict with `logits` (B, n_classes)
        outputs['emb'] = embedding
        return outputs


# %% Quick look: build the model and show what it returns for a batch of cached Perch embeddings
if __name__ == '__main__':
    import torch

    from model.base import describe

    describe(PerchMLP(), example=torch.randn(2, 1536))
