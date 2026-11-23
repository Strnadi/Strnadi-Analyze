"""Perch 2.0 spatial embedding kept as a time series -> shared sequence head.

Three uses, in increasing cost:
1. `precomputed=True`: input is the cached spatial embedding (B, T', 1536) (frequency already averaged).
   Frozen Perch + this head scores 0.741 macro-F1, i.e. the from-scratch StrnadNet baseline (0.745), and a
   fold trains in minutes. This is the fast track for searching weak-label recipes (factorised head, bag loss,
   pool shares) before spending fold-hours on them (fork review, step 6).
2. `precomputed=False, trainable=False`: the torch port of Perch runs inside the model (frozen).
3. `trainable=True`: Perch fine-tuned end to end = the strong teacher both reviews ask for. Use
   `backbone_lr_mult` ~0.1 and a small base lr (1e-4 .. 3e-4).
Origin: research session (frozen probe, `eval_frozen_emb.py`) + both reviews (fine-tuning, never started there).
"""
import torch.nn as nn
from layers.sequence_head import SequenceHead


class PerchTemporal(nn.Module):
    wave_input = True
    backbone_prefixes = ('encoder.',)

    def __init__(self, n_classes=7, precomputed=True, trainable=False, d=192, temporal='conv', n_temporal=2, drop=0.2,
                 head='linear', head_kw=None):
        super().__init__()
        self.frontend = None
        if precomputed:
            self.encoder = None
        else:
            from layers.perch_encoder import PerchEncoder
            self.encoder = PerchEncoder(trainable=trainable)
        self.head = SequenceHead(1536, n_classes, d, temporal, n_temporal, drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        if self.encoder is None:
            series = x.transpose(1, 2)                                # (B, T', 1536) -> (B, 1536, T')
        else:
            series = self.encoder(x)['spatial'].mean(-1)              # (B, 1536, T', F') -> average frequency
        return self.head(series)
