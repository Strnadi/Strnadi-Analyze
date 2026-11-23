"""Factorised output: p(song) * p(complete | song) * q(dialect | complete).

Composed into the 7 classes ['BC','BE','BhBl','BlBh','None','Unfinished','XB'] and returned as normalised
log-probabilities (so softmax / cross-entropy on them is a no-op / plain NLL). The three factors are returned
too: recording-level (weak) labels train only q, weighted by the detached p(complete), so they cannot drag songs
into "Unfinished"; bag losses and completeness-specific supervision attach to `log_complete`.

This is the only recipe change with a significant measured gain so far (R2: macro-F1 0.774 vs 0.745).
Origin: research session (independent reviewer's idea N2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DIALECT_IDX = [0, 1, 2, 3, 6]
NONE_IDX, UNFINISHED_IDX = 4, 5


class FactorizedHead(nn.Module):
    def __init__(self, d, n_classes=7):
        super().__init__()
        assert n_classes == 7, 'factorised head is defined for the 7 Strnadi classes'
        self.lin = nn.Linear(d, 7); self.in_features = d

    def forward(self, e):
        z = self.lin(e)
        ls, lns = F.logsigmoid(z[:, 0]), F.logsigmoid(-z[:, 0])       # song / no song
        lc, lnc = F.logsigmoid(z[:, 1]), F.logsigmoid(-z[:, 1])       # complete / unfinished
        lq = F.log_softmax(z[:, 2:7], 1)                              # dialect given complete
        dia = (ls + lc).unsqueeze(1) + lq                             # (B, 5) in the order of DIALECT_IDX
        # class order BC, BE, BhBl, BlBh, None, Unfinished, XB. Built by concatenation: the research version filled a
        # pre-allocated tensor in place, which froze the traced batch size into the exported ONNX graph.
        out = torch.cat([dia[:, :4], lns.unsqueeze(1), (ls + lnc).unsqueeze(1), dia[:, 4:]], 1)
        return {'logits': out, 'log_song': ls, 'log_complete': lc, 'log_dialect': lq}
