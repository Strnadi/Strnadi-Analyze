"""Factorised output: p(song) * p(complete | song) * q(dialect | complete).

Composed into the 7 classes ['BC','BE','BhBl','BlBh','None','Unfinished','XB'] and returned as
normalised log-probabilities (so softmax / cross-entropy on them is a no-op / plain NLL). The three
factors are returned too: recording-level (weak) labels train only q, weighted by the detached
p(complete), so they cannot drag songs into "Unfinished"; bag losses and completeness-specific
supervision attach to `log_complete`.

This is the only recipe change with a significant measured gain so far (R2: macro-F1 0.774 vs
0.745).
Origin: research session (independent reviewer's idea N2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# positions in the 7-class order BC, BE, BhBl, BlBh, None, Unfinished, XB
DIALECT_IDX = [0, 1, 2, 3, 6]
NONE_IDX = 4
UNFINISHED_IDX = 5


class FactorizedHead(nn.Module):
    def __init__(self, d, n_classes=7):
        super().__init__()
        assert n_classes == 7, 'factorised head is defined for the 7 Strnadi classes'
        self.lin = nn.Linear(d, 7)
        self.in_features = d

    def forward(self, e):
        """Embedding e (B, D) -> dict: `logits` (B, 7) log-probabilities, `log_song` (B,),
        `log_complete` (B,), `log_dialect` (B, 5)."""
        z = self.lin(e)  # (B, 7): song logit, complete logit, 5 dialect logits
        song_logit = z[:, 0]
        complete_logit = z[:, 1]
        dialect_logits = z[:, 2:7]

        log_song = F.logsigmoid(song_logit)  # log p(song)
        log_no_song = F.logsigmoid(-song_logit)  # log (1 - p(song))
        log_complete = F.logsigmoid(complete_logit)  # log p(complete | song)
        log_incomplete = F.logsigmoid(-complete_logit)  # log (1 - p(complete | song))
        log_dialect = F.log_softmax(dialect_logits, 1)  # (B, 5) log q(dialect | complete)

        # log of p(song) * p(complete | song) * q(dialect | complete), in the order of DIALECT_IDX
        log_song_and_complete = (log_song + log_complete).unsqueeze(1)  # (B, 1)
        dialect_log_probs = log_song_and_complete + log_dialect  # (B, 5)
        none_log_prob = log_no_song.unsqueeze(1)  # (B, 1)
        unfinished_log_prob = (log_song + log_incomplete).unsqueeze(1)  # (B, 1)

        # class order BC, BE, BhBl, BlBh, None, Unfinished, XB. Built by concatenation: the research
        # version filled a pre-allocated tensor in place, which froze the traced batch size into the
        # exported ONNX graph.
        out = torch.cat(
            [
                dialect_log_probs[:, :4],  # BC, BE, BhBl, BlBh
                none_log_prob,
                unfinished_log_prob,
                dialect_log_probs[:, 4:],  # XB
            ],
            1,
        )
        return {
            'logits': out,
            'log_song': log_song,
            'log_complete': log_complete,
            'log_dialect': log_dialect,
        }
