"""StrnadNet with a concept-bottleneck head: the network regresses the expert measurements of the song ending
(f1, f2, f3, df, bw, t2, element presence) and the dialect is predicted from those alone.

Train with `aux='measurement'` (losses/measurement.py) on the rows that carry expert values
(Diblikova et al. 2023, Table S1: check the frequency columns before use; its timing column is a 6.000 s
placeholder in 384 of 430 rows) and the usual class loss on everything.
Status: proposed (research backlog "physics-informed multitask"; fork review idea 7). Untrained.
"""
from .strnadnet import StrnadNet


class MeasurementStrnadNet(StrnadNet):
    def __init__(self, n_classes=7, **kw):
        kw['head'] = 'measurement'
        super().__init__(n_classes=n_classes, **kw)
