"""1-D temporal conv net over per-frame pitch-contour features instead of the full spectrogram
(0.27 M params, ~0.1 GMAC per 4 s: about 5x cheaper than StrnadNet, and interpretable).

features='topk'  : research version (top-3 mel peaks, quantised to one bin);
features='ridge' : sub-bin ridge track (fork review idea 5); use with a linear-frequency front-end, e.g.
                   frontend=dict(bank='linear', norm='percentile', n_bins=192, fmin=2000, fmax=9500, n_fft=2048).
Origin: research session (`ContourNet`) + fork review.
"""
import torch.nn as nn
from layers.registry import build_frontend
from layers.contour_topk import ContourTopK
from layers.ridge_contour import RidgeContour
from layers.temporal_dilated_block import TemporalDilatedBlock
from layers.sequence_head import SequenceHead


class ContourTCN(nn.Module):
    def __init__(self, n_classes=7, frontend=None, features='topk', ch=96, n_blocks=6, drop=0.2, head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        self.feat = RidgeContour() if features == 'ridge' else ContourTopK()
        self.stem = nn.Sequential(nn.Conv1d(self.feat.out_dim, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.SiLU())
        self.blocks = nn.Sequential(*[TemporalDilatedBlock(ch, 5, 2 ** (i % 4), drop=drop / 2) for i in range(n_blocks)])
        self.pool_t = nn.AvgPool1d(2, 2)
        self.head = SequenceHead(ch, n_classes, ch, 'none', drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.head(self.pool_t(self.blocks(self.stem(self.feat(spec)))))
