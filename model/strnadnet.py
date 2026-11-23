"""StrnadNet: small frequency-aware CNN + temporal module + attentive pooling (0.90 M params by default).

Design, each choice tied to how dialects are defined:
- front-end at ~60-70 Hz per bin where the song lives; a frequency-coordinate channel;
- 4 residual stages that downsample frequency by 16 in total but time only by 4 (40 ms frames);
- the remaining 8 frequency bins are folded into channels, so absolute pitch survives (Bh vs Bl);
- a temporal module with > 1 s receptive field sees element ORDER (BhBl vs BlBh), which pooled embeddings lose;
- attentive statistics pooling gives the clip embedding and a "where is the song" weight per frame.

Results (research session, recording-grouped 5-fold CV, 3,744 strong clips): strong labels only 0.745 macro-F1;
with weak pools + `head='factorized'` (R2) 0.774. Dialect accuracy given a complete song is ~97 %; the
remaining error is complete-vs-Unfinished (see docs/REVIEW.md).
Origin: research session (`strnadi_ml/models.py`); layer names kept so its checkpoints load
(`training/checkpoint.py::load_research_state`).
"""
import torch.nn as nn
from layers.registry import build_frontend, build_temporal, build_head
from layers.coord_freq import CoordFreq
from layers.conv_bn_act import ConvBNAct
from layers.tonal_line_conv import TonalLineConv
from layers.res_block import ResBlock
from layers.inverted_res_block import InvertedResBlock
from layers.freq_to_channels import FreqToChannels
from layers.attentive_stats_pool import AttentiveStatsPool


class StrnadNet(nn.Module):
    def __init__(self, n_classes=7, frontend=None, widths=(16, 32, 64, 128), depths=(1, 1, 2, 1), d=160, n_temporal=3,
                 emb_dim=128, dw=False, drop=0.2, stem_stride=(2, 2), time_strides=(1, 2, 1, 1), coord=True,
                 stem='conv', temporal='conv', head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        self.coord = CoordFreq() if coord else nn.Identity()
        cin = self.frontend.out_channels + int(coord)
        stem_stride = tuple(stem_stride)
        self.stem = (TonalLineConv(cin, widths[0], stride=stem_stride) if stem == 'tonal'
                     else ConvBNAct(cin, widths[0], 3, stem_stride))
        # total frequency downsampling is always 2 ** len(widths): if the stem strides in frequency, stage 0 does not
        first_fdown = 1 if stem_stride[0] == 2 else 0
        blocks, c = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, depths)):
            for j in range(n):
                s = ((2 if i >= first_fdown else 1), time_strides[i]) if j == 0 else (1, 1)
                blocks.append(InvertedResBlock(c, w, s) if (dw and i > 0) else ResBlock(c, w, s)); c = w
        self.cnn = nn.Sequential(*blocks)
        self.fold = FreqToChannels()
        fdim = self.frontend.n_bins // (2 ** len(widths))
        self.proj = nn.Sequential(nn.Conv1d(c * fdim, d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.temporal = build_temporal(temporal, d, n_temporal, drop / 2)
        self.pool = AttentiveStatsPool(d)
        self.frame_head = nn.Conv1d(d, 1, 1)
        self.emb = nn.Sequential(nn.Linear(2 * d, emb_dim), nn.BatchNorm1d(emb_dim))
        self.drop = nn.Dropout(drop)
        self.cls = build_head(head, emb_dim, n_classes, **(head_kw or {}))

    def frames(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.temporal(self.proj(self.fold(self.cnn(self.stem(self.coord(spec))))))

    def forward(self, x):
        h = self.frames(x)
        pooled, att = self.pool(h)
        e = self.emb(pooled)
        out = self.cls(self.drop(e))
        out.update(emb=e, att=att, frames=h, frame_logits=self.frame_head(h).squeeze(1))
        return out
