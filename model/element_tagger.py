"""Element-sequence model: label every 20 ms frame as background / first-phrase syllable / B / C / D / E / X
(Hansen's element names), the TweetyNet / DAS formulation of birdsong annotation, and read the dialect off
the sequence of terminal elements.

Why this formulation:
- "Unfinished" is simply a sequence with no (or one) terminal element; it stops being a fuzzy class;
- an unknown dialect is an unseen sequence (e.g. "BD"), not a forced choice among 5 known ones, which matters
  for the Italian data;
- the output shows the scientists WHERE the dialect sits;
- "sequence of unit types" is the generic output the BioAcu platform needs for any species.

The model also carries the usual clip head, so it trains on clip labels alone (frame targets -1 are ignored)
and adds frame supervision wherever element boxes exist (Vizina component annotations, counterfactual surgery
positions: `processing/build_element_frames.py`). `decode_elements` turns frame predictions into a dialect
string; two B elements are ordered by their ridge frequency (BhBl vs BlBh).
Status: proposed (fork review, architecture idea 6). Untrained.
"""
import numpy as np
import torch.nn as nn
from layers.registry import build_frontend, build_head
from layers.coord_freq import CoordFreq
from layers.conv_bn_act import ConvBNAct
from layers.res_block import ResBlock
from layers.freq_to_channels import FreqToChannels
from layers.temporal_bigru import TemporalBiGRU
from layers.attentive_stats_pool import AttentiveStatsPool
from layers.frame_tag_head import FrameTagHead

ELEMENTS = ['bg', 'phrase', 'B', 'C', 'D', 'E', 'X']


class ElementTagger(nn.Module):
    frame_hop_s = 0.02

    def __init__(self, n_classes=7, frontend=None, widths=(16, 32, 64, 128), d=160, emb_dim=128, drop=0.2,
                 head='factorized', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        self.coord = CoordFreq()
        layers = [ConvBNAct(self.frontend.out_channels + 1, widths[0], 3, (2, 2))]; c = widths[0]
        for w in widths[1:]:
            layers.append(ResBlock(c, w, (2, 1))); c = w                 # frequency only: frames stay at 20 ms
        self.cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        self.proj = nn.Sequential(nn.Conv1d(c * (self.frontend.n_bins // 2 ** len(widths)), d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.temporal = TemporalBiGRU(d)
        self.tags = FrameTagHead(d, len(ELEMENTS))
        self.pool = AttentiveStatsPool(d)
        self.emb = nn.Sequential(nn.Linear(2 * d, emb_dim), nn.BatchNorm1d(emb_dim))
        self.drop = nn.Dropout(drop)
        self.cls = build_head(head, emb_dim, n_classes, **(head_kw or {}))

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        h = self.temporal(self.proj(self.fold(self.cnn(self.coord(spec)))))
        pooled, att = self.pool(h)
        e = self.emb(pooled)
        out = self.cls(self.drop(e))
        out.update(emb=e, att=att, frames=h, element_logits=self.tags(h))
        return out


def decode_elements(tags, ridge_hz=None, min_frames=3):
    """tags: (T,) int array of per-frame element indices; ridge_hz: optional (T,) dominant frequency per frame.
    Returns (dialect_string, [(element, start_frame, end_frame), ...]).
    The dialect string is the terminal elements after the last phrase run, e.g. 'BC', 'XB', 'BhBl', 'B'
    (one element only = unfinished by the project's rule), '' (no terminal element)."""
    tags = np.asarray(tags); runs, s = [], 0
    for t in range(1, len(tags) + 1):
        if t == len(tags) or tags[t] != tags[s]:
            if tags[s] != 0 and t - s >= min_frames:
                runs.append((ELEMENTS[tags[s]], s, t))
            s = t
    last_phrase = max((i for i, r in enumerate(runs) if r[0] == 'phrase'), default=-1)
    term = runs[last_phrase + 1:]
    names = [r[0] for r in term]
    if names == ['B', 'B'] and ridge_hz is not None:
        f = [float(np.median(ridge_hz[a:b])) for _, a, b in term]
        names = ['Bh', 'Bl'] if f[0] > f[1] else ['Bl', 'Bh']
    return ''.join(names), runs
