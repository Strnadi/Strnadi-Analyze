"""Element-sequence model: label every 20 ms frame as background / first-phrase syllable / B / C / D
/ E / L / H / X (the element names of `losses/element_ctc.py`), the TweetyNet / DAS formulation of birdsong annotation, and read
the dialect off the sequence of terminal elements.

Why this formulation:
- "Unfinished" is simply a sequence with no (or one) terminal element; it stops being a fuzzy
  class;
- an unknown dialect is an unseen sequence (e.g. "BD"), not a forced choice among 5 known ones,
  which matters for the Italian data;
- the output shows the scientists WHERE the dialect sits;
- "sequence of unit types" is the generic output the BioAcu platform needs for any species.

The model also carries the usual clip head, so it trains on clip labels alone (frame targets -1 are
ignored) and adds frame supervision wherever element boxes exist (Vizina component annotations,
counterfactual surgery positions: `processing/build_element_frames.py`). `decode_elements` turns
frame predictions into a dialect string; two B elements are ordered by their ridge frequency (BhBl
vs BlBh).

Status: proposed (fork review, architecture idea 6). Untrained.
"""

import numpy as np
import torch.nn as nn

from layers.attentive_stats_pool import AttentiveStatsPool
from layers.conv_bn_act import ConvBNAct
from layers.coord_freq import CoordFreq
from layers.frame_tag_head import FrameTagHead
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend, build_head
from layers.res_block import ResBlock
from layers.temporal_bigru import TemporalBiGRU
from losses.element_ctc import (
    FRAME_TAGS,
    STRNADI_GRAMMAR,
    canonical_elements,
    decode_frame_tags,
)
from model.base import Model

# element index -> name; index 0 is background, index 1 the first-phrase syllables
ELEMENTS = FRAME_TAGS


class ElementTagger(Model):
    """Waveform (or spectrogram) -> an element label per 20 ms frame, plus the usual clip outputs.

    Args:
        n_classes: number of clip-level output classes.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        widths: output channels of the stem (first entry) and of the residual stages (the rest).
            The stem and every stage halve frequency, 2 ** len(widths) in total.
        d: channels of the frame series (the GRU width).
        emb_dim: size of the clip embedding.
        drop: dropout on the clip embedding before the classifier.
        head: kind of clip classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the clip classifier head.

    `forward` returns a dict: `logits` (B, n_classes), `emb` (B, emb_dim), `att` (B, T) frame
    weights, `frames` (B, d, T), `element_logits` (B, len(ELEMENTS), T), plus whatever the chosen
    head adds. T is one frame per 20 ms.
    """

    frame_hop_s = 0.02  # seconds per output frame

    def __init__(
        self,
        n_classes=7,
        frontend=None,
        widths=(16, 32, 64, 128),
        d=160,
        emb_dim=128,
        drop=0.2,
        head='factorized',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram, plus the frequency-coordinate channel
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)
        self.coord = CoordFreq()
        input_channels = self.frontend.out_channels + 1

        # --- 2-D convolutional body: stem (halves frequency and time), then residual stages
        layers = [ConvBNAct(input_channels, widths[0], 3, (2, 2))]
        channels = widths[0]
        for stage_width in widths[1:]:
            stage_stride = (2, 1)  # frequency only: frames stay at 20 ms
            layers.append(ResBlock(channels, stage_width, stage_stride))
            channels = stage_width
        self.cnn = nn.Sequential(*layers)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // 2 ** len(widths)
        self.proj = nn.Sequential(
            nn.Conv1d(channels * freq_bins_left, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.SiLU(),
        )

        # --- temporal module: bidirectional GRU over the frames
        self.temporal = TemporalBiGRU(d)

        # --- per-frame element classifier
        self.tags = FrameTagHead(d, len(ELEMENTS))

        # --- frame series -> clip embedding
        self.pool = AttentiveStatsPool(d)
        self.emb = nn.Sequential(
            nn.Linear(2 * d, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        # --- clip classifier head
        self.drop = nn.Dropout(drop)
        head_config = head_kw or {}
        self.cls = build_head(head, emb_dim, n_classes, **head_config)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        cnn_input = self.coord(spectrogram)  # (B, 2, 128, 400)   + frequency coordinate
        feature_map = self.cnn(cnn_input)  # (B, 128, 8, 200)   20 ms frames
        series = self.fold(feature_map)  # (B, 1024, 200)
        series = self.proj(series)  # (B, 160, 200)
        frames = self.temporal(series)  # (B, 160, 200)

        # --- clip-level outputs
        pooled, frame_weights = self.pool(frames)  # (B, 320), (B, 200)
        embedding = self.emb(pooled)  # (B, 128)
        outputs = self.cls(self.drop(embedding))  # dict with `logits` (B, n_classes)

        # --- frame-level outputs
        element_logits = self.tags(frames)  # (B, 7, 200)
        outputs.update(
            emb=embedding,
            att=frame_weights,
            frames=frames,
            element_logits=element_logits,
        )
        return outputs


def decode_elements(tags, min_frames=3):
    """Turn per-frame element indices into a class of the project.

    Args:
        tags: (T,) int array of per-frame element indices (positions in `ELEMENTS`).
        min_frames: runs shorter than this many frames are dropped.

    Returns (class_name, [(element, start_frame, end_frame), ...]): the class the sequence of
    elements stands for (`losses.element_ctc.ElementGrammar`), e.g. 'BhBl' for phrase, B, L and
    'Unfinished' for a phrase with one terminal element or none, and the runs themselves, which
    give every element's start and end.
    """
    runs = decode_frame_tags(tags, min_frames)
    names = canonical_elements([name for name, _, _ in runs])
    return STRNADI_GRAMMAR.class_of_sequence(names), runs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(ElementTagger())
