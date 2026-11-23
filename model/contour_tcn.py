"""1-D temporal conv net over per-frame pitch-contour features instead of the full spectrogram
(0.27 M params, ~0.1 GMAC per 4 s: about 5x cheaper than StrnadNet, and interpretable).

features='topk'  : research version (top-3 mel peaks, quantised to one bin);
features='ridge' : sub-bin ridge track (fork review idea 5); use with a linear-frequency front-end,
                   e.g. frontend=dict(bank='linear', norm='percentile', n_bins=192, fmin=2000,
                   fmax=9500, n_fft=2048).

Origin: research session (`ContourNet`) + fork review.
"""

import torch.nn as nn

from layers.contour_topk import ContourTopK
from layers.registry import build_frontend
from layers.ridge_contour import RidgeContour
from layers.sequence_head import SequenceHead
from layers.temporal_dilated_block import TemporalDilatedBlock
from model.base import Model


class ContourTCN(Model):
    """Front-end -> a few contour features per frame -> dilated 1-D convolutions -> sequence head.

    Args:
        n_classes: number of output classes.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        features: 'topk' (`ContourTopK`: the 3 strongest bins, band energies, flux) or 'ridge'
            (`RidgeContour`: sub-bin track of the dominant tone).
        ch: channels of the 1-D network.
        n_blocks: number of dilated temporal blocks; their dilation cycles through 1, 2, 4, 8.
        drop: dropout in the sequence head; the temporal blocks use half of it.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    def __init__(
        self,
        n_classes=7,
        frontend=None,
        features='topk',
        ch=96,
        n_blocks=6,
        drop=0.2,
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- spectrogram -> a handful of pitch-contour features per frame (no learned weights)
        if features == 'ridge':
            self.feat = RidgeContour()
        else:
            self.feat = ContourTopK()

        # --- 1-D stem: features -> `ch` channels
        self.stem = nn.Sequential(
            nn.Conv1d(self.feat.out_dim, ch, 5, padding=2),
            nn.BatchNorm1d(ch),
            nn.SiLU(),
        )

        # --- temporal body: residual dilated blocks, dilation 1, 2, 4, 8, 1, 2, ...
        temporal_blocks = []
        for block_index in range(n_blocks):
            dilation = 2 ** (block_index % 4)
            temporal_blocks.append(TemporalDilatedBlock(ch, 5, dilation, drop=drop / 2))
        self.blocks = nn.Sequential(*temporal_blocks)

        # --- halve the frame rate: 10 ms -> 20 ms frames
        self.pool_t = nn.AvgPool1d(2, 2)

        # --- frame series -> clip embedding -> classifier; the body above is the temporal module,
        #     so the head gets none of its own ('none')
        self.head = SequenceHead(ch, n_classes, ch, 'none', drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        contour = self.feat(spectrogram)  # (B, 15, 400)   15 features per frame
        series = self.stem(contour)  # (B, 96, 400)
        series = self.blocks(series)  # (B, 96, 400)
        series = self.pool_t(series)  # (B, 96, 200)   20 ms frames
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(ContourTCN())
