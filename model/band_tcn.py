"""BandTCN: a song detector of 8 thousand parameters (1-D convolutions over coarse bands).

A detector only has to say "a Yellowhammer is singing in this window", so it needs neither fine
frequency resolution nor the order of the terminal elements. Design:

- front-end: 40 mel bands on 1.5-11 kHz, one frame per 20 ms, per-band median removed (the same
  normalisation as the classifier, at a third of its resolution);
- the bands are treated as the CHANNELS of a 1-D signal over time, so every convolution sees the
  whole spectrum at once and the network is 1-D: depthwise-separable blocks (a 5-frame depthwise
  convolution, then a 1x1 convolution), the second and third with stride 2;
- attentive pooling over time -> clip embedding -> linear classifier.

In the spirit of the entries to the 2025 Yellowhammer detection challenge, where a 7.8 kB model
reached 0.99 average precision. The output has the seven Strnadi classes so that it trains and
scans with the same code as the classifiers; a detector uses `1 - p(None)`.
"""

import torch.nn as nn

from layers.attentive_stats_pool import AttentiveStatsPool
from layers.registry import build_frontend
from model.base import Model


class SeparableBlock(nn.Module):
    """Depthwise 1-D convolution over time, then a 1x1 convolution; BatchNorm and SiLU after each."""

    def __init__(self, in_channels, out_channels, stride=1, kernel=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel,
                stride,
                kernel // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )

    def forward(self, series):
        return self.layers(series)


class BandTCN(Model):
    """Waveform (or spectrogram) -> class logits from a tiny 1-D network.

    Args:
        n_classes: number of output classes.
        n_bands: mel bands of the front-end (they become the input channels).
        width: channels of the convolution blocks.
        n_blocks: number of depthwise-separable blocks.
        hop: samples between spectrogram frames (640 = 20 ms).
        drop: dropout before the classifier.

    `forward` returns a dict: `logits` (B, n_classes), `emb` (B, width), `att` (B, T) frame weights.
    """

    def __init__(self, n_classes=7, n_bands=40, width=32, n_blocks=4, hop=640, drop=0.1):
        super().__init__()
        self.frontend = build_frontend(n_bins=n_bands, hop=hop)

        blocks = [SeparableBlock(n_bands, width)]
        for index in range(1, n_blocks):
            blocks.append(SeparableBlock(width, width, stride=2 if index in (1, 2) else 1))
        self.blocks = nn.Sequential(*blocks)

        self.pool = AttentiveStatsPool(width)
        self.emb = nn.Sequential(nn.Linear(2 * width, width), nn.BatchNorm1d(width))
        self.drop = nn.Dropout(drop)
        self.cls = nn.Linear(width, n_classes)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 40, 200)   20 ms frames
        else:
            spectrogram = x
        bands = spectrogram[:, 0]  # (B, 40, 200)   bands as channels
        series = self.blocks(bands)  # (B, 32, 50)   80 ms frames
        pooled, frame_weights = self.pool(series)  # (B, 64), (B, 50)
        embedding = self.emb(pooled)  # (B, 32)
        logits = self.cls(self.drop(embedding))  # (B, n_classes)
        return {'logits': logits, 'emb': embedding, 'att': frame_weights}


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(BandTCN())
