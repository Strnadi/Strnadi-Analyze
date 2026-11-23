"""DenseNet built from scratch with configurable block counts (e.g. [4, 8, 16, 12] or [2, 4, 8, 6])
on a 1-channel spectrogram, with the shared sequence head.

Origin: legacy `layers/densenet.py` (a copy of the Keras DenseNet with custom `blocks`, commented
out in `model/densenet.py`). The legacy version pooled over patches; this one keeps the time axis
like every other model here, because pooled patches cannot represent element order.
"""

import torch.nn as nn

from layers.dense_block import DenseBlock
from layers.dense_transition import DenseTransition
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend
from layers.sequence_head import SequenceHead
from model.base import Model


class DenseNetSmall(Model):
    """Front-end -> DenseNet body -> frequency folded into channels -> shared sequence head.

    Args:
        n_classes: number of output classes.
        blocks: number of layers in each dense block, one entry per block. Every block but the last
            is followed by a transition that halves channels, frequency and time.
        growth: channels every dense layer adds (the DenseNet "growth rate"); the stem has twice
            as many.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        d: channels of the frame series inside the sequence head.
        temporal: kind of temporal module in the head: 'conv' | 'gru' | 'attn' | 'none'.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    def __init__(
        self,
        n_classes=7,
        blocks=(4, 8, 16, 12),
        growth=32,
        frontend=None,
        d=160,
        temporal='conv',
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- stem: 7x7 convolution, stride 2 in frequency and time
        channels = 2 * growth
        layers = [
            nn.Conv2d(self.frontend.out_channels, channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        # --- 2-D convolutional body: dense blocks with a downsampling transition between them
        for block_index, n_layers in enumerate(blocks):
            dense_block = DenseBlock(channels, n_layers, growth)
            layers.append(dense_block)
            channels = dense_block.out_channels

            is_last_block = block_index == len(blocks) - 1
            if not is_last_block:
                transition = DenseTransition(channels)
                layers.append(transition)
                channels = transition.out_channels

        # --- final normalisation of the feature map
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(*layers)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        # (the stem and every transition halve frequency: 2 ** len(blocks) in total)
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // (2 ** len(blocks))

        # --- frame series -> clip embedding -> classifier (shared with the other backbones)
        self.head = SequenceHead(
            channels * freq_bins_left,
            n_classes,
            d,
            temporal,
            head=head,
            head_kw=head_kw,
        )

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        feature_map = self.cnn(spectrogram)  # (B, 728, 8, 25)   160 ms frames
        series = self.fold(feature_map)  # (B, 5824, 25)
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(DenseNetSmall())
