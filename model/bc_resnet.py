"""BC-ResNet (Kim et al. 2021) adapted to 4 s song windows: a tiny, streaming-friendly model family
(tau = 1: ~0.05 M params before the head; tau = 3: ~0.3 M) proven in keyword spotting and the DCASE
low-complexity tasks.

It answers "how small can the phone model get": StrnadNet is a home-made design in the same spirit
(frequency structure kept, 1-D temporal processing) but was never compared against the published
family. The original ends in global pooling; here the trunk output goes through the shared sequence
head, because element order has to stay visible.

Status: proposed (fork review, architecture idea 9). Untrained.
"""

import torch.nn as nn

from layers.bc_res_block import BCResBlock
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend
from layers.sequence_head import SequenceHead
from model.base import Model


class BCResNet(Model):
    """Front-end -> broadcasted-residual trunk -> frequency folded into channels -> sequence head.

    Args:
        n_classes: number of output classes.
        tau: width multiplier of the trunk; every channel count is `int(tau * base width)`.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        d: channels of the frame series inside the sequence head.
        temporal: kind of temporal module in the head: 'conv' | 'gru' | 'attn' | 'none'.
        n_temporal: number of layers of the temporal module.
        drop: dropout, used both in the trunk blocks and in the sequence head.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    def __init__(
        self,
        n_classes=7,
        tau=2.0,
        frontend=None,
        d=96,
        temporal='conv',
        n_temporal=2,
        drop=0.1,
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- channel plan: base widths of the stem and of the four stages, scaled by `tau`
        base_widths = (16, 8, 12, 16, 20)
        widths = [int(tau * base_width) for base_width in base_widths]
        stem_channels = widths[0]

        # --- stem: 5x5 convolution, stride 2 in frequency and time
        layers = [
            nn.Conv2d(self.frontend.out_channels, stem_channels, 5, (2, 2), 2, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        ]

        # --- 2-D convolutional body: four stages of broadcasted-residual blocks
        # Only the first block of a stage downsamples, and only along frequency; later stages look
        # further in time through a larger dilation.
        # (output channels, number of blocks, frequency stride, time dilation)
        stages = (
            (widths[1], 2, 1, 1),
            (widths[2], 2, 2, 2),
            (widths[3], 4, 2, 4),
            (widths[4], 4, 2, 8),
        )
        channels = stem_channels
        for stage_channels, n_blocks, stage_freq_stride, time_dilation in stages:
            for block_index in range(n_blocks):
                freq_stride = stage_freq_stride if block_index == 0 else 1
                block = BCResBlock(channels, stage_channels, freq_stride, time_dilation, drop)
                layers.append(block)
                channels = stage_channels

        # --- halve the frame rate
        layers.append(nn.AvgPool2d((1, 2)))  # 20 ms -> 40 ms frames
        self.cnn = nn.Sequential(*layers)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        # (the stem and three of the stages halve frequency: 16 in total)
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // 16

        # --- frame series -> clip embedding -> classifier (shared with the other backbones)
        self.head = SequenceHead(
            channels * freq_bins_left,
            n_classes,
            d,
            temporal,
            n_temporal,
            drop=drop,
            head=head,
            head_kw=head_kw,
        )

    def forward(self, x):
        # Shape comments: default arguments (tau = 2), 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        feature_map = self.cnn(spectrogram)  # (B, 40, 8, 100)   40 ms frames
        series = self.fold(feature_map)  # (B, 320, 100)
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(BCResNet())
