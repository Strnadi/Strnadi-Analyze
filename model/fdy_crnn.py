"""FDY-CRNN: frequency-dynamic conv stages + bidirectional GRU (Nam et al. 2022), the standard
strong sound-event-detection network, with the shared attentive head on top.

Frequency-dynamic kernels adapt along the frequency axis, which suits patterns defined by absolute
pitch better than one shared kernel plus a coordinate channel. Because it is an SED architecture
its `frame_logits` are a natural song detector.

Status: proposed (fork review, architecture idea 4). Untrained.
"""

import torch.nn as nn

from layers.freq_dynamic_conv import FreqDynamicConv
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend
from layers.sequence_head import SequenceHead
from model.base import Model


class FDYCRNN(Model):
    """Front-end -> frequency-dynamic conv stages -> frequency folded -> BiGRU sequence head.

    Args:
        n_classes: number of output classes.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        widths: output channels of the conv stages, one entry per stage. Every stage halves
            frequency; the first two also halve time.
        d: channels of the frame series inside the sequence head (the GRU output width).
        drop: dropout in the sequence head.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    def __init__(
        self,
        n_classes=7,
        frontend=None,
        widths=(16, 32, 64, 128),
        d=160,
        drop=0.2,
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- 2-D convolutional body: conv -> BatchNorm -> SiLU -> average pool, once per stage
        layers = []
        channels = self.frontend.out_channels
        for stage, stage_width in enumerate(widths):
            if stage == 0:
                # first layer static, as in the paper
                conv = nn.Conv2d(channels, stage_width, 3, padding=1, bias=False)
            else:
                conv = FreqDynamicConv(channels, stage_width)
            time_pool = 2 if stage < 2 else 1
            layers.append(conv)
            layers.append(nn.BatchNorm2d(stage_width))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.AvgPool2d((2, time_pool)))
            channels = stage_width
        self.cnn = nn.Sequential(*layers)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // 2 ** len(widths)

        # --- frame series -> clip embedding -> classifier; the temporal module is a BiGRU
        #     (the "RNN" of the CRNN)
        self.head = SequenceHead(
            channels * freq_bins_left,
            n_classes,
            d,
            'gru',
            2,
            drop=drop,
            head=head,
            head_kw=head_kw,
        )

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        feature_map = self.cnn(spectrogram)  # (B, 128, 8, 100)   40 ms frames
        series = self.fold(feature_map)  # (B, 1024, 100)
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(FDYCRNN())
