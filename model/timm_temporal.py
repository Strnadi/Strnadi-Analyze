"""Any timm CNN as the 2-D backbone, with the shared sequence head on its last feature map.

The feature map keeps its time axis (output stride 16 where the architecture allows it); remaining
frequency bins are folded into channels. Used for the architecture comparison (ConvNeXt-atto,
MobileNetV3, RegNetY, EfficientNet-B0, ResNet18, DenseNet121 as the "same data, production
architecture" control).

Fairness note (base-session review): pretrained backbones need a lower learning rate than the
from-scratch head. Train with `backbone_lr_factor` (e.g. 0.1) in the experiment settings; the research
sweep used the from-scratch settings for everything, so its ranking mostly reflected
hyper-parameters.

Origin: research session (`TimmNet`).
"""

import torch

from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend
from layers.sequence_head import SequenceHead
from model.base import Model


class TimmTemporal(Model):
    """Front-end -> timm CNN -> frequency folded into channels -> shared sequence head.

    Args:
        n_classes: number of output classes.
        backbone: name of the timm architecture (anything `timm.create_model` accepts with
            `features_only=True`).
        pretrained: True loads timm's pretrained weights (a download on first use); False starts
            from random weights.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        d: channels of the frame series inside the sequence head.
        temporal: kind of temporal module in the head: 'conv' | 'gru' | 'attn' | 'none'.
        n_temporal: number of layers of the temporal module.
        drop: dropout in the sequence head.
        output_stride: total downsampling asked of the backbone (16 keeps more of the time axis
            than the usual 32). Backbones that do not accept the argument are built without it.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    backbone_prefixes = ('enc.',)  # parameter-name prefixes that get `backbone_lr_factor`

    def __init__(
        self,
        n_classes=7,
        backbone='convnext_atto',
        pretrained=True,
        frontend=None,
        d=192,
        temporal='conv',
        n_temporal=3,
        drop=0.2,
        output_stride=16,
        head='linear',
        head_kw=None,
    ):
        super().__init__()
        import timm  # imported here, not at the top of the file: only this model needs timm

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- 2-D convolutional body: a timm CNN that returns only its last feature map
        backbone_options = dict(
            pretrained=pretrained,
            in_chans=self.frontend.out_channels,
            features_only=True,
            out_indices=(-1,),
        )
        try:
            self.enc = timm.create_model(backbone, output_stride=output_stride, **backbone_options)
        except Exception:
            # this architecture does not let us choose the output stride: take its own
            self.enc = timm.create_model(backbone, **backbone_options)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        self.fold = FreqToChannels()
        # The backbone's output size is not known in advance: measure it on a dummy spectrogram.
        # In eval mode, so the all-zero input does not leak into the (pretrained) BatchNorm
        # running statistics.
        dummy_spectrogram = torch.zeros(1, self.frontend.out_channels, self.frontend.n_bins, 400)
        self.enc.eval()
        with torch.no_grad():
            dummy_feature_map = self.enc(dummy_spectrogram)[-1]  # (1, C', F', T')
        self.enc.train()
        channels = dummy_feature_map.shape[1]
        freq_bins_left = dummy_feature_map.shape[2]

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
        # Shape comments: default arguments (ConvNeXt-atto, output stride 16), 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        feature_map = self.enc(spectrogram)[-1]  # (B, 320, 8, 25)   160 ms frames
        series = self.fold(feature_map)  # (B, 2560, 25)
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(TimmTemporal(pretrained=False))
