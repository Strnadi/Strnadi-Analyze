"""Low-latency legacy variant, ported from Keras: 224-bin mel (3-9 kHz) -> [dB, delta, delta-delta]
channels scaled to [0, 1] -> ImageNet MobileNetV3-Small -> attentive statistics pooling -> BN ->
128 -> classes.

Same corrections as `densenet121_attnpool.py` (ImageNet normalisation, per-example SpecAugment with
min fill, grouped folds via the shared trainer).

Origin: legacy `model/mobilenet.py`.
"""

import torch

from layers.attentive_stats_pool_2d import AttentiveStatsPool2D
from layers.mlp_head import MLPHead
from layers.registry import build_frontend
from model.base import Model


class MobileNetV3Delta(Model):
    """Mel + time-derivative channels -> MobileNetV3-Small -> pooling over patches -> MLP head.

    Args:
        n_classes: number of output classes.
        pretrained: True loads the torchvision ImageNet weights of MobileNetV3-Small (a download on
            first use); False starts from random weights.
        frontend: dict of front-end settings (see `layers.registry.build_frontend`) that override
            this model's own defaults: 224-bin mel on 3-9 kHz with the 3-channel 'delta'
            normalisation (dB, first and second time derivative).

    `forward` returns a dict: `logits` (B, n_classes) and `emb` (B, 1152), the pooled features.
    """

    backbone_prefixes = ('cnn.',)  # parameter-name prefixes that get `backbone_lr_factor`
    spec_fill = 'min'  # SpecAugment fills masked regions with the example minimum

    def __init__(self, n_classes=7, pretrained=True, frontend=None):
        super().__init__()
        import torchvision  # imported here, not at the top: only the legacy models need it

        # --- front-end: waveform -> [dB, delta, delta-delta] mel channels, each scaled to [0, 1]
        frontend_config = dict(bank='mel', norm='delta', n_bins=224, fmin=3000.0, fmax=9000.0)
        frontend_config.update(frontend or {})
        self.frontend = build_frontend(**frontend_config)

        # --- 2-D convolutional body: the convolutional part of torchvision's MobileNetV3-Small
        weights = 'DEFAULT' if pretrained else None
        self.cnn = torchvision.models.mobilenet_v3_small(weights=weights).features

        # --- ImageNet colour statistics the pretrained weights expect (constants, not saved)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', imagenet_mean, persistent=False)
        self.register_buffer('std', imagenet_std, persistent=False)

        # --- feature map -> clip embedding: attention-weighted mean and std over all patches
        self.pool = AttentiveStatsPool2D(576)

        # --- classifier head: BN -> 128 -> classes
        self.cls = MLPHead(1152, n_classes, hidden=(128,), drop=0.3)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 3, 224, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        image = (spectrogram - self.mean) / self.std  # (B, 3, 224, 400)
        feature_map = self.cnn(image)  # (B, 576, 7, 13)
        embedding = self.pool(feature_map)  # (B, 1152)   mean and std of the 91 patches
        outputs = self.cls(embedding)  # dict with `logits` (B, n_classes)
        outputs['emb'] = embedding
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(MobileNetV3Delta(pretrained=False))
