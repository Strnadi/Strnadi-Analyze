"""The production architecture ("good-model-2" lineage), ported from Keras: 224-bin mel (3-9 kHz) ->
magma colour map -> ImageNet DenseNet121 -> attentive statistics pooling over all patches -> BN ->
256 -> 64 -> classes. 7.5 M params; about 12x the training cost per step of StrnadNet on CPU.

Kept as the control for "is it the data or the architecture?". Differences from the Keras script
that are corrections, not changes of design:
- input is normalised with the ImageNet mean/std the pretrained weights expect (the Keras model fed
  raw [0, 1]);
- SpecAugment masks to the example minimum, per example (see layers/spec_augment.py);
- splits, augmentation and checkpointing come from the shared trainer (recording-grouped folds; the
  Keras script split randomly by file with a time-seeded split, so its reported validation scores
  were optimistic).

On held-out Vizina males the deployed model of this architecture calls ~45 % of complete BhBl songs
"Unfinished" (BhBl recall 0.55 over 58 males).

Origin: legacy `model/densenet.py`.
"""

import torch
import torch.nn.functional as F

from layers.attentive_stats_pool_2d import AttentiveStatsPool2D
from layers.mlp_head import MLPHead
from layers.registry import build_frontend
from model.base import Model


class DenseNet121AttnPool(Model):
    """Colour-mapped mel spectrogram -> DenseNet121 -> pooling over all patches -> MLP classifier.

    Args:
        n_classes: number of output classes.
        pretrained: True loads the torchvision ImageNet weights of DenseNet121 (a download on first
            use); False starts from random weights.
        frontend: dict of front-end settings (see `layers.registry.build_frontend`) that override
            this model's own defaults: 224-bin mel on 3-9 kHz with the 3-channel 'magma'
            normalisation.

    `forward` returns a dict: `logits` (B, n_classes) and `emb` (B, 2048), the pooled features.
    """

    backbone_prefixes = ('cnn.',)  # parameter-name prefixes that get `backbone_lr_factor`
    spec_fill = 'min'  # SpecAugment fills masked regions with the example minimum

    def __init__(self, n_classes=7, pretrained=True, frontend=None):
        super().__init__()
        import torchvision  # imported here, not at the top: only the legacy models need it

        # --- front-end: waveform -> mel spectrogram in dB, scaled to [0, 1], coloured with magma
        frontend_config = dict(bank='mel', norm='magma', n_bins=224, fmin=3000.0, fmax=9000.0)
        frontend_config.update(frontend or {})
        self.frontend = build_frontend(**frontend_config)

        # --- 2-D convolutional body: the convolutional part of torchvision's DenseNet121
        weights = 'DEFAULT' if pretrained else None
        self.cnn = torchvision.models.densenet121(weights=weights).features

        # --- ImageNet colour statistics the pretrained weights expect (constants, not saved)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', imagenet_mean, persistent=False)
        self.register_buffer('std', imagenet_std, persistent=False)

        # --- feature map -> clip embedding: attention-weighted mean and std over all patches
        self.pool = AttentiveStatsPool2D(1024)

        # --- classifier head: BN -> 256 -> 64 -> classes
        self.cls = MLPHead(2048, n_classes, hidden=(256, 64), drop=0.3)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 3, 224, 400)   RGB image, 10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        image = (spectrogram - self.mean) / self.std  # (B, 3, 224, 400)
        feature_map = self.cnn(image)  # (B, 1024, 7, 12)
        feature_map = F.relu(feature_map)  # (B, 1024, 7, 12)
        embedding = self.pool(feature_map)  # (B, 2048)   mean and std of the 84 patches
        outputs = self.cls(embedding)  # dict with `logits` (B, n_classes)
        outputs['emb'] = embedding
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(DenseNet121AttnPool(pretrained=False))
