"""EfficientAT MobileNetV3 (AudioSet-pretrained, distilled from PaSST; MIT licence) + shared sequence head.

The strongest small audio CNNs in the literature (mn04: 0.98 M params, mn10: 4.9 M), and unlike the ImageNet
backbones in `timm_temporal.py` they already know what sound looks like. Candidate mobile model and cheap
second teacher.

Requires a clone of https://github.com/fschmid56/EfficientAT (it is not pip-installable); point
`EFFICIENTAT_DIR` at it. The repo's own top-level package is called `models`, which is why this repo keeps its
models under `model/`.

NOT exercised by this repo's tests (needs the clone and a weight download). The loading call follows the
EfficientAT README (`get_model(pretrained_name='mn10_as')`); the code below assumes the returned network
exposes its convolutional trunk as `.features` and takes a (B, 1, 128, T) log-mel made by the repo's own
`AugmentMelSTFT` (32 kHz, 128 mels, win 800, hop 320). Verify both on first use.
Status: proposed by the research literature report and both reviews; never run.
"""
import os
import sys
import torch
import torch.nn as nn
from layers.freq_to_channels import FreqToChannels
from layers.sequence_head import SequenceHead


class EfficientATTemporal(nn.Module):
    wave_input = True
    backbone_prefixes = ('enc.',)

    def __init__(self, n_classes=7, pretrained_name='mn10_as', d=192, temporal='conv', n_temporal=3, drop=0.2,
                 head='linear', head_kw=None):
        super().__init__()
        root = os.environ.get('EFFICIENTAT_DIR')
        if not root or not os.path.isdir(root):
            raise RuntimeError('set EFFICIENTAT_DIR to a clone of https://github.com/fschmid56/EfficientAT')
        sys.path.insert(0, root); cwd = os.getcwd(); os.chdir(root)      # the repo downloads weights into ./resources
        try:
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT
            net = get_model(pretrained_name=pretrained_name)
        finally:
            os.chdir(cwd); sys.path.remove(root)
        if not hasattr(net, 'features'):
            raise RuntimeError('EfficientAT model has no `.features` trunk; adapt model/efficientat_temporal.py to its API')
        self.frontend = None
        self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
        self.mel.eval()                                                  # disable its built-in training-time masking
        self.enc = net.features
        self.fold = FreqToChannels()
        with torch.no_grad():
            f = self.enc(torch.zeros(1, 1, 128, 400))
        self.head = SequenceHead(f.shape[1] * f.shape[2], n_classes, d, temporal, n_temporal, drop=drop, head=head, head_kw=head_kw)

    def train(self, mode=True):
        super().train(mode); self.mel.eval()
        return self

    def forward(self, wave):
        with torch.no_grad():
            spec = self.mel(wave).unsqueeze(1)
        return self.head(self.fold(self.enc(spec)))
