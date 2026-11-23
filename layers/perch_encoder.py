"""Perch 2.0 (Google, EfficientNet-B3, ~12 M params, Apache-2.0) as a torch module.

Uses the PyTorch port `janclemenslab/perch2_torch` (torch.hub; weights from the Hugging Face repo
DAS-DeepAudioSegmenter/perch2-torch; the port reports <= 1.1e-5 max abs difference to the ONNX
reference).
Input: mono 32 kHz waveform (B, N); Perch works on 5 s windows, shorter input is zero-padded here.
Output: dict(embedding (B, 1536), spatial (B, 1536, T', F')).

`trainable=False` freezes it (frozen-embedding heads, distillation teacher); `trainable=True`
fine-tunes it (the strongest teacher both reviews ask for; frozen Perch + a temporal head already
equals the from-scratch baseline, 0.741 vs 0.745 macro-F1).

NOT exercised by the tests in this repo: it needs a network download. The loading call and output
names follow the port's README; whether `forward` pads / windows exactly as assumed must be checked
on first use (`python -m tests.check_perch`). The legacy TensorFlow path
(`processing/extract_foundation_emb.py`) remains the verified way to get frozen embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerchEncoder(nn.Module):
    SR = 32000  # sample rate, Hz
    WINDOW = 5 * 32000  # samples in one 5 s window

    def __init__(self, trainable=False, repo='janclemenslab/perch2_torch', entry='perch_v2'):
        super().__init__()
        self.net = torch.hub.load(repo, entry)
        self.trainable = trainable
        if not trainable:
            self.net.eval()
            for param in self.net.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        """Like `nn.Module.train`, but a frozen Perch always stays in eval mode."""
        super().train(mode)
        if not self.trainable:
            self.net.eval()
        return self

    def forward(self, wave):
        """Waveform (B, N) -> dict: `embedding` (B, 1536), `spatial` (B, 1536, T', F')."""
        n_samples = wave.shape[-1]
        if n_samples < self.WINDOW:
            # zero-pad the end up to one window
            wave = F.pad(wave, (0, self.WINDOW - n_samples))
        # gradients only when fine-tuning, and never inside a no_grad block of the caller
        needs_grad = self.trainable and torch.is_grad_enabled()
        with torch.set_grad_enabled(needs_grad):
            first_window = wave[..., : self.WINDOW]  # (B, WINDOW); anything after 5 s is dropped
            out = self.net(first_window)
        # move the last axis to position 1: (B, T', F', 1536) -> (B, 1536, T', F')
        spatial = out['spatial_embedding'].permute(0, 3, 1, 2)
        return {'embedding': out['embedding'], 'spatial': spatial}
