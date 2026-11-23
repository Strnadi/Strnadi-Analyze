"""SpecAugment (Park et al. 2019): random frequency and time masks, independent per example, training only.

Deliberately narrow defaults (2 x <= 8 of 128 bins, 2 x <= 0.1 s): a wider mask can erase a whole terminal
element and silently turn a dialect into "Unfinished".

Fixes over the legacy Keras layer (`layers/spec_augument.py`):
- masks were drawn once per batch (every example got the same mask);
- masked cells were set to 0.0 on a dB spectrogram, i.e. to the *loudest* possible value after the min-max
  colour mapping, painting bright bars instead of hiding energy. `fill='min'` masks to the example's minimum;
  `fill='zero'` is right for floor-normalised inputs (log-median / percentile / PCEN), where 0 is the noise floor.
"""
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, freq_width=8, time_masks=2, time_width=10, max_time_ratio=0.2, fill='zero'):
        super().__init__()
        self.fm, self.fw, self.tm, self.tw, self.ratio, self.fill = freq_masks, freq_width, time_masks, time_width, max_time_ratio, fill

    def forward(self, x):  # (B, C, F, T)
        if not self.training or (self.fm == 0 and self.tm == 0):
            return x
        B, _, Fq, T = x.shape
        tw = max(1, min(self.tw, int(T * self.ratio)))
        keep = torch.ones(B, 1, Fq, T, dtype=torch.bool, device=x.device)
        f_idx, t_idx = torch.arange(Fq, device=x.device)[None], torch.arange(T, device=x.device)[None]
        for _ in range(self.fm):
            w = torch.randint(0, self.fw + 1, (B, 1), device=x.device)
            f0 = (torch.rand(B, 1, device=x.device) * (Fq - w + 1)).long()
            keep &= ~((f_idx >= f0) & (f_idx < f0 + w))[:, None, :, None]
        for _ in range(self.tm):
            w = torch.randint(0, tw + 1, (B, 1), device=x.device)
            t0 = (torch.rand(B, 1, device=x.device) * (T - w + 1)).long()
            keep &= ~((t_idx >= t0) & (t_idx < t0 + w))[:, None, None, :]
        fill = x.amin(dim=(1, 2, 3), keepdim=True) if self.fill == 'min' else torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.where(keep, x, fill)
