"""SpecAugment (Park et al. 2019): random frequency and time masks, independent per example,
training only.

Deliberately narrow defaults (2 x <= 8 of 128 bins, 2 x <= 0.1 s): a wider mask can erase a whole
terminal element and silently turn a dialect into "Unfinished".

Fixes over the legacy Keras layer (`layers/spec_augument.py`):
- masks were drawn once per batch (every example got the same mask);
- masked cells were set to 0.0 on a dB spectrogram, i.e. to the *loudest* possible value after the
  min-max colour mapping, painting bright bars instead of hiding energy. `fill='min'` masks to the
  example's minimum; `fill='zero'` is right for floor-normalised inputs (log-median / percentile /
  PCEN), where 0 is the noise floor.
"""

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks=2,
        freq_width=8,
        time_masks=2,
        time_width=10,
        max_time_ratio=0.2,
        fill='zero',
    ):
        super().__init__()
        self.freq_masks = freq_masks  # number of frequency masks per example
        self.freq_width = freq_width  # largest frequency mask, in bins
        self.time_masks = time_masks  # number of time masks per example
        self.time_width = time_width  # largest time mask, in frames
        self.max_time_ratio = max_time_ratio  # ... and as a fraction of the clip length
        self.fill = fill  # 'min' or 'zero'

    def forward(self, x):
        """(B, C, F, T) -> (B, C, F, T); returns x unchanged in eval mode."""
        if not self.training or (self.freq_masks == 0 and self.time_masks == 0):
            return x
        B, _, n_bins, T = x.shape
        device = x.device
        max_time_width = max(1, min(self.time_width, int(T * self.max_time_ratio)))

        keep = torch.ones(B, 1, n_bins, T, dtype=torch.bool, device=device)  # False = masked cell
        bin_index = torch.arange(n_bins, device=device)[None]  # (1, F)
        frame_index = torch.arange(T, device=device)[None]  # (1, T)

        for _ in range(self.freq_masks):
            # one mask per example: `width` bins starting at bin `start` (width 0 = no mask)
            width = torch.randint(0, self.freq_width + 1, (B, 1), device=device)  # (B, 1)
            start = (torch.rand(B, 1, device=device) * (n_bins - width + 1)).long()  # (B, 1)
            masked_bins = (bin_index >= start) & (bin_index < start + width)  # (B, F)
            keep &= ~masked_bins[:, None, :, None]  # (B, 1, F, 1), broadcast over time

        for _ in range(self.time_masks):
            # one mask per example: `width` frames starting at frame `start`
            width = torch.randint(0, max_time_width + 1, (B, 1), device=device)  # (B, 1)
            start = (torch.rand(B, 1, device=device) * (T - width + 1)).long()  # (B, 1)
            masked_frames = (frame_index >= start) & (frame_index < start + width)  # (B, T)
            keep &= ~masked_frames[:, None, None, :]  # (B, 1, 1, T), broadcast over frequency

        if self.fill == 'min':
            fill_value = x.amin(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1) minimum of each example
        else:
            fill_value = torch.zeros((), dtype=x.dtype, device=x.device)
        return torch.where(keep, x, fill_value)
