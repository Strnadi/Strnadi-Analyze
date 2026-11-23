"""Exponential moving average (EMA) of the model weights.

The EMA copy is the one that is evaluated, saved and deployed: it cuts validation jitter roughly
three-fold in the BirdCLEF reports the research session cites.
"""

import copy

import torch


class EMA:
    def __init__(self, model, decay=0.998):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        self.step = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Move the averaged weights a small step towards the current weights of `model`."""
        self.step += 1
        # Early in training follow the model closely, then settle at the configured decay.
        decay = min(self.decay, (1 + self.step) / (10 + self.step))

        averaged_state = self.model.state_dict()
        current_state = model.state_dict()
        for name, averaged in averaged_state.items():
            current = current_state[name]
            if averaged.dtype.is_floating_point:
                averaged.mul_(decay).add_(current.detach(), alpha=1 - decay)
            else:
                averaged.copy_(current)  # integer buffers such as BatchNorm's step counter
