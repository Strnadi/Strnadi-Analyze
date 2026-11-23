"""Exponential moving average of the weights; the EMA copy is the one evaluated, saved and deployed
(it cuts validation jitter roughly 3x in the BirdCLEF reports the research cites)."""
import copy
import torch


class EMA:
    def __init__(self, model, decay=0.998):
        self.model = copy.deepcopy(model).eval(); self.decay, self.step = decay, 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))        # warm-up: follow the model closely at first
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)
