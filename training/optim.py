"""AdamW with a separate (lower) learning rate for pretrained backbones, linear warm-up and cosine decay.

`backbone_lr_mult`: pretrained backbones (ImageNet / AudioSet / Perch) are trained at lr * mult; the model
declares which parameters those are through `backbone_prefixes`. The research architecture sweep trained
pretrained ConvNeXt / DenseNet with the from-scratch settings (lr 2e-3 for everything), so its ranking mostly
reflected hyper-parameters (base-session review).
"""
import math
import torch


def build_optimizer(model, extra_params, cfg, steps):
    prefixes = tuple(getattr(model, 'backbone_prefixes', ()))
    back = [p for n, p in model.named_parameters() if p.requires_grad and prefixes and n.startswith(prefixes)]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not (prefixes and n.startswith(prefixes))]
    groups = [{'params': rest + list(extra_params), 'lr': cfg['lr']}]
    if back:
        groups.append({'params': back, 'lr': cfg['lr'] * cfg['backbone_lr_mult']})
    opt = torch.optim.AdamW(groups, lr=cfg['lr'], weight_decay=cfg['wd'])
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    return opt, sched
