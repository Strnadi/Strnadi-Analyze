"""AdamW with a separate learning rate for pretrained backbones, linear warm-up and cosine decay.

Pretrained backbones (ImageNet, AudioSet, Perch) are trained at `learning_rate * backbone_lr_factor`;
a model says which of its parameters are backbone through `backbone_prefixes`. The research
architecture sweep trained pretrained ConvNeXt / DenseNet with the from-scratch learning rate, so
its ranking mostly reflected hyper-parameters (base-session review).
"""

import math

import torch


def build_optimizer(model, extra_parameters, settings, total_steps):
    """Create the optimiser and its per-step learning-rate schedule.

    Args:
        model: the model being trained.
        extra_parameters: parameters outside the model (e.g. a distillation projection).
        settings: `TrainSettings`.
        total_steps: number of optimiser steps in the whole run.
    """
    backbone_prefixes = tuple(getattr(model, 'backbone_prefixes', ()))

    backbone_parameters = []
    other_parameters = list(extra_parameters)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if backbone_prefixes and name.startswith(backbone_prefixes):
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    groups = [{'params': other_parameters, 'lr': settings.learning_rate}]
    if backbone_parameters:
        backbone_lr = settings.learning_rate * settings.backbone_lr_factor
        groups.append({'params': backbone_parameters, 'lr': backbone_lr})

    optimizer = torch.optim.AdamW(
        groups, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    warmup_steps = max(1, total_steps // 20)

    def lr_factor(step):
        warmup = min(1.0, (step + 1) / warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
        return warmup * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return optimizer, scheduler
