"""Saving and loading models.

A checkpoint written by this repository stores the model's class and constructor arguments next to
the weights, so it can be rebuilt without the experiment script that created it:

    {'state', 'model_class', 'model_args', 'classes', 'epoch', 'settings', ...}

Checkpoints written by the research session's `train_cv.py` ({'state', 'args', 'classes', 'epoch'},
with `args` the argparse namespace) are supported too.
"""

import dataclasses
import glob
import importlib
import inspect
import os

import torch

from config import CLASSES


# ------------------------------------------------------------------------------------------- save
def save_model(path, state, model, epoch, settings=None, **extra):
    """Write a checkpoint.

    Args:
        path: output file.
        state: the weights to store (usually the EMA weights), a `state_dict`.
        model: the model object; its class and constructor arguments are recorded.
        epoch: the epoch the weights come from.
        settings: optional `TrainSettings`, stored for reference.
        extra: any further entries to store.
    """
    checkpoint = {
        'state': state,
        'model_class': model.class_path,
        'model_args': model.init_args,
        'classes': CLASSES,
        'epoch': epoch,
        'settings': dataclasses.asdict(settings) if settings is not None else None,
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


# ------------------------------------------------------------------------------------------- load
def build_from_class_path(class_path, arguments):
    """Import `package.module.ClassName` and construct it, never downloading pretrained weights."""
    module_name, class_name = class_path.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_name), class_name)

    arguments = dict(arguments)
    if 'pretrained' in inspect.signature(model_class.__init__).parameters:
        arguments['pretrained'] = False  # the checkpoint holds the weights
    return model_class(**arguments)


def load_model(path):
    """Load a checkpoint (new or research format) and return the model in eval mode."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    if 'model_class' in checkpoint:
        model = build_from_class_path(checkpoint['model_class'], checkpoint['model_args'])
        model.load_state_dict(checkpoint['state'])
    else:
        model = build_research_model(checkpoint['args'])
        load_research_state(model, checkpoint['state'])
    return model.eval()


def run_checkpoints(run_or_file, name='model.pt'):
    """A run directory -> its fold checkpoints; a checkpoint file -> itself."""
    if os.path.isdir(run_or_file):
        return sorted(glob.glob(os.path.join(run_or_file, 'fold*', name)))
    return [run_or_file]


# ------------------------------------------------------------------- research-session checkpoints
def build_research_model(args):
    """Rebuild the model a research-session checkpoint was trained with, from its argparse values."""
    from model.contour_tcn import ContourTCN
    from model.patch_transformer import PatchTransformer
    from model.strnadnet import StrnadNet
    from model.timm_temporal import TimmTemporal

    frontend = dict(
        bank=args.get('scale', 'mel'),
        norm=args.get('norm', 'median'),
        n_bins=args.get('n_bins', 128),
        fmin=args.get('fmin', 1500.0),
        fmax=args.get('fmax', 11000.0),
        hop=args.get('hop', 320),
    )

    if args.get('factorized'):
        head, head_kw = 'factorized', None
    elif args.get('cosine_head'):
        head = 'cosine'
        head_kw = dict(k=args.get('subcenters', 1), s=args.get('margin_scale', 16.0))
    else:
        head, head_kw = 'linear', None

    architecture = args.get('arch', 'strnad')
    if architecture == 'strnad':
        return StrnadNet(
            frontend=frontend,
            widths=tuple(args.get('widths', (16, 32, 64, 128))),
            depths=tuple(args.get('depths', (1, 1, 2, 1))),
            d=args.get('d', 160),
            n_temporal=args.get('n_temporal', 3),
            dw=args.get('dw', False),
            drop=args.get('drop', 0.2),
            stem_stride=tuple(args.get('stem_stride', (2, 2))),
            time_strides=tuple(args.get('time_strides', (1, 2, 1, 1))),
            temporal=args.get('temporal', 'conv'),
            head=head,
            head_kw=head_kw,
        )
    if architecture.startswith('timm:'):
        return TimmTemporal(
            backbone=architecture[len('timm:') :],
            pretrained=False,
            frontend=frontend,
            temporal=args.get('temporal', 'conv'),
            drop=args.get('drop', 0.2),
            head=head,
            head_kw=head_kw,
        )
    if architecture == 'patchtf':
        return PatchTransformer(
            frontend=frontend,
            dim=args.get('tf_dim', 192),
            depth=args.get('tf_depth', 6),
            head=head,
            head_kw=head_kw,
        )
    if architecture == 'contour':
        return ContourTCN(
            frontend=frontend,
            ch=args.get('contour_ch', 96),
            n_blocks=args.get('contour_blocks', 6),
            head=head,
            head_kw=head_kw,
        )
    raise ValueError(f'unknown research architecture: {architecture}')


#: Parameter-name prefixes that changed when the research model was split into layers.
RESEARCH_RENAMES = [
    ('att.', 'pool.att.'),
    ('head.att.', 'head.pool.att.'),
    ('cls.weight', 'cls.lin.weight'),
    ('cls.bias', 'cls.lin.bias'),
    ('head.cls.weight', 'head.cls.lin.weight'),
    ('head.cls.bias', 'head.cls.lin.bias'),
]


def load_research_state(model, state):
    """Load research-session weights into a refactored model (same tensors, new module layout)."""
    own_names = model.state_dict()
    renamed_state = {}

    for name, tensor in state.items():
        if name.startswith('frontend.'):
            continue  # fixed front-end buffers are rebuilt deterministically
        if name not in own_names:
            for old_prefix, new_prefix in RESEARCH_RENAMES:
                candidate = new_prefix + name[len(old_prefix) :]
                if name.startswith(old_prefix) and candidate in own_names:
                    name = candidate
                    break
        renamed_state[name] = tensor

    missing, unexpected = model.load_state_dict(renamed_state, strict=False)
    missing = [name for name in missing if not name.startswith('frontend.')]
    if missing or unexpected:
        raise RuntimeError(
            f'research checkpoint does not fit the model: '
            f'missing {missing[:5]}, unexpected {list(unexpected)[:5]}'
        )
    return model


# --------------------------------------------------------------------------- pretrained encoders
HEAD_PREFIXES = ('cls.', 'head.cls.', 'frontend.')


def load_encoder_state(model, state):
    """Initialise a model's encoder from a pretrained one (self-supervised or distilled).

    Classifier heads and front-end buffers are skipped; old and new parameter names are accepted.
    Tensors that do not fit are left at their initial values and reported.
    """
    own_state = model.state_dict()
    usable = {}

    for name, tensor in state.items():
        if name.startswith(HEAD_PREFIXES):
            continue
        if name.startswith('att.') and name not in own_state:
            name = 'pool.' + name
        if name in own_state and own_state[name].shape == tensor.shape:
            usable[name] = tensor

    not_initialised = [
        name
        for name in own_state
        if name not in usable
        and not name.startswith(HEAD_PREFIXES)
        and 'num_batches_tracked' not in name
    ]
    if not_initialised:
        print(
            f'load_encoder_state: {len(not_initialised)} encoder tensors not initialised, '
            f'e.g. {not_initialised[:4]}',
            flush=True,
        )

    model.load_state_dict(usable, strict=False)
    return model


# ------------------------------------------------------------------------ helpers for old scripts
class LogitsModel(torch.nn.Module):
    """A view of a model that returns the logits tensor instead of the output dict.

    The evaluation scripts taken over from the research session were written against that API:
    `m(x)` -> logits, `m(x, return_all=True)` -> the dict (with the old key names `frame`, `h`).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, return_all=False):
        outputs = self.model(x)
        if not return_all:
            return outputs['logits']
        outputs = dict(outputs)
        outputs.setdefault('frame', outputs.get('frame_logits'))
        outputs.setdefault('h', outputs.get('frames'))
        return outputs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)


def load_logits_model(path):
    return LogitsModel(load_model(path)).eval()


def export_mode(model, on=True):
    """Switch every STFT to its conv1d path (standard ops only) before ONNX export."""
    from layers.stft_power import StftPower

    for module in model.modules():
        if isinstance(module, StftPower):
            module.use_fft = not on
    return model
