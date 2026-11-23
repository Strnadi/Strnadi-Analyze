"""Saving / loading models, including checkpoints written by the research session's `train_cv.py`.

New format:      dict(state, model, model_kw, classes, epoch, config)
Research format: dict(state, args, classes, epoch)  with `args` = the argparse namespace of `train_cv.py`.
"""
import glob
import os
import torch
from config import CLASSES
from model import build_model


def save_model(path, state, cfg, epoch, **extra):
    torch.save(dict(state=state, model=cfg['model'], model_kw=cfg['model_kw'], classes=CLASSES, epoch=epoch, config=cfg, **extra), path)


def _research_model(args):
    """Rebuild the model a research-session checkpoint was trained with."""
    arch = args.get('arch', 'strnad')
    fe = dict(bank=args.get('scale', 'mel'), norm=args.get('norm', 'median'), n_bins=args.get('n_bins', 128), fmin=args.get('fmin', 1500.0),
              fmax=args.get('fmax', 11000.0), hop=args.get('hop', 320))
    head = 'factorized' if args.get('factorized') else ('cosine' if args.get('cosine_head') else 'linear')
    hk = dict(k=args.get('subcenters', 1), s=args.get('margin_scale', 16.0)) if head == 'cosine' else None
    if arch == 'strnad':
        return build_model('strnadnet', frontend=fe, widths=tuple(args.get('widths', (16, 32, 64, 128))), depths=tuple(args.get('depths', (1, 1, 2, 1))),
                           d=args.get('d', 160), n_temporal=args.get('n_temporal', 3), dw=args.get('dw', False), drop=args.get('drop', 0.2),
                           stem_stride=tuple(args.get('stem_stride', (2, 2))), time_strides=tuple(args.get('time_strides', (1, 2, 1, 1))),
                           temporal=args.get('temporal', 'conv'), head=head, head_kw=hk)
    if arch.startswith('timm:'):
        return build_model('timm_temporal', backbone=arch[5:], pretrained=False, frontend=fe, temporal=args.get('temporal', 'conv'),
                           drop=args.get('drop', 0.2), head=head, head_kw=hk)
    if arch == 'patchtf':
        return build_model('patch_transformer', frontend=fe, dim=args.get('tf_dim', 192), depth=args.get('tf_depth', 6), head=head, head_kw=hk)
    if arch == 'contour':
        return build_model('contour_tcn', frontend=fe, ch=args.get('contour_ch', 96), n_blocks=args.get('contour_blocks', 6), head=head, head_kw=hk)
    raise ValueError(arch)


def load_research_state(model, state):
    """Map research parameter names onto the refactored modules (same tensors, new module layout)."""
    own = model.state_dict(); out = {}
    for k, v in state.items():
        if k.startswith('frontend.'):
            continue                                              # fixed buffers, rebuilt deterministically
        for old, new in (('att.', 'pool.att.'), ('head.att.', 'head.pool.att.'), ('cls.weight', 'cls.lin.weight'), ('cls.bias', 'cls.lin.bias'),
                         ('head.cls.weight', 'head.cls.lin.weight'), ('head.cls.bias', 'head.cls.lin.bias')):
            if k.startswith(old) and (new + k[len(old):]) in own and k not in own:
                k = new + k[len(old):]; break
        out[k] = v
    missing, unexpected = model.load_state_dict(out, strict=False)
    missing = [m for m in missing if not m.startswith('frontend.')]
    if missing or unexpected:
        raise RuntimeError(f'research checkpoint does not fit: missing {missing[:5]} unexpected {list(unexpected)[:5]}')
    return model


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in ck:
        kw = dict(ck['model_kw'])
        if 'pretrained' in kw:
            kw['pretrained'] = False
        m = build_model(ck['model'], n_classes=len(ck['classes']), **kw); m.load_state_dict(ck['state'])
    else:
        m = load_research_state(_research_model(ck['args']), ck['state'])
    return m.eval()


def load_encoder_state(model, state):
    """Initialise a model from a pretrained encoder (SSL / distilled), old or new parameter names; heads are skipped."""
    own = model.state_dict(); out = {}
    for k, v in state.items():
        if k.startswith(('cls.', 'head.cls.', 'frontend.')):
            continue
        if k.startswith('att.') and k not in own:
            k = 'pool.' + k
        if k in own and own[k].shape == v.shape:
            out[k] = v
    missing = [k for k in own if k not in out and not k.startswith(('cls.', 'head.cls.', 'frontend.')) and 'num_batches_tracked' not in k]
    if missing:
        print(f'load_encoder_state: {len(missing)} encoder tensors not initialised, e.g. {missing[:4]}', flush=True)
    model.load_state_dict(out, strict=False)
    return model


class LogitsModel(torch.nn.Module):
    """Tensor-returning view of a model, for evaluation scripts written against the research API:
    `m(x)` -> logits, `m(x, return_all=True)` -> the output dict (with the old key names `frame` and `h` added)."""

    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, x, return_all=False):
        out = self.model(x)
        if not return_all:
            return out['logits']
        out = dict(out); out.setdefault('frame', out.get('frame_logits')); out.setdefault('h', out.get('frames'))
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('model'), name)


def load_logits_model(path):
    return LogitsModel(load_model(path)).eval()


def run_checkpoints(spec, name='model.pt'):
    """A run directory -> its fold checkpoints; a file -> itself."""
    return sorted(glob.glob(os.path.join(spec, 'fold*', name))) if os.path.isdir(spec) else [spec]


def export_mode(model, on=True):
    """Switch every STFT to the conv1d path (standard ops only) before ONNX export."""
    from layers.stft_power import StftPower
    for m in model.modules():
        if isinstance(m, StftPower):
            m.use_fft = not on
    return model
