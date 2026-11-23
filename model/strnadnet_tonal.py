"""StrnadNet aimed at the faint terminal element: multi-resolution input + long-kernel stem, full frequency
resolution kept through the stem.

- input channels from 16 / 32 / 128 ms STFT windows (the long window gains SNR on steady thin tones);
- stem = 3x3 branch + (1 x 9) and (1 x 15) time-long branches (integrate along tonal lines);
- the stem strides in time only, so the first layer still sees every frequency bin;
- percentile noise floor instead of the median (the median erases part of long songs);
- factorised head by default.
Status: proposed (fork review, architecture ideas 1 + 2; base-session front-end finding). Untrained.
"""
from .strnadnet import StrnadNet


class TonalStrnadNet(StrnadNet):
    def __init__(self, n_classes=7, frontend=None, **kw):
        fe = dict(bank='mel', norm='percentile', n_fft=[512, 1024, 4096])
        fe.update(frontend or {})
        kw.setdefault('stem', 'tonal'); kw.setdefault('stem_stride', (1, 2)); kw.setdefault('time_strides', (1, 2, 1, 1))
        kw.setdefault('head', 'factorized')
        super().__init__(n_classes=n_classes, frontend=fe, **kw)
