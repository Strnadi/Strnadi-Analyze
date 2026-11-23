"""StrnadNet aimed at the faint terminal element: multi-resolution input + long-kernel stem, full
frequency resolution kept through the stem.

- input channels from 16 / 32 / 128 ms STFT windows (the long window gains SNR on steady thin
  tones);
- stem = 3x3 branch + (1 x 9) and (1 x 15) time-long branches (integrate along tonal lines);
- the stem strides in time only, so the first layer still sees every frequency bin;
- percentile noise floor instead of the median (the median erases part of long songs);
- factorised head by default.

Status: proposed (fork review, architecture ideas 1 + 2; base-session front-end finding). Untrained.
"""

from .strnadnet import StrnadNet


class TonalStrnadNet(StrnadNet):
    """`StrnadNet` with different defaults; everything else (layers, outputs) is inherited.

    Args:
        n_classes: number of output classes.
        frontend: dict of front-end settings (see `layers.registry.build_frontend`) that override
            this model's own defaults: mel bank, percentile normalisation and three STFT windows
            (512, 1024 and 4096 samples) stacked as three input channels.
        **kw: any other `StrnadNet` argument. Four of them get a different default here unless
            passed explicitly: `stem='tonal'`, `stem_stride=(1, 2)`, `time_strides=(1, 2, 1, 1)`
            and `head='factorized'`.
    """

    def __init__(self, n_classes=7, frontend=None, **kw):
        # --- front-end: three window lengths as channels, percentile noise floor
        frontend_config = dict(bank='mel', norm='percentile', n_fft=[512, 1024, 4096])
        frontend_config.update(frontend or {})

        # --- stem: long time kernels, striding in time only (every frequency bin is still seen)
        kw.setdefault('stem', 'tonal')
        kw.setdefault('stem_stride', (1, 2))
        kw.setdefault('time_strides', (1, 2, 1, 1))

        # --- classifier head: song / complete / dialect factors
        kw.setdefault('head', 'factorized')

        # --- everything else is the plain StrnadNet
        super().__init__(n_classes=n_classes, frontend=frontend_config, **kw)


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(TonalStrnadNet())
