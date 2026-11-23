from .recipes import CLEAN, CLEAN_NEEDS, with_model_kw

EXPERIMENT = dict(
    with_model_kw(CLEAN, frontend=dict(bank='linear', norm='percentile', n_bins=256, fmin=2000.0, fmax=9680.0, n_fft=2048),
                  stem_stride=(2, 2)),
    description='Front-end ablation: linear frequency axis, 256 bins over 2-9.7 kHz (30 Hz per bin), 64 ms window.',
    hypothesis='Dialect criteria are 100-250 Hz bandwidths and 1000-2000 Hz offsets, measured by experts at 86 Hz resolution; '
               'the default mel axis has 60-70 Hz bins and the stem halves that at once.',
    origin='research-session', status='proposed',
    result='Not run (in the research experiment matrix and literature report, never queued).',
    needs=CLEAN_NEEDS,
)
