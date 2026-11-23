from .recipes import CLEAN, CLEAN_NEEDS, with_model_kw

EXPERIMENT = dict(
    with_model_kw(CLEAN, frontend=dict(norm='percentile')),
    description='Front-end ablation: 15th-percentile noise floor instead of the per-frequency median.',
    hypothesis='The median contains song energy when the song fills the window: 36-64 % of dialect clips lose > 6 dB in some '
               'bin, worst for BlBh. A low percentile stays on the noise.',
    origin='review-base', status='proposed', result='Not run. No front-end variant was ever queued in the research session.',
    needs=CLEAN_NEEDS,
)
