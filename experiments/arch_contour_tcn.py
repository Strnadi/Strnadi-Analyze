from .recipes import R7

EXPERIMENT = dict(
    R7, model='contour_tcn', model_kw=dict(head='factorized'), folds=[0, 1, 2],
    description='Architecture comparison: 1-D TCN over top-3 pitch-contour features (0.27 M params, ~5x cheaper than StrnadNet).',
    hypothesis='The "ornithologist\'s view" (pitch tracks only) is enough, at a fraction of the cost.',
    origin='research-session', status='queued', result='Not finished. See contour_ridge.py for the sub-bin variant.',
)
