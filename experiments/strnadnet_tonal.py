from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='strnadnet_tonal', model_kw={},
    description='The combined "faint ending" model: multi-resolution input + long-kernel stem + percentile floor + factorised head.',
    hypothesis='The front-end and stem changes add up (run the single ablations first: frontend_multires, '
               'frontend_percentile_floor, stem_tonal_kernels).',
    origin='review-fork', status='proposed', result='Not run.', needs=CLEAN_NEEDS,
)
