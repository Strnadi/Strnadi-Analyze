from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='bc_resnet', model_kw=dict(tau=2.0, head='factorized'),
    description='BC-ResNet (tau = 2, ~0.15 M params): the published tiny-model family from keyword spotting / DCASE low-complexity.',
    hypothesis='How small can the phone model get? StrnadNet is a home-made design of the same kind that was never compared '
               'against the proven family.',
    origin='review-fork', status='proposed', result='Not run.', needs=CLEAN_NEEDS,
)
