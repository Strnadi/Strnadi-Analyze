from .recipes import R7

EXPERIMENT = dict(
    R7, model='patch_transformer', model_kw=dict(head='factorized'), folds=[0, 1, 2],
    description='Architecture comparison: small patch transformer (3.1 M) from scratch on the R7 recipe.',
    hypothesis='Can attention over patches beat convolution here?',
    origin='research-session', status='queued',
    result='Not finished. Both reviews expect it to lose: 3,744 labelled clips are too few for a transformer from scratch. '
           'Low priority.',
)
