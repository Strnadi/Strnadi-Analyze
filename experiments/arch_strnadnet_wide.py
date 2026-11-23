from .recipes import R7

EXPERIMENT = dict(
    R7, model='strnadnet', model_kw=dict(widths=(32, 64, 128, 256), d=256, head='factorized'), folds=[0, 1, 2],
    description='Architecture comparison: StrnadNet twice as wide (3.5 M params) on the R7 recipe.',
    hypothesis='Is 0.9 M parameters holding accuracy back?',
    origin='research-session', status='queued',
    result='Not finished. Prior from the learning curves: capacity is not the limit (the 0.9 M model already memorises the '
           'labelled set and held-out F1 is flat from epoch 4-8).',
)
