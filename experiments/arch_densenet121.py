from .recipes import R7

EXPERIMENT = dict(
    R7, model='timm_temporal', model_kw=dict(backbone='densenet121', head='factorized'), backbone_lr_mult=0.1, folds=[0, 1, 2],
    description='Control: the production backbone (ImageNet DenseNet121, 8.2 M) with the new data, losses and temporal head.',
    hypothesis='Is the production model weak because of its data and training protocol, or because of the architecture?',
    origin='research-session', status='queued',
    result='Not finished. ~5x the training cost per step of StrnadNet on CPU. See legacy_densenet121.py for the original design '
           '(patch pooling instead of a temporal head).',
)
