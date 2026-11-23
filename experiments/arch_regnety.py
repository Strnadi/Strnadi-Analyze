from .recipes import R7

EXPERIMENT = dict(
    R7, model='timm_temporal', model_kw=dict(backbone='regnety_004', head='factorized'), backbone_lr_mult=0.1, folds=[0, 1, 2],
    description='Architecture comparison: ImageNet RegNetY-004 (5.1 M) + shared temporal head on the R7 recipe.',
    hypothesis='Another mobile-class CNN family, for the size / accuracy curve.',
    origin='research-session', status='queued', result='Not finished. Backbone learning rate corrected to 0.1x.',
)
