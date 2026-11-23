from .recipes import R7

EXPERIMENT = dict(
    R7, model='timm_temporal', model_kw=dict(backbone='efficientnet_b0', head='factorized'), backbone_lr_mult=0.1, folds=[0, 1, 2],
    description='Architecture comparison: ImageNet EfficientNet-B0 (4.6 M) + shared temporal head on the R7 recipe.',
    hypothesis='The backbone family behind Perch and the BirdCLEF students.',
    origin='research-session', status='queued', result='Not finished. Backbone learning rate corrected to 0.1x.',
)
