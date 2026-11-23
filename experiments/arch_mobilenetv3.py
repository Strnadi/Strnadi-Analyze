from .recipes import R7

EXPERIMENT = dict(
    R7, model='timm_temporal', model_kw=dict(backbone='mobilenetv3_large_100', head='factorized'), backbone_lr_mult=0.1, folds=[0, 1, 2],
    description='Architecture comparison: ImageNet MobileNetV3-Large (5.0 M) + shared temporal head on the R7 recipe.',
    hypothesis='MBConv backbone as the phone candidate.',
    origin='research-session', status='queued', result='Not finished. Backbone learning rate corrected to 0.1x (see arch_convnext_atto).',
)
