from .recipes import R7

EXPERIMENT = dict(
    R7, model='timm_temporal', model_kw=dict(backbone='convnext_atto', head='factorized'), backbone_lr_mult=0.1, folds=[0, 1, 2],
    description='Architecture comparison: ImageNet ConvNeXt-atto (4.4 M) + shared temporal head on the R7 recipe.',
    hypothesis='Does a pretrained mobile-class CNN beat the from-scratch StrnadNet on the same data?',
    origin='research-session', status='queued',
    result='Not finished. Corrected here: the research job trained the pretrained backbone at the from-scratch learning rate '
           '(2e-3); this file uses 0.1x for the backbone. With 3 folds and one seed the sweep cannot resolve differences < ~0.03.',
)
