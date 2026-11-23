from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='efficientat_temporal', model_kw=dict(pretrained_name='mn10_as', head='factorized'), lr=1e-3, backbone_lr_mult=0.1,
    description='EfficientAT MobileNetV3 mn10 (4.9 M, AudioSet-pretrained, distilled from PaSST) + shared temporal head.',
    hypothesis='An audio-pretrained mobile CNN beats ImageNet initialisation; candidate phone model and cheap second teacher.',
    origin='review-base', status='proposed',
    result='Not run. Recommended by the research session\'s own literature report; its sweep used ImageNet weights only. '
           'mn04_as (0.98 M) is the small variant.',
    needs=CLEAN_NEEDS + '; EFFICIENTAT_DIR = clone of github.com/fschmid56/EfficientAT (model wrapper untested, see its docstring)',
)
