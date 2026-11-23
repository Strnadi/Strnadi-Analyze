from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='perch_temporal', model_kw=dict(precomputed=False, trainable=True, head='factorized'), win=5.0,
    lr=3e-4, backbone_lr_mult=0.1, epochs=12, bs=16, epoch_size=4096, threads=10,
    description='Perch 2.0 (EfficientNet-B3, 12 M) fine-tuned end to end on 5 s windows with the weak-label recipe: the teacher.',
    hypothesis='A fine-tuned bird foundation model is the accuracy ceiling and the right teacher for the small model; frozen, it '
               'already equals the from-scratch baseline.',
    origin='review-base', status='proposed',
    result='Not run. The research session planned it and never queued it. Rough cost estimate from the base review: 2-3 h per '
           'fold on 10 cores.',
    needs=CLEAN_NEEDS + '; network access for torch.hub (janclemenslab/perch2_torch); run `python -m tests.check_perch` first.',
)
