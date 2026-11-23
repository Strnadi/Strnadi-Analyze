EXPERIMENT = dict(
    model='densenet121_attnpool', model_kw=dict(pretrained=True), lr=1e-4, backbone_lr_mult=1.0, wd=0.01, epochs=30, bs=32,
    spec_aug=dict(freq_masks=2, freq_width=27, time_masks=2, time_width=80), speed=0.0, label_smoothing=0.0, balance=1.0,
    select='inner_argmax', eval_crop='start',
    description='The production recipe (good-model-2 lineage) as it was, ported to PyTorch: magma mel -> ImageNet DenseNet121 -> '
                'patch attention pooling -> MLP; AdamW 1e-4, wide SpecAugment, inverse-frequency class balance, strong labels only.',
    hypothesis='Reproduce the deployed model under a leak-free protocol, as the reference the new models are compared with.',
    origin='legacy-repo', status='proposed',
    result='The Keras original reported ~82.5 % accuracy / 64.5 % macro recall on a random by-file split (optimistic: clips of one '
           'recording and pre-baked augmented copies crossed the split). The deployed model on held-out Vizina males: BhBl recall '
           '0.55 over 58 males, ~45 % of complete BhBl songs called Unfinished. Not yet re-trained in this form. '
           'Note the 27-bin / 0.8 s masks can erase a whole terminal element; kept only to reproduce the original.',
)
