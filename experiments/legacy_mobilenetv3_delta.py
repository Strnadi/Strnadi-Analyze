EXPERIMENT = dict(
    model='mobilenetv3_delta', model_kw=dict(pretrained=True), lr=1e-4, wd=0.01, epochs=30, bs=32,
    spec_aug=dict(freq_masks=2, freq_width=27, time_masks=2, time_width=80), speed=0.0, label_smoothing=0.0, balance=1.0,
    select='inner_argmax', eval_crop='start',
    description='Legacy low-latency variant, ported: mel + delta + delta-delta channels -> ImageNet MobileNetV3-Small -> patch '
                'attention pooling -> 128 -> classes.',
    hypothesis='An edge-friendly backbone for the phone at the production recipe\'s accuracy.',
    origin='legacy-repo', status='proposed', result='No stored result in the legacy repo. Not re-trained.',
)
