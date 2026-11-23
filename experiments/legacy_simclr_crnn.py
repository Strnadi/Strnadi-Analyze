EXPERIMENT = dict(
    runner='unsupervised', args='--name {name} --model densenet_crnn_embed --objective nt_xent --threads {threads} --steps 6000 --bs 16',
    description='Legacy "embed experiment", ported: SimCLR (NT-Xent) on a DenseNet121 -> BiGRU -> attentive pooling -> 128-d '
                'L2-normalised embedding.',
    hypothesis='A contrastive embedding separates dialects without labels (for clustering / the dialect map).',
    origin='legacy-repo', status='proposed',
    result='No stored result. Corrected in the port: the Keras script passed the same waveform twice through the model, so the '
           'two views differed only by SpecAugment and dropout; the trainer now builds two waveform-level augmented views. '
           '12.7 M params: expensive on CPU; the research session\'s same-song contrastive run (ssl_same_song.py) is the cheaper '
           'equivalent.',
)
