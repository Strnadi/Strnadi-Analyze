EXPERIMENT = dict(
    runner='ssl', args='--name {name} --threads {threads} --workers 3 --steps 8000 --bs 48 --p_other_song 0 --w_supcon 0 --w_mask 0 '
                       '--w_ae 1 --sources vizina,audiomoth_bhbl,birdnet_yh',
    description='ssl_ae: bottleneck autoencoder: StrnadNet 128-d clip embedding -> decoder reconstructs the pooled spectrogram.',
    hypothesis='A true information bottleneck gives a compact space for clustering / the "draw a circle" map.',
    origin='research-session', status='queued',
    result='Not run. Known risk: reconstruction spends capacity on background and loudness, not on the few-hundred-Hz details '
           'that define a dialect. The legacy repo\'s autoencoders are in legacy_conv_autoencoder.py / legacy_mobile_autoencoder.py.',
)
