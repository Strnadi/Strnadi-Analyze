EXPERIMENT = dict(
    runner='ssl', args='--name {name} --threads {threads} --workers 3 --steps 12000 --bs 48 --p_other_song 0 --w_supcon 0 --w_mask 0 '
                       '--w_mae 1 --sources vizina,audiomoth_bhbl,birdnet_yh',
    description='ssl_mae: masked autoencoder on the StrnadNet encoder (hide 40 % of time spans, reconstruct the spectrogram).',
    hypothesis='Masked reconstruction is SOTA in bioacoustics when fine-tuned (Bird-MAE); does a CNN-scale version help here?',
    origin='research-session', status='queued', result='Not run. Low priority per both reviews.',
)
