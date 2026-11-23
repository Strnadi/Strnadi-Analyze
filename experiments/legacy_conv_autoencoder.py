EXPERIMENT = dict(
    runner='unsupervised', args='--name {name} --model conv_autoencoder --objective reconstruction --threads {threads} --steps 6000 --bs 32',
    description='Legacy convolutional bottleneck autoencoder (asymmetric pooling + spatial attention, 64-d latent), ported; '
                'the latent is clustered afterwards (HDBSCAN / GMM).',
    hypothesis='Unsupervised dialect clusters emerge in the latent space of a reconstruction model.',
    origin='legacy-repo', status='proposed',
    result='No stored result. Expectation from the research session: reconstruction mostly encodes background and loudness.',
)
