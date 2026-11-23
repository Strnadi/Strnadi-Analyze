EXPERIMENT = dict(
    runner='unsupervised', args='--name {name} --model mobile_autoencoder --objective reconstruction --threads {threads} --steps 6000 --bs 32',
    description='Legacy MobileNetV2-style bottleneck autoencoder (MBConv encoder, spatial attention, 64-d latent), ported.',
    hypothesis='Same as legacy_conv_autoencoder with a mobile-size encoder.',
    origin='legacy-repo', status='proposed', result='No stored result.',
)
