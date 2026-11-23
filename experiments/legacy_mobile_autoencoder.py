# %% [markdown]
# # legacy_mobile_autoencoder
#
# Legacy MobileNetV2-style bottleneck autoencoder (MBConv encoder, spatial attention, 64-d
# latent), ported.
#
# **Question.** Same as legacy_conv_autoencoder with a mobile-size encoder.
#
# **Status:** proposed (from the original Keras repository). No stored result.
#
# Run from the repository root with `python -m experiments.legacy_mobile_autoencoder`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/ssl/legacy_mobile_autoencoder/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.mobile_autoencoder import MobileAutoencoder
from training.unsupervised import fit_unsupervised

# %% Settings
RUN = 'legacy_mobile_autoencoder'


# %% Model
def make_model():
    return MobileAutoencoder()


# %% Train without labels
fit_unsupervised(
    RUN,
    make_model,
    objective='reconstruction',
    steps=6000,
    batch_size=32,
)

# %% Evaluate: how much dialect information does the frozen embedding carry?
probe_checkpoint(f'ssl/{RUN}/model.pt')
