# %% [markdown]
# # legacy_conv_autoencoder
#
# Legacy convolutional bottleneck autoencoder (asymmetric pooling + spatial attention, 64-d
# latent), ported; the latent is clustered afterwards (HDBSCAN / GMM).
#
# **Question.** Unsupervised dialect clusters emerge in the latent space of a reconstruction
# model.
#
# **Status:** proposed (from the original Keras repository). No stored result. Expectation from
# the research session: reconstruction mostly encodes background and loudness.
#
# Run from the repository root with `python -m experiments.legacy_conv_autoencoder`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/ssl/legacy_conv_autoencoder/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.conv_autoencoder import ConvAutoencoder
from training.unsupervised import fit_unsupervised

# %% Settings
RUN = 'legacy_conv_autoencoder'


# %% Model
def make_model():
    return ConvAutoencoder()


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
