# %% [markdown]
# # ssl_bottleneck_autoencoder
#
# ssl_ae: bottleneck autoencoder: StrnadNet 128-d clip embedding -> decoder reconstructs the
# pooled spectrogram.
#
# **Question.** A true information bottleneck gives a compact space for clustering / the "draw a
# circle" map.
#
# **Status:** queued (from the research session). Not run. Known risk: reconstruction spends
# capacity on background and loudness, not on the few-hundred-Hz details that define a dialect.
# The legacy repo's autoencoders are in legacy_conv_autoencoder.py / legacy_mobile_autoencoder.py.
#
# Run from the repository root with `python -m experiments.ssl_bottleneck_autoencoder`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/ssl/ssl_bottleneck_autoencoder/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.strnadnet import StrnadNet
from training.pretrain import pretrain_encoder

# %% Settings
RUN = 'ssl_bottleneck_autoencoder'

OBJECTIVES = dict(bottleneck_reconstruction=1.0)


# %% Model: the encoder that is pretrained (its classifier head is not used)
def make_model():
    return StrnadNet()


# %% Pretrain without labels
pretrain_encoder(
    RUN,
    make_model,
    objectives=OBJECTIVES,
    steps=8000,
    batch_size=48,
    sources=('vizina', 'audiomoth_bhbl', 'birdnet_yh'),
)

# %% Evaluate: how much dialect information does the frozen embedding carry?
probe_checkpoint(f'ssl/{RUN}/encoder.pt')

# To use the encoder, fine-tune a classifier from it: copy an experiment such as
# `r2_factorized_weak.py` and add `init_from=f'ssl/{RUN}/encoder.pt'` to its settings.
