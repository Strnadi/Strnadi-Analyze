# %% [markdown]
# # ssl_masked_autoencoder
#
# ssl_mae: masked autoencoder on the StrnadNet encoder (hide 40 % of time spans, reconstruct the
# spectrogram).
#
# **Question.** Masked reconstruction is SOTA in bioacoustics when fine-tuned (Bird-MAE); does a
# CNN-scale version help here?
#
# **Status:** queued (from the research session). Not run. Low priority per both reviews.
#
# Run from the repository root with `python -m experiments.ssl_masked_autoencoder`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/ssl/ssl_masked_autoencoder/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.strnadnet import StrnadNet
from training.pretrain import pretrain_encoder

# %% Settings
RUN = 'ssl_masked_autoencoder'

OBJECTIVES = dict(masked_reconstruction=1.0)


# %% Model: the encoder that is pretrained (its classifier head is not used)
def make_model():
    return StrnadNet()


# %% Pretrain without labels
pretrain_encoder(
    RUN,
    make_model,
    objectives=OBJECTIVES,
    steps=12000,
    batch_size=48,
    sources=('vizina', 'audiomoth_bhbl', 'birdnet_yh'),
)

# %% Evaluate: how much dialect information does the frozen embedding carry?
probe_checkpoint(f'ssl/{RUN}/encoder.pt')

# To use the encoder, fine-tune a classifier from it: copy an experiment such as
# `r2_factorized_weak.py` and add `init_from=f'ssl/{RUN}/encoder.pt'` to its settings.
