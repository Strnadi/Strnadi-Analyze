# %% [markdown]
# # identity_embedding
#
# id_v1: individual-identity embedding: StrnadNet + ArcFace over ring-coded Vizina males, scored
# by equal error rate and by synthetic two-bird diarisation.
#
# **Question.** A per-bird embedding lets a recording with n birds be split into each bird's songs
# (and gives per-bird dialects).
#
# **Status:** queued (from the research session). Not run. Audit note: male codes stop carrying
# over between years in 2024, so "different males" across that boundary may be the same birds.
#
# Run from the repository root with `python -m experiments.identity_embedding`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Results go to
# `$STRNADI_WS/results/identity_identity_embedding.json`.

# %% Imports
from model.strnadnet import StrnadNet
from training.identity import fit_identity

# %% Settings
RUN = 'identity_embedding'


# %% Model: StrnadNet with a 2-sub-centre cosine head over the training males
def make_model(n_males):
    return StrnadNet(n_classes=n_males, head='cosine', head_kw=dict(k=2, s=24.0))


# %% Train and evaluate
# Prints the verification equal error rate and the synthetic-diarisation scores on held-out males.
fit_identity(RUN, make_model, steps=4000)
