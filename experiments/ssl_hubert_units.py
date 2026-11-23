# %% [markdown]
# # ssl_hubert_units
#
# ssl_hub: HuBERT-style masked prediction of k-means "acoustic units" (the objective behind AVES);
# doubles as a tokeniser of song elements (evaluation/vq_tokens.py).
#
# **Question.** Learned units make song endings clusterable into dialects without labels.
#
# **Status:** queued (from the research session). Not run. Stage-0 units (k-means on log-mel
# frames) carry no dialect information (ARI ~0); on supervised features kNN dialect-F1 0.30 vs
# 0.22 chance. Bag-of-units discards element order; a sequence model over units is needed.
#
# Run from the repository root with `python -m experiments.ssl_hubert_units`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/ssl/ssl_hubert_units/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.strnadnet import StrnadNet
from training.pretrain import pretrain_encoder

# %% Settings
RUN = 'ssl_hubert_units'

OBJECTIVES = dict(masked_units=1.0)


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
    codebook='vq/cb_logmel_k100.npz',
)

# %% Evaluate: how much dialect information does the frozen embedding carry?
probe_checkpoint(f'ssl/{RUN}/encoder.pt')

# To use the encoder, fine-tune a classifier from it: copy an experiment such as
# `r2_factorized_weak.py` and add `init_from=f'ssl/{RUN}/encoder.pt'` to its settings.
