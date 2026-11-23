# %% [markdown]
# # ssl_same_song
#
# ssl_v2: label-free pretraining of StrnadNet: same-song contrastive (two noisy / re-equalised
# views) + data2vec-style masked-frame prediction, on ~100k unlabelled songs (no NCS audio, no
# test males).
#
# **Question.** Label-free pretraining beats training from scratch once fine-tuned with the R2
# recipe (--init).
#
# **Status:** running (from the research session). Not finished. Both reviews and the research
# session's own literature report rate self-supervised learning from scratch on CPU as a poor bet
# next to distilling Perch 2.0 (experiments/perch_distill_cached.py).
#
# Run from the repository root with `python -m experiments.ssl_same_song`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/ssl/ssl_same_song/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.strnadnet import StrnadNet
from training.pretrain import pretrain_encoder

# %% Settings
RUN = 'ssl_same_song'

OBJECTIVES = dict(same_song=1.0, masked_frames=1.0)


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
