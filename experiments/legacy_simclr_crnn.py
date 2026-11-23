# %% [markdown]
# # legacy_simclr_crnn
#
# Legacy "embed experiment", ported: SimCLR (NT-Xent) on a DenseNet121 -> BiGRU -> attentive
# pooling -> 128-d L2-normalised embedding.
#
# **Question.** A contrastive embedding separates dialects without labels (for clustering / the
# dialect map).
#
# **Status:** proposed (from the original Keras repository). No stored result. Corrected in the
# port: the Keras script passed the same waveform twice through the model, so the two views
# differed only by SpecAugment and dropout; the trainer now builds two waveform-level augmented
# views. 12.7 M params: expensive on CPU; the research session's same-song contrastive run
# (ssl_same_song.py) is the cheaper equivalent.
#
# Run from the repository root with `python -m experiments.legacy_simclr_crnn`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/ssl/legacy_simclr_crnn/`.

# %% Imports
from evaluation.report import probe_checkpoint
from model.densenet_crnn_embed import DenseNetCRNNEmbed
from training.unsupervised import fit_unsupervised

# %% Settings
RUN = 'legacy_simclr_crnn'


# %% Model
def make_model():
    return DenseNetCRNNEmbed(pretrained=True)


# %% Train without labels
fit_unsupervised(
    RUN,
    make_model,
    objective='contrastive',
    steps=6000,
    batch_size=16,
)

# %% Evaluate: how much dialect information does the frozen embedding carry?
probe_checkpoint(f'ssl/{RUN}/model.pt')
