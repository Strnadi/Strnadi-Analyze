# %% [markdown]
# # backbone_regnety_imagenet
#
# Architecture comparison: ImageNet RegNetY-004 (5.1 M) + shared temporal head on the R7 recipe.
#
# **Question.** Another mobile-class CNN family, for the size / accuracy curve.
#
# **Status:** queued (from the research session). Not finished. Backbone learning rate corrected
# to 0.1x.
#
# Run from the repository root with `python -m experiments.backbone_regnety_imagenet`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/backbone_regnety_imagenet/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.timm_temporal import TimmTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'backbone_regnety_imagenet'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_audiomoth.npy',
    eval_window='start',
    # protocol
    folds=[0, 1, 2],
    checkpoint_rule='inner_argmax',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    backbone_lr_factor=0.1,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return TimmTemporal(backbone='regnety_004', head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
