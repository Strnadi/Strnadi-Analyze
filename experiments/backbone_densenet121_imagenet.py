# %% [markdown]
# # backbone_densenet121_imagenet
#
# Control: the production backbone (ImageNet DenseNet121, 8.2 M) with the new data, losses and
# temporal head.
#
# **Question.** Is the production model weak because of its data and training protocol, or because
# of the architecture?
#
# **Status:** rerun (from the research session). Re-targeted on 2026-11-24 to the audited clips and
# best-validation-loss checkpoints, 5 folds; results in docs/RESULTS.md. Not finished. ~5x the training cost per step of
# StrnadNet on CPU. See legacy_densenet121.py for the original design (patch pooling instead of a
# temporal head).
#
# Run from the repository root with `python -m experiments.backbone_densenet121_imagenet`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/backbone_densenet121_imagenet/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.timm_temporal import TimmTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'backbone_densenet121_imagenet'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
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
    return TimmTemporal(backbone='densenet121', head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
