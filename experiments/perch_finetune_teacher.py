# %% [markdown]
# # perch_finetune_teacher
#
# Perch 2.0 (EfficientNet-B3, 12 M) fine-tuned end to end on 5 s windows with the weak-label
# recipe: the teacher.
#
# **Question.** A fine-tuned bird foundation model is the accuracy ceiling and the right teacher
# for the small model; frozen, it already equals the from-scratch baseline.
#
# **Status:** proposed (from review proposal). Not run. The research session planned it and never
# queued it. Rough cost estimate from the base review: 2-3 h per fold on 10 cores.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; network access for torch.hub (janclemenslab/perch2_torch); run `python -m
# tests.check_perch` first.
#
# Run from the repository root with `python -m experiments.perch_finetune_teacher`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/perch_finetune_teacher/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.perch_temporal import PerchTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'perch_finetune_teacher'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    window_seconds=5.0,
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=12,
    samples_per_epoch=4096,
    batch_size=16,
    learning_rate=0.0003,
    backbone_lr_factor=0.1,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    # runtime
    threads=10,
)


# %% Model
def make_model():
    return PerchTemporal(precomputed=False, trainable=True, head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
