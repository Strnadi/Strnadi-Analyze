# %% [markdown]
# # distill_function_matching
#
# The final small model: StrnadNet trained to match a fine-tuned teacher ensemble (soft labels +
# embeddings) on the SAME augmented views, over a long schedule.
#
# **Question.** "Function matching" (Beyer et al. 2022): consistent teaching on identical views
# with long schedules is what makes distillation work; EfficientAT gained +5.3 mAP this way.
#
# **Status:** proposed (from review proposal). Not run. Extend to the ~100k unlabelled songs by
# pseudo-labelling them with the teacher (evaluation/pseudo_label.py) and adding the table to
# `extra`: pseudo-labels only from a strong teacher.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; a finished teacher run (experiments/perch_finetune_teacher.py)
#
# Run from the repository root with `python -m experiments.distill_function_matching`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/distill_function_matching/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'distill_function_matching'

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
    epochs=60,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    # weak-label and auxiliary losses
    distill_weight=0.5,
    teacher_source='online',
    teacher_run='perch_finetune_teacher',
    logit_distill_weight=1.0,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
