# %% [markdown]
# # measurement_bottleneck
#
# Concept bottleneck: the network regresses the expert measurements of the ending (f1, f2, f3, df,
# bw, t2, element presence) and the class is predicted from those values only.
#
# **Question.** Forcing the model to "measure like an ornithologist" makes it use the right
# evidence and makes every decision explainable in the scientists' own terms.
#
# **Status:** proposed (from review proposal). Not run. Research backlog item "physics-informed
# multitask", never built. Model, loss and trainer hook are in place; the targets table is NOT
# built yet (docs/STATUS.md). Check Table S1's frequency columns first: its timing column is a
# 6.000 s placeholder in 384 of 430 rows.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; rows with columns m_f1, m_f2, m_f3, m_df, m_bw (kHz), m_t2 (s), m_has_first,
# m_has_second (from Diblikova et al. 2023 Table S1; blank where unknown)
#
# Run from the repository root with `python -m experiments.measurement_bottleneck`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/measurement_bottleneck/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet_measure import MeasurementStrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'measurement_bottleneck'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    # weak-label and auxiliary losses
    measurement_loss_weight=0.5,
)


# %% Model
def make_model():
    return MeasurementStrnadNet()


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
