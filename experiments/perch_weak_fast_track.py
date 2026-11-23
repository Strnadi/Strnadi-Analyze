# %% [markdown]
# # perch_weak_fast_track
#
# The weak-label recipes (factorised head, pool shares, bag loss) searched on cached Perch
# embeddings, where a fold takes minutes instead of 2-4 CPU hours.
#
# **Question.** Recipe choices transfer from frozen-Perch features to the small model, so the
# search can run ~100x faster; the result is also a strong distillation teacher.
#
# **Status:** proposed (from review proposal). Not run.
#
# **Needs.** Perch embeddings for the weak / counterfactual / Vizina tables:
# processing/extract_foundation_emb.py <table> (-> emb/rows_perch_spatial.npy). The trainer
# refuses rows without a cached embedding.
#
# Run from the repository root with `python -m experiments.perch_weak_fast_track`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/perch_weak_fast_track/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.perch_temporal import PerchTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'perch_weak_fast_track'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    inputs='perch_spatial',
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=20,
    samples_per_epoch=16384,
    batch_size=64,
    learning_rate=0.001,
)


# %% Model
def make_model():
    return PerchTemporal(precomputed=True, head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
