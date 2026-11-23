# %% [markdown]
# # element_tagger
#
# Element-sequence model: every 20 ms frame is tagged background / phrase / B / C / D / E / X; the
# dialect is read off the terminal elements. Trained jointly with the clip labels.
#
# **Question.** "Unfinished" becomes a sequence without an ending, an unknown dialect becomes an
# unseen sequence (Italian data), the output shows where the dialect sits, and the same
# formulation serves any species on the BioAcu platform.
#
# **Status:** proposed (from review proposal). Not run. Frame supervision exists only for Vizina
# (component boxes) and the counterfactual surgery positions. The model, loss and trainer hook are
# in place; the table builder is NOT written yet: it has to map the Vizina phrase types A-F /
# component names onto phrase / B / C / D / E / X, which needs the annotators' key
# (docs/STATUS.md).
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv; data/element_frames_v1.csv with an `elements` column: JSON [[name, start_s,
# end_s], ...] per row
#
# Run from the repository root with `python -m experiments.element_tagger`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/element_tagger/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.element_tagger import ElementTagger
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_tagger'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
        'data/element_frames_v1.csv': 0.1,
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
    element_loss_weight=1.0,
)


# %% Model
def make_model():
    return ElementTagger()


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
