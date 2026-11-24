# %% [markdown]
# # counterfactual_treble_cut
#
# The reference plus treble-cut counterfactuals: an XB, BC or BE song without its treble is Unfinished.
#
# **Question.** Every model calls "phrase, pause, low B" XB even when no X is visible (15 of the 64
# clips all ten compared runs get wrong), and with everything above 6 kHz removed the reference
# still calls 81 % of XB songs XB: it reads the dialect from the timing, not from the X. Here a
# quarter of the hand-labelled clips have their treble removed, and XB / BC / BE clips are then
# labelled Unfinished (`augment/treble_cut.py`). Does "Unfinished called XB" go away, and what does
# it cost in XB recall on faint recordings?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `treble_cut_probability=0.25`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.counterfactual_treble_cut`.
# Training writes to `$STRNADI_WS/runs/counterfactual_treble_cut/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'counterfactual_treble_cut'

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
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
    treble_cut_probability=0.25,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
