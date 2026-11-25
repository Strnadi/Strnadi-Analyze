# %% [markdown]
# # pretrained_same_song_contrastive
#
# The combined recipe started from an encoder pretrained without labels.
#
# **Question.** About 100,000 unlabelled yellowhammer songs (focal recordings, recorder archive,
# detector finds; no archive clips, no test males) are available besides the labelled data. Does an
# encoder pretrained on them give a better classifier than random initial weights? Pretraining
# task: two noisy, re-equalised views of the same song attract each other and masked frames are predicted (`ssl_same_song`).
#
# **Differs from `final_combined` in:** `init_from='ssl/ssl_v2/encoder.pt'`.
#
# **Status:** queued (2026-11-25), final label table.
#
# Run from the repository root with `python -m experiments.pretrained_same_song_contrastive`.
# Training writes to `$STRNADI_WS/runs/pretrained_same_song_contrastive/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'pretrained_same_song_contrastive'

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
    snr_reference='ending',
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    treble_cut_probability=0.25,
    # initial weights
    init_from='ssl/ssl_v2/encoder.pt',
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
