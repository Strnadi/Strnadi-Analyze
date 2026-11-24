# %% [markdown]
# # element_sequence_anchored
#
# The element-sequence model with its tokens anchored in time by the experts' element annotation.
#
# **Question.** With CTC alone the tokens name the elements but are not where the elements are:
# on the held-out focal songs the `phrase` token falls inside the annotated phrase 80 % of the time
# (40 % in the shipped recipe), the first terminal token is a median 0.5 s from the annotated
# terminal onset, and the number of decoded terminal elements matches the annotation in only a
# third of the songs (`evaluation/element_timing.py`). The experts marked phrase and terminal
# components in 1,600 songs of the training males. Here those songs constrain the tokens: `phrase`
# only inside the phrase, terminal tokens only inside the terminal span, blank around the song
# (`processing/build_element_spans.py`, `training.data.token_frame_mask`). Does the decoder then
# localise elements, and count them, on males it never heard? This is the step from naming a
# dialect towards detecting its elements.
#
# **Differs from `element_sequence_ctc` in:** `sequence_anchor_weight=1.0`, and the expert songs
# come from `data/vizina_element_spans.csv` (with timing) instead of `data/vizann_v1.csv`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.element_sequence_anchored`.
# Training writes to `$STRNADI_WS/runs/element_sequence_anchored/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_sequence_anchored'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizina_element_spans.csv': 0.07,
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
    # auxiliary losses
    sequence_loss_weight=0.5,
    sequence_anchor_weight=1.0,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_tokens=len(TOKENS))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
