"""Everything that can be set for a training run, with its default.

An experiment script creates one `TrainSettings(...)` and passes only what differs from the defaults,
so reading the script shows exactly how that experiment departs from the standard recipe.
Table and bank paths are relative to the workspace (`$STRNADI_WS`) unless absolute.
"""

from dataclasses import dataclass, field


@dataclass
class TrainSettings:
    # ------------------------------------------------------------------------------------ data
    #: Human-labelled clips with cross-validation folds (columns: npy, t0, t1, label, grp, fold).
    labelled_table: str = 'data/orig_index.csv'

    #: Extra training-only tables (weak labels, synthetic songs, ...) -> share of training draws.
    #: The labelled clips get the remaining share.
    extra_tables: dict = field(default_factory=dict)

    #: Unlabelled songs for FixMatch consistency training (None = off).
    unlabelled_table: str | None = None

    #: Weakly labelled songs to draw recording-level bags from, for the bag loss (None = off).
    bag_table: str | None = None
    bag_size: int = 6
    bags_per_step: int = 4
    bag_sources: tuple = ('ncs',)

    #: Bank of background-noise chunks (.npy) mixed into training clips.
    noise_bank: str | None = None
    #: Bank of other-species vocalisations (.npy) mixed into the background.
    species_bank: str | None = None
    #: Bank of unfinished Yellowhammer songs (.npy) mixed in as a second, quieter bird.
    song_bank: str | None = None

    #: What the model is fed: 'wave' (audio), or cached Perch embeddings
    #: ('perch_pooled' = one 1536-d vector per clip, 'perch_spatial' = a 16-step series).
    inputs: str = 'wave'

    #: Length of the analysis window in seconds.
    window_seconds: float = 4.0

    #: Where the evaluation window sits inside a longer table window: 'center', or 'start'
    #: (what the research runs used; it cuts the end of songs longer than 3.4 s).
    eval_window: str = 'center'

    # -------------------------------------------------------------------------------- protocol
    folds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    seed: int = 0

    #: Share of the training recordings held out as an inner validation set (0 = none).
    inner_validation_share: float = 0.15

    #: Which epoch's weights are kept:
    #:   'inner_min_loss' lowest inner-validation loss (a checkpoint is written each time the loss
    #:                    is lower than every earlier one)
    #:   'inner_smooth3'  best 3-epoch moving average of inner-validation macro-F1
    #:   'inner_argmax'   best single epoch on the inner validation set (research runs)
    #:   'last'           the last epoch (fixed schedule)
    checkpoint_rule: str = 'inner_min_loss'

    #: Also score the test fold every epoch. For learning curves only; never used to choose anything.
    log_test_fold_curve: bool = True

    # ---------------------------------------------------------------------------- optimisation
    epochs: int = 25
    samples_per_epoch: int = 4096
    batch_size: int = 48
    learning_rate: float = 2e-3
    weight_decay: float = 0.05
    #: Learning-rate factor for pretrained backbones (the model lists them in `backbone_prefixes`).
    backbone_lr_factor: float = 1.0
    ema_decay: float = 0.998
    gradient_clip: float = 5.0
    label_smoothing: float = 0.05
    #: Class balancing inside each data pool: sampling weight = class count ** -power.
    class_balance_power: float = 0.5
    focal_gamma: float = 0.0
    #: Additive cosine margin (CosFace), for models with a cosine head.
    margin: float = 0.0
    margin_scale: float = 16.0
    #: Initialise the encoder from a pretrained checkpoint (self-supervised or distilled).
    init_from: str | None = None

    # -------------------------------------------------------------------- waveform augmentation
    #: Maximum relative speed change (pitch and tempo together); 0.02 = +-2 %.
    speed_jitter: float = 0.02
    noise_probability: float = 0.0
    snr_db_range: tuple = (0.0, 20.0)
    #: What "signal" means when setting the noise level: 'ending', 'band' or 'clip'
    #: (see augment/noise_mix.py).
    snr_reference: str = 'ending'
    far_field_probability: float = 0.0
    codec_probability: float = 0.0
    splice_probability: float = 0.0
    species_probability: float = 0.0
    random_eq_probability: float = 0.0
    #: Treble removed from hand-labelled clips; XB / BC / BE then become "Unfinished"
    #: (augment/treble_cut.py).
    treble_cut_probability: float = 0.0
    #: A second, quieter Yellowhammer in the background (never added to a "None" clip).
    background_song_probability: float = 0.0

    # ----------------------------------------------------------------- spectrogram augmentation
    freq_masks: int = 2
    freq_mask_width: int = 8
    time_masks: int = 2
    time_mask_width: int = 10
    mixstyle_probability: float = 0.0

    # --------------------------------------------------------------- weak / auxiliary losses
    #: Probability mass a candidate-set label leaves for classes outside the set (NCS rows only).
    candidate_escape_mass: float = 0.05
    #: Weight of the dialect-only weak loss (models with a factorised head).
    weak_loss_weight: float = 1.0

    fixmatch_weight: float = 1.0
    unlabelled_batch_ratio: float = 1.0
    fixmatch_min_threshold: float = 0.5

    #: Embedding distillation. 'cache' = cached Perch embeddings of the clean clip;
    #: 'online' = a trained run (`teacher_run`) applied to the same augmented audio.
    distill_weight: float = 0.0
    teacher_source: str = 'cache'
    teacher_run: str | None = None
    #: Soft-label distillation from `teacher_run`.
    logit_distill_weight: float = 0.0
    distill_temperature: float = 2.0

    bag_loss_weight: float = 0.0
    frame_loss_weight: float = 0.0
    element_loss_weight: float = 0.0
    measurement_loss_weight: float = 0.0
    #: Weight of the CTC loss on element sequences (models built with `element_tokens`).
    sequence_loss_weight: float = 0.0
    #: Weight of the discriminative sequence loss: cross-entropy over the classes, each scored by
    #: the CTC likelihood of its element sequences (`losses.element_ctc.class_log_likelihoods`).
    sequence_class_loss_weight: float = 0.0

    # --------------------------------------------------------------------------------- runtime
    #: 'cuda' trains on the GPU when one is available and falls back to the CPU otherwise.
    device: str = 'cuda'
    #: Half-precision network on the GPU (the spectrogram front-end always runs in full precision).
    mixed_precision: bool = True
    threads: int = 2
    data_workers: int = 10
    compile_model: bool = False

    def spec_augment_arguments(self):
        return dict(
            freq_masks=self.freq_masks,
            freq_width=self.freq_mask_width,
            time_masks=self.time_masks,
            time_width=self.time_mask_width,
        )

    @property
    def window_samples(self):
        return int(self.window_seconds * 32000)
