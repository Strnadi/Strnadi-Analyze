"""Training: recording-grouped cross-validation.

    cross_validate(run_name, make_model, settings)

trains one model per fold and writes, under `$STRNADI_WS/runs/<run_name>/fold<k>/`:

    best.pt     written every time the inner-validation loss reaches a new minimum
    model.pt    the kept weights (see `TrainSettings.checkpoint_rule`; by default the same as best.pt)
    last.pt     the weights after the last epoch
    oof.npz     predictions for the fold's held-out clips ("out of fold")
    log.json    one entry per epoch: loss, inner-validation and test-fold scores
    tb/         TensorBoard event files (`tensorboard --logdir $STRNADI_WS/runs`)

Protocol for fold k:

- the test fold (labelled clips with fold == k) is never used to choose anything;
- an inner validation set (a share of the TRAINING recordings) is scored every epoch and decides
  which epoch's weights are kept; weak and unlabelled windows of those recordings are left out of
  training as well;
- the EMA weights are the ones that are evaluated and saved;
- a resume file is written after every epoch, so a crash or reboot continues where it stopped.

Environment overrides, so one script can be spread over cores or re-run with another seed:
STRNADI_FOLDS=0,1  STRNADI_THREADS=5  STRNADI_WORKERS=2  STRNADI_SEED=1 (run name gets `_s1`).
"""

import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from augment.pipeline import WaveAugment
from config import C2I, CLASSES, WS, dry_run, effective_run_name, workspace_path
from layers.distill_projection import DistillProjection
from layers.freq_mixstyle import FreqMixStyle
from layers.spec_augment import SpecAugment
from losses.bag_complete import bag_complete_nll
from losses.distill_cosine import distill_cosine
from losses.element_ctc import class_log_likelihoods, element_ctc_loss
from losses.factorized_weak import factorized_weak_loss
from losses.fixmatch import FreeMatch
from losses.generalized_cross_entropy import generalized_cross_entropy
from losses.frame_bce import frame_bce
from losses.frame_ce import frame_ce
from losses.margin import cosface_margin
from losses.measurement import measurement_loss
from losses.partial_label import partial_label_nll
from losses.soft_cross_entropy import soft_cross_entropy
from model.base import count_parameters

from .checkpoint import load_encoder_state, save_model
from .data import (
    CANDIDATE_SET,
    HARD_LABEL,
    SOFT_LABEL,
    BagDataset,
    ClipDataset,
    EmbeddingStore,
    UnlabeledDataset,
    duplicate_recordings,
    read_table,
)
from .ema import EMA
from .metrics import metrics
from .optim import build_optimizer
from .sampler import pool_sampling_weights
from .validation import load_eval_waveforms, mean_nll, predict_waveforms


# =========================================================================== public entry points
def cross_validate(run_name, make_model, settings):
    """Train and evaluate one model per fold.

    Args:
        run_name: name of the run directory under `$STRNADI_WS/runs/`.
        make_model: function without arguments that returns a freshly initialised model.
        settings: `TrainSettings`.

    Returns:
        {fold: metrics on that fold's held-out clips} for the folds trained in this call.
    """
    settings = _apply_environment_overrides(settings)
    run_name = effective_run_name(run_name)

    if dry_run():
        _dry_run(run_name, make_model, settings)
        return {}

    torch.set_num_threads(settings.threads)
    run_dir = f'{WS}/runs/{run_name}'
    os.makedirs(run_dir, exist_ok=True)
    with open(f'{run_dir}/settings.json', 'w') as file:
        json.dump(vars(settings), file, indent=1, default=str)

    results = {}
    for fold in settings.folds:
        if os.path.exists(f'{run_dir}/fold{fold}/oof.npz'):
            print(f'fold {fold} is already finished, skipping', flush=True)
            continue
        results[fold] = fit_fold(run_name, make_model, settings, fold)
        _print_fold_result(fold, results[fold])
    return results


#: "Fold" number of the final model: no labelled clip has it, so nothing is held out.
ALL_DATA = 99


def train_final_model(run_name, make_model, settings):
    """Train the model that is shipped: on the labelled clips of ALL folds.

    Nothing is held out for testing, so this run has no accuracy of its own: its expected accuracy
    is the cross-validated result of the same recipe (`cross_validate` with the same settings).
    The inner validation set (a share of the recordings, never trained on) still chooses the
    checkpoint with the lowest validation loss.

    Writes `$STRNADI_WS/runs/<run_name>/final/` with `best.pt`, `model.pt`, `last.pt`, `log.json`
    and TensorBoard files. Returns the path of `model.pt`.
    """
    settings = _apply_environment_overrides(settings)
    run_name = effective_run_name(run_name)
    if dry_run():
        _dry_run(run_name, make_model, settings)
        return None

    torch.set_num_threads(settings.threads)
    os.makedirs(f'{WS}/runs/{run_name}', exist_ok=True)
    FoldTraining(run_name, make_model, settings, ALL_DATA).run()
    return f'{WS}/runs/{run_name}/final/model.pt'


def fit_fold(run_name, make_model, settings, fold):
    """Train one fold and return the metrics on its held-out clips."""
    return FoldTraining(run_name, make_model, settings, fold).run()


# ================================================================================ data splitting
def split_tables(settings, fold):
    """Split the tables into training, inner-validation and test rows for one fold.

    Returns:
        (train, inner_validation, test, unlabelled, bag_rows); the last two are None when unused.
        Training rows carry a `pool` column naming the table they came from.
    """
    labelled = read_table(settings.labelled_table)
    if 'aug' in labelled:
        labelled = labelled[~labelled['aug']]  # pre-baked augmented copies are never used

    test = labelled[labelled.fold == fold]
    train = labelled[labelled.fold != fold]

    # Inner validation: a fixed random subset of the training recordings.
    recordings = train.grp.unique()
    n_inner = int(len(recordings) * settings.inner_validation_share)
    inner_recordings = set(
        np.random.default_rng(1000 + fold).choice(recordings, size=n_inner, replace=False)
    )
    inner_validation = train[train.grp.isin(inner_recordings)]
    train = train[~train.grp.isin(inner_recordings)].assign(pool='strong')

    # Relabelled training targets (e.g. disputed clips as candidate sets); evaluation keeps `label`.
    if 'train_label' in train:
        train['label'] = train.train_label.fillna(train.label)

    recording_fold = dict(zip(labelled.grp.astype(str), labelled.fold))
    same_audio = duplicate_recordings()

    def usable_for_training(table):
        """Drop rows of the test fold and of the inner-validation recordings.

        A row of an extra table belongs to the fold of its recording in the labelled table (also
        when the recording is a byte-identical copy filed under another id); only recordings
        without labelled clips keep the fold written in their own table.
        """
        if 'rec' in table:
            recording = table.rec.astype(str).str.replace('ncs:', '', regex=False)
            recording = recording.map(lambda name: same_audio.get(name, name))
            if 'fold' in table:
                table = table.assign(fold=recording.map(recording_fold).fillna(table.fold))
            table = table[~recording.isin(inner_recordings).values]
        if 'fold' in table:
            table = table[table.fold != fold]
        return table

    for path in settings.extra_tables:
        extra = read_table(path)
        if 'label_fold0' in extra:
            # machine-labelled songs: use the labels of the teachers that never saw this fold
            # (processing/machine_label_songs.py); the final model uses the joint label
            column = 'label' if fold == ALL_DATA else f'label_fold{fold}'
            extra = extra.assign(label=extra[column]).dropna(subset=['label'])
        extra = usable_for_training(extra)
        extra = extra.assign(pool=_pool_name(path))
        train = pd.concat([train, extra], ignore_index=True)

    if 'w' in train:
        train['w'] = train['w'].fillna(1.0)
    train = train.reset_index(drop=True)

    unlabelled = None
    if settings.unlabelled_table:
        unlabelled = usable_for_training(read_table(settings.unlabelled_table))

    bag_rows = None
    if settings.bag_table and settings.bag_loss_weight > 0:
        bag_rows = usable_for_training(read_table(settings.bag_table))

    return train, inner_validation, test, unlabelled, bag_rows


def _pool_name(table_path):
    """'data/weak_v3.csv' -> 'weak_v3'."""
    return os.path.basename(table_path).replace('.csv', '')


# ======================================================================== checkpoint selection
class CheckpointKeeper:
    """Decides which epoch's weights to keep.

    'inner_min_loss' the epoch with the lowest inner-validation loss (the default: a checkpoint
                     is kept only when the loss is lower than every earlier one);
    'inner_argmax'   the epoch with the best inner-validation score;
    'inner_smooth3'  the epoch with the best centred 3-epoch moving average of that score
                     (a single good epoch is often a fluke on a ~380-clip validation set);
    'last'           the last epoch.
    """

    def __init__(self, rule, n_epochs):
        self.rule = rule
        self.n_epochs = n_epochs
        self.scores = []
        self.best_single = (-1.0, -1, None)  # (score, epoch, weights)
        self.best_smoothed = (-1.0, -1, None)
        self.best_loss = (float('inf'), -1, None)  # (loss, epoch, weights)
        self.previous_weights = None

    def update(self, epoch, weights, inner_score, inner_loss=None):
        """Record this epoch's weights and inner-validation results (None if there is no inner set).

        Returns True when the inner-validation loss reached a new minimum.
        """
        improved = False
        if inner_loss is not None and inner_loss < self.best_loss[0]:
            self.best_loss = (inner_loss, epoch, weights)
            improved = True

        if inner_score is not None:
            self.scores.append(inner_score)

            if inner_score > self.best_single[0]:
                self.best_single = (inner_score, epoch, weights)

            # The moving average centred on the PREVIOUS epoch is complete now.
            if self.previous_weights is not None:
                smoothed = float(np.mean(self.scores[-3:]))
                if smoothed > self.best_smoothed[0]:
                    self.best_smoothed = (smoothed, epoch - 1, self.previous_weights)

            # The last epoch has no successor: average it with its predecessor only.
            if epoch == self.n_epochs - 1:
                smoothed = float(np.mean(self.scores[-2:]))
                if smoothed > self.best_smoothed[0]:
                    self.best_smoothed = (smoothed, epoch, weights)

        self.previous_weights = weights
        return improved

    def chosen(self, last_weights):
        """Return (weights, epoch, rule actually applied)."""
        have_scores = len(self.scores) > 0
        if self.rule == 'inner_min_loss' and self.best_loss[2] is not None:
            return self.best_loss[2], self.best_loss[1], 'inner_min_loss'
        if self.rule == 'inner_argmax' and have_scores:
            return self.best_single[2], self.best_single[1], 'inner_argmax'
        if self.rule == 'inner_smooth3' and have_scores:
            if self.best_smoothed[2] is not None:
                return self.best_smoothed[2], self.best_smoothed[1], 'inner_smooth3'
            return self.best_single[2], self.best_single[1], 'inner_argmax'
        return last_weights, self.n_epochs - 1, 'last'

    def state(self):
        return {
            'scores': self.scores,
            'best_single': self.best_single,
            'best_smoothed': self.best_smoothed,
            'best_loss': self.best_loss,
            'previous_weights': self.previous_weights,
        }

    def load_state(self, state):
        self.scores = state['scores']
        self.best_single = tuple(state['best_single'])
        self.best_smoothed = tuple(state['best_smoothed'])
        self.best_loss = tuple(state.get('best_loss', (float('inf'), -1, None)))
        self.previous_weights = state['previous_weights']


# =================================================================================== one fold
class FoldTraining:
    """Everything needed to train and evaluate one fold."""

    def __init__(self, run_name, make_model, settings, fold):
        self.settings = settings
        self.fold = fold
        directory = 'final' if fold == ALL_DATA else f'fold{fold}'
        self.out_dir = f'{WS}/runs/{run_name}/{directory}'
        os.makedirs(self.out_dir, exist_ok=True)
        self.resume_path = f'{self.out_dir}/resume.pt'

        use_cuda = settings.device == 'cuda' and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.mixed_precision = settings.mixed_precision and use_cuda
        torch.backends.cudnn.benchmark = True
        self.board = SummaryWriter(f'{self.out_dir}/tb')

        seed = settings.seed + fold
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self._prepare_data()
        self._prepare_model(make_model)
        self._prepare_auxiliary_training()
        self._prepare_optimisation()

    # ------------------------------------------------------------------------------ set-up
    def _prepare_data(self):
        settings = self.settings
        tables = split_tables(settings, self.fold)
        train, self.inner_validation, self.test, unlabelled, bag_rows = tables

        self.augment = WaveAugment(
            settings,
            noise_bank=_load_bank(settings.noise_bank),
            species_bank=_load_bank(settings.species_bank),
            song_bank=_load_bank(settings.song_bank),
            far_field_csv=f'{WS}/data/farfield_stats.csv',
        )

        # Cached embeddings: as model input, and/or as distillation targets.
        self.input_store = None
        if settings.inputs != 'wave':
            self.input_store = EmbeddingStore(spatial=settings.inputs == 'perch_spatial')
        teacher_store = None
        if settings.distill_weight > 0 and settings.teacher_source == 'cache':
            teacher_store = EmbeddingStore()

        # Labelled + weak rows, drawn with fixed shares per source table.
        weights, shares = pool_sampling_weights(
            train,
            {_pool_name(p): s for p, s in settings.extra_tables.items()},
            settings.class_balance_power,
        )
        print('share of training draws per table:', shares, flush=True)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=settings.samples_per_epoch,
            replacement=True,
        )
        train_dataset = ClipDataset(
            train,
            settings,
            train=True,
            augment=self.augment,
            teacher_store=teacher_store,
            input_store=self.input_store,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=settings.batch_size, sampler=sampler, **self._loader_options()
        )

        self.unlabelled_batches = None
        if unlabelled is not None:
            dataset = UnlabeledDataset(unlabelled, settings, self.augment)
            batch_size = int(settings.batch_size * settings.unlabelled_batch_ratio)
            self.unlabelled_batches = self._endless_batches(dataset, batch_size)

        self.bag_batches = None
        if bag_rows is not None:
            dataset = BagDataset(bag_rows, settings, self.augment)
            self.bag_batches = self._endless_batches(dataset, settings.bags_per_step)

        self.test_labels = self.test.label.map(C2I).values
        self.inner_labels = self.inner_validation.label.map(C2I).values
        self.class_weights = _loss_class_weights(train, settings.class_weight_power)

        # The evaluation windows never change: read them once and keep them in memory.
        self.inner_waveforms = load_eval_waveforms(
            self.inner_validation, settings, self.input_store
        )
        self.test_waveforms = load_eval_waveforms(self.test, settings, self.input_store)

    def _loader_options(self):
        workers = self.settings.data_workers
        return dict(
            num_workers=workers,
            drop_last=True,
            persistent_workers=workers > 0,
            pin_memory=self.device.type == 'cuda',
            prefetch_factor=4 if workers > 0 else None,
        )

    def _endless_batches(self, dataset, batch_size):
        """An iterator that never runs out (sampling with replacement)."""
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=10**9)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, **self._loader_options()
        )
        return iter(loader)

    def _prepare_model(self, make_model):
        settings = self.settings
        self.model = make_model().to(self.device).to(memory_format=torch.channels_last)
        print(
            f'{type(self.model).__name__}: {count_parameters(self.model):,} parameters', flush=True
        )

        if settings.init_from:
            pretrained = torch.load(
                workspace_path(settings.init_from), map_location='cpu', weights_only=False
            )
            load_encoder_state(self.model, pretrained['state'])

        # Models fed raw audio or embeddings get no spectrogram augmentation from the trainer.
        self.takes_waveform = (
            getattr(self.model, 'wave_input', False) or self.model.frontend is None
        )
        self.spec_augment = SpecAugment(
            **settings.spec_augment_arguments(), fill=getattr(self.model, 'spec_fill', 'zero')
        ).train()
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)
        self.mixstyle = None
        if settings.mixstyle_probability > 0:
            self.mixstyle = FreqMixStyle(settings.mixstyle_probability).train()

        self.ema = EMA(self.model, settings.ema_decay)
        self.compiled_model = torch.compile(self.model) if settings.compile_model else self.model

    def _prepare_auxiliary_training(self):
        settings = self.settings

        # A trained run as teacher (soft labels and/or embeddings on the same augmented audio).
        self.teacher = None
        wants_online_embeddings = (
            settings.distill_weight > 0 and settings.teacher_source == 'online'
        )
        if settings.teacher_run and (settings.logit_distill_weight > 0 or wants_online_embeddings):
            from .inference import TorchEnsemble

            teacher_dir = settings.teacher_run
            if not os.path.isabs(teacher_dir):
                teacher_dir = f'{WS}/runs/{teacher_dir}'
            self.teacher = TorchEnsemble(teacher_dir, device=self.device)

        # Projection of the student's embedding into the teacher's embedding space.
        self.distill_projection = None
        if settings.distill_weight > 0:
            example = self._example_input()
            with torch.no_grad():
                student_dim = self.model.eval()(example)['emb'].shape[1]
                if wants_online_embeddings:
                    teacher_dim = self.teacher.models[0](example)['emb'].shape[1]
                else:
                    teacher_dim = 1536  # Perch 2.0
            self.distill_projection = DistillProjection(student_dim, teacher_dim).to(self.device)

        self.fixmatch = None
        if self.unlabelled_batches is not None:
            self.fixmatch = FreeMatch(len(CLASSES), settings.fixmatch_min_threshold)

    def _example_input(self):
        if self.settings.inputs == 'wave':
            return torch.zeros(2, self.settings.window_samples, device=self.device)
        return next(iter(self.train_loader))['wave'][:2].to(self.device)

    def _prepare_optimisation(self):
        settings = self.settings
        extra_parameters = []
        if self.distill_projection is not None:
            extra_parameters = list(self.distill_projection.parameters())

        steps_per_epoch = settings.samples_per_epoch // settings.batch_size
        self.optimizer, self.scheduler = build_optimizer(
            self.model, extra_parameters, settings, settings.epochs * steps_per_epoch
        )
        self.trainable_parameters = [
            parameter for group in self.optimizer.param_groups for parameter in group['params']
        ]

        self.keeper = CheckpointKeeper(settings.checkpoint_rule, settings.epochs)
        self.log = []
        self.first_epoch = 0

    # ---------------------------------------------------------------------------- main loop
    def run(self):
        settings = self.settings
        self._resume_if_possible()
        started = time.time()

        for epoch in range(self.first_epoch, settings.epochs):
            mean_loss = self._train_one_epoch()
            entry = {'ep': epoch, 'loss': mean_loss, 't': time.time() - started}

            weights = self._ema_weights_on_cpu()
            inner_score, inner_loss = self._validate(entry)
            improved = self.keeper.update(epoch, weights, inner_score, inner_loss)
            if improved:
                # Best practice: keep a checkpoint only when the validation loss is a new minimum.
                save_model(
                    f'{self.out_dir}/best.pt',
                    weights,
                    self.model,
                    epoch,
                    settings,
                    inner_loss=inner_loss,
                )
            entry['new_best'] = improved

            self.log.append(entry)
            print(self.fold, entry, flush=True)
            with open(f'{self.out_dir}/log.json', 'w') as file:
                json.dump(self.log, file)
            self._write_board(epoch, entry)
            self._save_resume_point(epoch)

            patience = settings.early_stopping_patience
            epochs_since_best = epoch - self.keeper.best_loss[1]
            if patience > 0 and inner_loss is not None and epochs_since_best >= patience:
                print(f'{self.fold} early stop: no new validation minimum for {patience} epochs')
                self.board.add_text('early_stopping', f'stopped after epoch {epoch}')
                break

        return self._finish()

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in self.train_loader:
            batch = {
                name: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for name, value in batch.items()
            }
            loss = self._training_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.trainable_parameters, self.settings.gradient_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()
            self.ema.update(self.model)

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _ema_weights_on_cpu(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.ema.model.state_dict().items()
        }

    def _write_board(self, epoch, entry):
        """One TensorBoard point per epoch for every number in the log entry."""
        tags = {
            'loss': 'loss/train',
            'inner_loss': 'loss/inner_validation',
            'val_loss': 'loss/test_fold_diagnostic',
            'inner_macro_f1': 'macro_f1/inner_validation',
            'val_macro_f1': 'macro_f1/test_fold_diagnostic',
            'inner_acc': 'accuracy/inner_validation',
            'val_acc': 'accuracy/test_fold_diagnostic',
        }
        for key, tag in tags.items():
            if key in entry:
                self.board.add_scalar(tag, entry[key], epoch)
        for name, recall in entry.get('inner_recall', {}).items():
            if recall is not None:
                self.board.add_scalar(f'recall_inner_validation/{name}', recall, epoch)
        self.board.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        self.board.flush()

    def _validate(self, entry):
        """Score the EMA weights and fill `entry`.

        Returns:
            (inner-validation macro-F1, inner-validation loss); both None without an inner set.
        """
        settings = self.settings
        inner_score = None
        inner_loss = None

        if len(self.inner_validation):
            probabilities = self._predict(self.inner_waveforms)
            inner = metrics(self.inner_labels, probabilities)
            inner_score = inner['macro_f1']
            inner_loss = mean_nll(self.inner_labels, probabilities)
            entry['inner_macro_f1'] = inner['macro_f1']
            entry['inner_acc'] = inner['acc']
            entry['inner_loss'] = inner_loss
            entry['inner_recall'] = inner['per_class_recall']

        if settings.log_test_fold_curve and len(self.test):
            # For learning curves only: this number is never used to choose anything.
            probabilities = self._predict(self.test_waveforms)
            outer = metrics(self.test_labels, probabilities)
            entry['val_macro_f1'] = outer['macro_f1']
            entry['val_acc'] = outer['acc']
            entry['val_loss'] = mean_nll(self.test_labels, probabilities)

        return inner_score, inner_loss

    def _predict(self, waveforms):
        return predict_waveforms(self.ema.model, waveforms, self.device, self.mixed_precision)

    # -------------------------------------------------------------------------------- losses
    def _forward(self, waveforms, augment_spectrogram=True, extra_masking=False):
        """Run the model on a batch of waveforms, with spectrogram augmentation where it applies."""
        if self.takes_waveform:
            return self.compiled_model(waveforms)

        # The spectrogram is computed in full precision (the log of a half-precision power
        # underflows); the network itself runs in mixed precision on a GPU.
        spectrogram = self.model.frontend(waveforms).contiguous(memory_format=torch.channels_last)
        if augment_spectrogram:
            spectrogram = self.spec_augment(spectrogram)
            if extra_masking:
                spectrogram = self.spec_augment(spectrogram)
            if self.mixstyle is not None:
                spectrogram = self.mixstyle(spectrogram)
        with torch.autocast('cuda', enabled=self.mixed_precision):
            outputs = self.compiled_model(spectrogram)
        return {
            name: value.float() if torch.is_tensor(value) else value
            for name, value in outputs.items()
        }

    def _training_loss(self, batch):
        settings = self.settings
        waveforms = batch['wave']
        outputs = self._forward(waveforms)

        example_weights = batch['w'].float()
        if self.class_weights is not None:
            # hand-labelled clips of rare classes count more (their target is one-hot)
            class_weight = (batch['target'] * self.class_weights.to(self.device)).sum(1)
            is_hard = batch['kind'] == HARD_LABEL
            example_weights = torch.where(is_hard, example_weights * class_weight, example_weights)
        loss = classification_loss(
            outputs, batch['target'], batch['kind'], example_weights, settings
        )

        # --- per-frame auxiliary targets
        if settings.frame_loss_weight > 0 and outputs.get('frame_logits') is not None:
            frame_loss = frame_bce(outputs['frame_logits'], batch['frames'])
            loss = loss + settings.frame_loss_weight * frame_loss
        if settings.element_loss_weight > 0 and 'element_logits' in outputs:
            element_loss = frame_ce(outputs['element_logits'], batch['elements'])
            loss = loss + settings.element_loss_weight * element_loss
        if settings.measurement_loss_weight > 0 and 'measurements' in outputs:
            measure_loss = measurement_loss(outputs['measurements'], batch['measure'])
            loss = loss + settings.measurement_loss_weight * measure_loss

        if settings.sequence_loss_weight > 0 and 'token_log_probs' in outputs:
            # Only clips with a hard label teach the sequence. A candidate set such as
            # `BC|Unfinished|None` would let CTC settle on the easy reading "phrase B" for every
            # weakly labelled song, and the decoder then stops emitting terminal elements.
            hard_targets = batch['target'] * (batch['kind'] == HARD_LABEL)[:, None]
            sequence_loss = element_ctc_loss(
                outputs['token_log_probs'], hard_targets, batch['w'].float()
            )
            loss = loss + settings.sequence_loss_weight * sequence_loss

        if settings.sequence_anchor_weight > 0 and 'token_log_probs' in outputs:
            # probability mass each frame puts on the tokens allowed there (1 where unconstrained)
            allowed_mass = (outputs['token_log_probs'].exp() * batch['token_mask']).sum(-1)
            anchor_loss = -torch.log(allowed_mass.clamp_min(1e-6)).mean()
            loss = loss + settings.sequence_anchor_weight * anchor_loss

        if settings.sequence_class_loss_weight > 0 and 'token_log_probs' in outputs:
            # Discriminative version: the classes compete. Plain CTC only raises the likelihood of
            # the labelled sequence; here "phrase B C" must also beat "phrase B".
            is_hard = batch['kind'] == HARD_LABEL
            if is_hard.any():
                class_scores = class_log_likelihoods(outputs['token_log_probs'][is_hard])
                true_class = batch['target'][is_hard].argmax(1)
                sequence_class_loss = nn.functional.nll_loss(class_scores, true_class)
                loss = loss + settings.sequence_class_loss_weight * sequence_class_loss

        # --- distillation
        if self.distill_projection is not None:
            loss = loss + settings.distill_weight * self._embedding_distillation(
                outputs, batch, waveforms
            )
        if self.teacher is not None and settings.logit_distill_weight > 0:
            loss = loss + settings.logit_distill_weight * self._logit_distillation(
                outputs, waveforms
            )

        # --- unlabelled songs and recording-level bags
        if self.unlabelled_batches is not None:
            loss = loss + settings.fixmatch_weight * self._fixmatch_loss()
        if self.bag_batches is not None:
            loss = loss + settings.bag_loss_weight * self._bag_loss()

        return loss

    def _embedding_distillation(self, outputs, batch, waveforms):
        student = self.distill_projection(outputs['emb'])

        if self.teacher is not None and self.settings.teacher_source == 'online':
            # "Function matching": the teacher sees exactly the augmented audio the student saw.
            with torch.no_grad():
                teacher_embeddings = [model(waveforms)['emb'] for model in self.teacher.models]
                target = torch.stack(teacher_embeddings).mean(0)
            return distill_cosine(student, target)

        return distill_cosine(student, batch['teacher_emb'], batch['teacher_mask'].float())

    def _logit_distillation(self, outputs, waveforms):
        temperature = self.settings.distill_temperature
        with torch.no_grad():
            teacher_log_probs = [
                torch.log_softmax(model(waveforms)['logits'], 1) for model in self.teacher.models
            ]
            ensemble = torch.stack(teacher_log_probs).logsumexp(0)
            teacher_probs = torch.softmax(ensemble / temperature, 1)

        student_log_probs = torch.log_softmax(outputs['logits'] / temperature, 1)
        cross_entropy = -(teacher_probs * student_log_probs).sum(1).mean()
        return temperature**2 * cross_entropy

    def _fixmatch_loss(self):
        weak_view, strong_view = next(self.unlabelled_batches)
        weak_view, strong_view = weak_view.to(self.device), strong_view.to(self.device)
        with torch.no_grad():
            teacher_logits = self.ema.model(weak_view)['logits']
            pseudo_labels, confident = self.fixmatch.pseudo_labels(teacher_logits)

        student_logits = self._forward(strong_view, extra_masking=True)['logits']
        return self.fixmatch.loss(student_logits, pseudo_labels, confident)

    def _bag_loss(self):
        bags, dialects = next(self.bag_batches)  # (n_bags, bag_size, samples), (n_bags,)
        bags, dialects = bags.to(self.device), dialects.to(self.device)
        bag_size = bags.shape[1]
        outputs = self._forward(bags.flatten(0, 1))
        if 'log_dialect' not in outputs:
            raise ValueError('the bag loss needs a model with head="factorized"')
        return bag_complete_nll(outputs, dialects, bag_size).mean()

    # ------------------------------------------------------------------- resume and finish
    def _save_resume_point(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'ema': self.ema.model.state_dict(),
            'ema_step': self.ema.step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'log': self.log,
            'keeper': self.keeper.state(),
            'distill_projection': None,
            'fixmatch': None,
        }
        if self.distill_projection is not None:
            state['distill_projection'] = self.distill_projection.state_dict()
        if self.fixmatch is not None:
            state['fixmatch'] = self.fixmatch.state()

        # Write to a temporary file first, so a crash while saving cannot corrupt the resume point.
        torch.save(state, self.resume_path + '.tmp')
        os.replace(self.resume_path + '.tmp', self.resume_path)

    def _resume_if_possible(self):
        already_finished = os.path.exists(f'{self.out_dir}/oof.npz')
        if not os.path.exists(self.resume_path) or already_finished:
            return

        state = torch.load(self.resume_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state['model'])
        self.ema.model.load_state_dict(state['ema'])
        self.ema.step = state['ema_step']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.log = state['log']
        self.keeper.load_state(state['keeper'])
        if self.distill_projection is not None and state['distill_projection'] is not None:
            self.distill_projection.load_state_dict(state['distill_projection'])
        if self.fixmatch is not None and state['fixmatch'] is not None:
            self.fixmatch.load(state['fixmatch'])

        self.first_epoch = state['epoch'] + 1
        print('resumed after epoch', state['epoch'], flush=True)

    def _finish(self):
        """Save the checkpoints, predict the held-out clips and return their metrics."""
        settings = self.settings
        last_weights = self._ema_weights_on_cpu()
        save_model(
            f'{self.out_dir}/last.pt', last_weights, self.model, settings.epochs - 1, settings
        )

        weights, epoch, rule = self.keeper.chosen(last_weights)
        save_model(
            f'{self.out_dir}/model.pt', weights, self.model, epoch, settings, checkpoint_rule=rule
        )

        if os.path.exists(self.resume_path):
            os.remove(self.resume_path)
        if len(self.test) == 0:  # the final model: nothing was held out
            self.board.close()
            return None

        self.ema.model.load_state_dict(weights)
        probabilities = self._predict(self.test_waveforms)
        np.savez(
            f'{self.out_dir}/oof.npz',
            p=probabilities,
            y=self.test_labels,
            idx=self.test.index.values,
            name=self.test.name.values,
            strong=settings.labelled_table,
            eval_crop=settings.eval_window,
            select=rule,
            epoch=epoch,
        )

        sources = self.test.src.values if 'src' in self.test else None
        result = metrics(self.test_labels, probabilities, sources)
        self._write_board_summary(result, epoch, rule)
        return result

    def _write_board_summary(self, result, epoch, rule):
        """The held-out result of this fold as TensorBoard text, hyper-parameters and a figure."""
        from evaluation.report import confusion_figure

        self.board.add_text(
            'held_out_fold',
            f'kept epoch {epoch} ({rule}); macro-F1 {result["macro_f1"]:.3f}; '
            f'accuracy {result["acc"]:.3f}; recall {result["per_class_recall"]}',
        )
        self.board.add_figure('confusion/held_out_fold', confusion_figure(np.array(result['cm'])))
        simple_settings = {
            name: value if isinstance(value, (int, float, str, bool)) else str(value)
            for name, value in vars(self.settings).items()
        }
        self.board.add_hparams(
            simple_settings,
            {'hparam/macro_f1': result['macro_f1'], 'hparam/accuracy': result['acc']},
            run_name='.',
        )
        self.board.close()


# ============================================================================ classification loss
def classification_loss(outputs, targets, kinds, weights, settings):
    """The class loss of one batch, mixing the three kinds of labels.

    - hard and soft labels: cross-entropy (label smoothing, optional focal modulation);
    - candidate sets: either the partial-label loss on the 7 classes, or, for a model with a
      factorised head, the dialect-only weak loss (the recording label then cannot push a song
      towards "Unfinished").

    Args:
        outputs: model output dict.
        targets: (B, 7) target vectors from `encode_label`.
        kinds: (B,) label kind per example.
        weights: (B,) loss weight per example.
    """
    logits = outputs['logits']
    has_full_label = (kinds == HARD_LABEL) | (kinds == SOFT_LABEL)
    is_candidate_set = kinds == CANDIDATE_SET

    if settings.margin > 0:
        margin_targets = targets * (~is_candidate_set)[:, None].float()
        logits = cosface_margin(logits, margin_targets, settings.margin, settings.margin_scale)

    use_factorized_weak_loss = bool(is_candidate_set.any()) and 'log_dialect' in outputs

    per_example = torch.zeros(len(logits), device=logits.device)
    if has_full_label.any() and settings.robust_loss_q > 0:
        per_example[has_full_label] = generalized_cross_entropy(
            logits[has_full_label], targets[has_full_label], settings.robust_loss_q
        )
    elif has_full_label.any():
        per_example[has_full_label] = soft_cross_entropy(
            logits[has_full_label],
            targets[has_full_label],
            settings.label_smoothing,
            settings.focal_gamma,
        )
    if is_candidate_set.any() and not use_factorized_weak_loss:
        per_example[is_candidate_set] = partial_label_nll(
            logits[is_candidate_set], targets[is_candidate_set]
        )

    if use_factorized_weak_loss:
        counted = has_full_label.float()
    else:
        counted = (has_full_label | is_candidate_set).float()
    loss = (per_example * weights * counted).sum() / (weights * counted).sum().clamp_min(1e-6)

    if use_factorized_weak_loss:
        weak_outputs = {
            name: outputs[name][is_candidate_set]
            for name in ('log_song', 'log_complete', 'log_dialect')
        }
        weak_loss = factorized_weak_loss(
            weak_outputs, targets[is_candidate_set], weights[is_candidate_set]
        )
        loss = loss + settings.weak_loss_weight * weak_loss

    return loss


# ====================================================================================== helpers
def _loss_class_weights(train, power):
    """Per-class loss weights (7,) from the class counts of the hand-labelled clips, or None."""
    if power <= 0:
        return None
    labelled = train[train.pool == 'strong']
    counts = labelled.label.value_counts().reindex(CLASSES).fillna(1).values.astype(float)
    weights = counts**-power
    # scale so that the average labelled clip keeps a weight of 1
    weights = weights / (weights * counts).sum() * counts.sum()
    return torch.tensor(weights, dtype=torch.float32)


def _load_bank(path):
    """Load a bank of audio chunks (.npy object array) from the workspace, or None.

    Chunks of equal length are stacked into one array: the data-loader workers then share its
    memory, where a Python list of arrays would be copied into every worker.
    """
    if not path:
        return None
    chunks = list(np.load(workspace_path(path), allow_pickle=True))
    if len({len(chunk) for chunk in chunks}) == 1:
        return np.stack(chunks)
    return chunks


def _apply_environment_overrides(settings):
    """Let STRNADI_FOLDS / _THREADS / _WORKERS / _SEED override the script's settings."""
    settings = copy.copy(settings)
    if os.environ.get('STRNADI_FOLDS'):
        settings.folds = [int(fold) for fold in os.environ['STRNADI_FOLDS'].split(',')]
    if os.environ.get('STRNADI_THREADS'):
        settings.threads = int(os.environ['STRNADI_THREADS'])
    if os.environ.get('STRNADI_WORKERS'):
        settings.data_workers = int(os.environ['STRNADI_WORKERS'])
    if os.environ.get('STRNADI_SEED'):
        settings.seed = int(os.environ['STRNADI_SEED'])
    return settings


def _dry_run(run_name, make_model, settings):
    """Build the model and push one batch through it, without touching data or disk."""
    model = make_model()
    print(f'[dry run] {run_name}: {type(model).__name__}, {count_parameters(model):,} parameters')

    for share in settings.extra_tables.values():
        assert isinstance(share, float), 'extra_tables maps a table path to its share of draws'

    if settings.inputs == 'wave':
        example = torch.randn(2, settings.window_samples) * 0.1
    elif settings.inputs == 'perch_spatial':
        example = torch.randn(2, 16, 1536)
    else:
        example = torch.randn(2, 1536)
    with torch.no_grad():
        outputs = model.eval()(example)

    logits = outputs['logits']
    assert logits.shape == (2, len(CLASSES)), logits.shape
    if settings.bag_loss_weight > 0:
        assert 'log_dialect' in outputs, 'the bag loss needs a model with a factorised head'
    print(f'[dry run] forward pass ok, logits {tuple(logits.shape)}')


def _print_fold_result(fold, result):
    print(
        f'fold {fold}: macro-F1 {result["macro_f1"]:.3f}, '
        f'dialect accuracy given complete {result["dialect_acc_given_complete"]}, '
        f'complete-vs-Unfinished AUC {result["complete_vs_unfinished_auc"]}',
        flush=True,
    )
