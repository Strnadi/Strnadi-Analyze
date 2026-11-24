"""Tables -> tensors.

A table row is one window of one cached recording:

    npy, t0, t1      32 kHz int16 mono array on disk, window start / end in seconds
    label            'BC'                     a hard label
                     'BC|Unfinished|None'     a candidate set (partial label)
                     'soft'                   a soft label in the columns p_<class> (pseudo-labels)
    fold, grp, rec   cross-validation fold, recording group, recording id
    source           'ncs', 'vizina', 'jcu', ...
    w, sw            loss weight, sampling weight
    song_t0, song_t1 where the song is in the file (for per-frame song-activity targets)
    elements         JSON [[name, start_s, end_s], ...] (for per-frame element tags)
    m_<name>         expert measurements of the song ending (for the measurement head)
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from config import C2I, CLASSES, DIALECTS, SR, WS, workspace_path
from processing.audio_repair import background_noise, fill_zero_runs

#: Frame targets are produced every 20 ms; the losses resample them to the model's frame rate.
FRAME_RATE = 50
#: Element tags, in the order used by model/element_tagger.py.
ELEMENTS = ['bg', 'phrase', 'B', 'C', 'D', 'E', 'X']
#: Measurement targets, in the order used by layers/measurement_head.py (table columns m_<name>).
MEASUREMENTS = ['f1', 'f2', 'f3', 'df', 'bw', 't2', 'has_first', 'has_second']

HARD_LABEL, CANDIDATE_SET, SOFT_LABEL = 0, 1, 2


def read_table(path, **read_csv_arguments):
    """Read a table from the workspace, keeping the literal label 'None' as a string."""
    return pd.read_csv(
        workspace_path(path), keep_default_na=False, na_values=[''], **read_csv_arguments
    )


def duplicate_recordings():
    """{recording id: id of the recording it is a byte-identical copy of}.

    Written by `processing/make_folds.py --dedupe` (data/duplicate_recordings.csv). The citizen
    archive holds ~90 groups of identical audio under different file ids; every table must put all
    copies into the same cross-validation fold.
    """
    path = workspace_path('data/duplicate_recordings.csv')
    if not os.path.exists(path):
        return {}
    table = pd.read_csv(path)
    return dict(zip(table.recording.astype(str), table.representative.astype(str)))


def load_seg(npy, t0, t1, n, raw=False):
    """Read `n` samples starting at `t0` seconds from a cached recording.

    The window is shifted to stay inside the file. Two repairs keep artificial silence out of the
    model's input (research log section 17.3; a dark block after a song reads as "no ending"):

    - runs of digital zeros of 10 ms or more (MP3 decoder delay, zero-padded excerpts) are filled
      with noise shaped like the recording's own background;
    - a file shorter than the window is padded with the same background noise. Never zeros, never
      reflection (reversed audio turns BhBl into a BlBh-like pattern).

    `raw=True` gives the loader as it was before 2026-11-23 (no zero filling, faint white-noise
    padding): the stored predictions of the runs up to R6 were made with it. `t1` is unused and kept
    for compatibility with the data-preparation scripts.
    """
    recording = np.load(npy, mmap_mode='r')
    start = int(round(t0 * SR))
    start = max(0, min(start, len(recording) - n))
    waveform = np.asarray(recording[start : start + n], dtype=np.float32) / 32768.0

    if raw:
        if len(waveform) < n:
            noise_level = float(np.std(waveform)) * 0.05 + 1e-5 if len(waveform) else 1e-4
            padding = np.random.default_rng(len(waveform)).normal(0, noise_level, n - len(waveform))
            waveform = np.concatenate([waveform, padding.astype(np.float32)])
        return waveform

    waveform = fill_zero_runs(waveform)
    if len(waveform) < n:
        waveform = np.concatenate([waveform, background_noise(waveform, n - len(waveform))])
    return waveform


def encode_label(row, candidate_escape_mass=0.0):
    """Turn a table label into a target vector over the classes.

    Returns:
        (target, kind) where kind is HARD_LABEL, CANDIDATE_SET or SOFT_LABEL.
    """
    label = row['label']

    if label == 'soft':
        target = np.array([row[f'p_{name}'] for name in CLASSES], np.float32)
        return target, SOFT_LABEL

    target = np.zeros(len(CLASSES), np.float32)

    if '|' in label:
        # A citizen recording's file label can be wrong, so classes outside the set keep a
        # small escape mass. Site-level labels (Vizina, JCU) are reliable and get none.
        if str(row.get('source', '')) == 'ncs':
            target[:] = candidate_escape_mass
        for name in label.split('|'):
            target[C2I[name]] = 1
        return target, CANDIDATE_SET

    target[C2I[label]] = 1
    return target, HARD_LABEL


class EmbeddingStore:
    """Cached Perch 2.0 embeddings of unaugmented clips, looked up by row name.

    Written by `processing/extract_foundation_emb.py`: one store for the labelled clips and,
    optionally, one for the rows of the weak tables.
    """

    def __init__(self, spatial=False):
        self.stores = []
        self.location = {}

        labelled = np.load(f'{WS}/emb/orig_perch.npz')
        labelled_names = read_table('data/orig_index.csv').name
        self.stores.append(labelled['spatial'] if spatial else labelled['emb'])
        for index, name in enumerate(labelled_names):
            self.location[name] = (0, index)

        kind = 'spatial' if spatial else 'emb'
        extra_path = f'{WS}/emb/rows_perch_{kind}.npy'
        extra_complete = os.path.exists(f'{WS}/emb/rows_perch_yh.npy')  # written last
        if os.path.exists(extra_path) and extra_complete:
            extra_names = pd.read_parquet(f'{WS}/data/perch_targets_rows.parquet').name
            self.stores.append(np.load(extra_path, mmap_mode='r'))
            for index, name in enumerate(extra_names):
                self.location[name] = (1, index)

    def get(self, name):
        """Return (embedding, found) where found is 1.0 or 0.0."""
        if name not in self.location:
            empty = np.zeros(self.stores[0].shape[1:], np.float32)
            return empty, 0.0
        store, index = self.location[name]
        return np.asarray(self.stores[store][index], np.float32), 1.0


class ClipDataset(torch.utils.data.Dataset):
    """Rows of a table as training / evaluation examples.

    Each example is a dict with `wave`, `target`, `kind`, `w` and, where the table and the settings
    provide them, `frames`, `elements`, `measure`, `teacher_emb`, `teacher_mask`.
    """

    def __init__(self, table, settings, train, augment=None, teacher_store=None, input_store=None):
        self.table = table.reset_index(drop=True)
        self.settings = settings
        self.train = train
        self.augment = augment
        self.teacher_store = teacher_store
        self.input_store = input_store

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        settings = self.settings

        target, kind = encode_label(row, settings.candidate_escape_mass)
        example = {
            'target': torch.from_numpy(target),
            'kind': kind,
            'w': float(row.get('w', 1.0)),
        }

        if self.input_store is not None:
            # The model is fed a cached embedding instead of audio.
            embedding, found = self.input_store.get(row['name'])
            if not found:
                raise KeyError(
                    f'no cached embedding for {row["name"]}: '
                    'run processing/extract_foundation_emb.py on this table first'
                )
            example['wave'] = torch.from_numpy(embedding)
            return example

        window_start = self._window_start(row)
        waveform = load_seg(
            row.npy, window_start, window_start + settings.window_seconds, settings.window_samples
        )
        if self.train and self.augment is not None:
            contains_song = target[C2I['None']] < 0.5  # hard label or candidate set without None
            waveform = self.augment(waveform, may_add_song=bool(contains_song))
        else:
            waveform = _peak_normalize(waveform)
        example['wave'] = torch.from_numpy(np.ascontiguousarray(waveform, np.float32))

        example['frames'] = torch.from_numpy(self._song_activity(row, window_start))
        if settings.element_loss_weight > 0:
            example['elements'] = torch.from_numpy(self._element_tags(row, window_start))
        if settings.measurement_loss_weight > 0:
            measurements = [float(row.get(f'm_{name}', np.nan)) for name in MEASUREMENTS]
            example['measure'] = torch.tensor(measurements, dtype=torch.float32)

        if self.teacher_store is not None:
            teacher_embedding, found = self.teacher_store.get(row['name'])
            example['teacher_emb'] = torch.from_numpy(teacher_embedding)
            example['teacher_mask'] = found
        return example

    def _window_start(self, row):
        """Where the analysis window starts inside the (possibly longer) table window."""
        slack = (row.t1 - row.t0) - self.settings.window_seconds
        if slack <= 0.05:
            return row.t0
        if self.train:
            return row.t0 + np.random.uniform(0, slack)
        if self.settings.eval_window == 'center':
            return row.t0 + slack / 2
        return row.t0

    def _n_target_frames(self):
        return int(self.settings.window_seconds * FRAME_RATE)

    def _song_activity(self, row, window_start):
        """Per-frame target: 1 inside the song, 0 outside, -1 where the song span is unknown."""
        n_frames = self._n_target_frames()
        if row['label'] == 'None':
            return np.zeros(n_frames, np.float32)

        song_start = row.get('song_t0', np.nan)
        song_end = row.get('song_t1', np.nan)
        if pd.isna(song_start) or pd.isna(song_end):
            return np.full(n_frames, -1.0, np.float32)

        first_frame = max(0, int((song_start - window_start) * FRAME_RATE))
        last_frame = int(np.ceil((song_end - window_start) * FRAME_RATE))
        last_frame = max(0, min(n_frames, last_frame))
        activity = np.zeros(n_frames, np.float32)
        activity[first_frame:last_frame] = 1
        return activity

    def _element_tags(self, row, window_start):
        """Per-frame element index; frames outside every element are background (0).

        Rows without an `elements` column are all -1 and ignored by the loss.
        """
        n_frames = self._n_target_frames()
        spans = row.get('elements', None)
        if not isinstance(spans, str) or not spans:
            return np.full(n_frames, -1, np.int64)

        tags = np.zeros(n_frames, np.int64)
        for name, start_s, end_s in json.loads(spans):
            first_frame = max(0, int((start_s - window_start) * FRAME_RATE))
            last_frame = int(np.ceil((end_s - window_start) * FRAME_RATE))
            last_frame = max(0, min(n_frames, last_frame))
            tags[first_frame:last_frame] = ELEMENTS.index(name)
        return tags


class UnlabeledDataset(torch.utils.data.Dataset):
    """Two views of an unlabelled song for FixMatch.

    weak view   = a random crop, peak-normalised (the teacher labels this one)
    strong view = the same crop through the full augmentation chain with noise always on
    """

    def __init__(self, table, settings, augment):
        self.table = table.reset_index(drop=True)
        self.settings = settings
        self.augment = augment

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        waveform = _random_crop(row, self.settings)
        weak_view = _peak_normalize(waveform)
        strong_view = self.augment(waveform, force_noise=True)
        return torch.from_numpy(weak_view), torch.from_numpy(strong_view)


class BagDataset(torch.utils.data.Dataset):
    """Bags of `bag_size` songs from one recording that carries a recording-level dialect.

    Used by `losses/bag_complete.py` ("at least one song of this recording is a complete song of
    its dialect"). Only recordings with at least two songs and exactly one dialect in their
    candidate set are used.
    """

    def __init__(self, table, settings, augment):
        is_candidate_set = table.label.str.contains('|', regex=False)
        table = table[table.source.isin(settings.bag_sources) & is_candidate_set].copy()

        table['dialects'] = table.label.map(
            lambda label: [name for name in label.split('|') if name in DIALECTS]
        )
        table = table[table.dialects.map(len) == 1]

        self.recordings = [songs for _, songs in table.groupby('rec') if len(songs) >= 2]
        self.settings = settings
        self.augment = augment

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        songs = self.recordings[index]
        bag_size = self.settings.bag_size
        chosen = songs.sample(bag_size, replace=len(songs) < bag_size)

        waveforms = [self.augment(_random_crop(row, self.settings)) for row in chosen.itertuples()]
        dialect = DIALECTS.index(chosen.iloc[0].dialects[0])
        return torch.from_numpy(np.stack(waveforms)), dialect


def _random_crop(row, settings):
    """A window of `window_seconds` at a random position inside the row's table window."""
    slack = max(0.0, row.t1 - row.t0 - settings.window_seconds)
    start = row.t0 + np.random.uniform(0, slack)
    return load_seg(row.npy, start, start + settings.window_seconds, settings.window_samples)


def _peak_normalize(waveform, peak=0.5):
    current_peak = np.abs(waveform).max()
    if current_peak <= 1e-6:
        return waveform.astype(np.float32)
    return (waveform / current_peak * peak).astype(np.float32)
