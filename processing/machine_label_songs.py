"""Label the detected songs that no person has labelled, with trained models and three checks.

    python -m processing.machine_label_songs --runs strnadnet_audited_labels element_sequence_ctc
                                             [--out data/songs_machine_labelled.csv]

Tens of thousands of Yellowhammer songs were cut from the citizen archive, the focal recordings,
the passive recorders and the BirdNET detections, but only 3.6k clips carry a human label. This
script lets the cross-validated models label the rest, and keeps a label only when all checks pass:

1. **confidence**: the averaged class probability is at least `--confidence` (default 0.9);
2. **candidate set**: the class is one the recording can contain (its file- or site-level label,
   e.g. `BC|Unfinished|None`); BirdNET detections have no such constraint;
3. **two readings agree**: the class head and the decoded element sequence (`losses/element_ctc`)
   name the same class.

Nothing existing is modified: the result is a new table. Rows keep the columns of the weak tables
and gain

    label_fold0 .. label_fold4   the label given by the teachers that did NOT train on that fold
    label                        the label given by all teachers together (for the final model)
    confidence, candidates       the ensemble confidence; the recording's candidate set

**Why one label column per fold.** A model evaluated on fold k must not learn from labels produced
by a teacher that saw fold k's clips. When the trainer builds fold k it therefore reads
`label_fold<k>` (written by the fold-k teachers, which never saw fold k), and drops the songs of
fold-k recordings as it does for every extra table. Only the final model, which has no test fold,
uses `label`.

The loop can be repeated: train with this table, then run the script again with the new runs as
teachers (`--out` another name) and compare; `tools/results_table.py` shows whether it still helps.
"""

import argparse

import numpy as np
import pandas as pd
import torch

from config import CLASSES, WS, effective_run_name, workspace_path
from losses.element_ctc import class_of_sequence, decode_tokens
from training.checkpoint import load_model
from training.data import duplicate_recordings, load_seg, read_table
from training.inference import peak_normalize

N_FOLDS = 5
WINDOW_SAMPLES = 128000


# --------------------------------------------------------------------------------- candidates
def unlabelled_songs(labelled_table):
    """Every detected song without a human label, with its candidate set and its fold."""
    weak = read_table('data/weak_v5.csv')
    weak = weak[weak.label.str.contains('|', regex=False)].copy()
    weak['candidates'] = weak.label

    detections = read_table('data/unlabeled_v4.csv')
    detections = detections[detections.source == 'birdnet_yh'].copy()
    detections['candidates'] = ''  # a BirdNET detection says "Yellowhammer", nothing about dialect
    detections['w'], detections['sw'] = 1.0, 1.0

    songs = pd.concat([weak, detections], ignore_index=True).drop_duplicates('name')

    # the fold of a song is the fold of its recording among the labelled clips, where it has one
    labelled = read_table(labelled_table)
    recording_fold = dict(zip(labelled.grp.astype(str), labelled.fold))
    same_audio = duplicate_recordings()
    recording = songs.rec.astype(str).str.replace('ncs:', '', regex=False)
    recording = recording.map(lambda name: same_audio.get(name, name))
    songs['fold'] = recording.map(recording_fold).fillna(songs.fold).astype(int)
    return songs.reset_index(drop=True)


class SongWindows(torch.utils.data.Dataset):
    """The centre 4 s of every table window, peak-normalised."""

    def __init__(self, songs):
        self.songs = songs

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, index):
        song = self.songs.iloc[index]
        start = song.t0 + max((song.t1 - song.t0) - 4.0, 0) / 2
        waveform = load_seg(song.npy, start, start + 4.0, WINDOW_SAMPLES)
        return torch.from_numpy(peak_normalize(waveform[None])[0])


# ----------------------------------------------------------------------------------- teachers
@torch.no_grad()
def teacher_readings(songs, runs, device, workers):
    """Class probabilities and decoded classes of every fold's teachers for every song.

    Returns:
        probabilities (n_folds, N, 7): mean over the runs' fold models;
        sequence_class (n_folds, N) of str: the class of the decoded element sequence, from the
        first run that has an element decoder ('' when none has).
    """
    teachers = {
        fold: [
            load_model(f'{WS}/runs/{effective_run_name(run)}/fold{fold}/model.pt').to(device)
            for run in runs
        ]
        for fold in range(N_FOLDS)
    }
    probabilities = np.zeros((N_FOLDS, len(songs), len(CLASSES)), np.float32)
    sequence_class = np.full((N_FOLDS, len(songs)), '', dtype=object)

    loader = torch.utils.data.DataLoader(SongWindows(songs), batch_size=256, num_workers=workers)
    first = 0
    for waveforms in loader:
        waveforms = waveforms.to(device)
        rows = slice(first, first + len(waveforms))
        for fold, models in teachers.items():
            for model in models:
                outputs = model(waveforms)
                probabilities[fold, rows] += torch.softmax(outputs['logits'], 1).cpu().numpy()
                decodes = 'token_log_probs' in outputs and sequence_class[fold, first] == ''
                if decodes:
                    sequences, _ = decode_tokens(outputs['token_log_probs'])
                    sequence_class[fold, rows] = [class_of_sequence(tokens) for tokens in sequences]
            probabilities[fold, rows] /= len(models)
        first += len(waveforms)
        if (first // 256) % 40 == 0:
            print(f'{first} / {len(songs)} songs read', flush=True)
    return probabilities, sequence_class


# ------------------------------------------------------------------------------------- checks
def checked_labels(probabilities, sequence_classes, candidates, min_confidence):
    """Apply the three checks. `sequence_classes` is (n_readings, N); all readings must agree.

    Returns:
        (labels, confidence): `labels[i]` is a class name, or '' where a check failed.
    """
    predicted = np.array(CLASSES)[probabilities.argmax(1)]
    confidence = probabilities.max(1)

    confident = confidence >= min_confidence
    in_candidate_set = np.array(
        [not allowed or name in allowed.split('|') for name, allowed in zip(predicted, candidates)]
    )
    has_decoder = (sequence_classes != '').all(0)
    readings_agree = (sequence_classes == predicted[None]).all(0) | ~has_decoder

    accepted = confident & in_candidate_set & readings_agree
    return np.where(accepted, predicted, ''), confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--labelled', default='data/clips_audited.csv')
    parser.add_argument('--out', default='data/songs_machine_labelled.csv')
    parser.add_argument('--confidence', type=float, default=0.9)
    parser.add_argument('--workers', type=int, default=12)
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    songs = unlabelled_songs(arguments.labelled)
    print(f'{len(songs)} unlabelled songs:', songs.source.value_counts().to_dict(), flush=True)

    probabilities, sequence_class = teacher_readings(songs, arguments.runs, device, arguments.workers)
    candidates = songs.candidates.values

    for fold in range(N_FOLDS):
        labels, _ = checked_labels(
            probabilities[fold], sequence_class[fold : fold + 1], candidates, arguments.confidence
        )
        songs[f'label_fold{fold}'] = labels
    songs['label'], songs['confidence'] = checked_labels(
        probabilities.mean(0), sequence_class, candidates, arguments.confidence
    )

    label_columns = ['label'] + [f'label_fold{fold}' for fold in range(N_FOLDS)]
    kept = songs[(songs[label_columns] != '').any(axis=1)]
    kept.to_csv(workspace_path(arguments.out), index=False)

    print(f'\n{len(kept)} of {len(songs)} songs got a label from at least one set of teachers')
    accepted = kept[kept.label != '']
    print(f'labelled by all teachers together: {len(accepted)}')
    print(pd.crosstab(accepted.source, accepted.label).to_string())
    print(f'written to {arguments.out}')


if __name__ == '__main__':
    main()
