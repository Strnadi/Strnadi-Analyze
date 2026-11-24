"""From windows to songs: how should overlapping windows of a recording be combined?

    python -m evaluation.whole_recording <model.pt | run directory> [--hop 0.2]

The model classifies 4 s windows; a recording is scanned with overlapping windows, so every song
is seen by many of them. This script compares rules for turning those windows into one answer per
song, on the whole recordings of the held-out Vizina males (663 songs annotated by ornithologists:
complete BhBl or Unfinished).

The catch any rule has to deal with: a window that cuts the ending of a complete song shows an
unfinished song, and the model says so, correctly and confidently. Of the ~20 windows that overlap
a song at a 0.2 s hop, only those that contain its ending can see the dialect.

Rules compared (the annotation is used only to decide which windows belong to which song):

    most confident window     the deployed logic: the non-None window with the highest confidence
    mean                      plain average of the class probabilities
    song-weighted mean        average weighted by p(song)
    centred-weighted mean     average weighted by p(song) and by where the model's own temporal
                              attention peaks: full weight when the peak (the ending) sits well
                              inside the window, none when it is at the window's edge
    any complete window       a dialect if some window gives it >= 0.5, else Unfinished
    aligned window            one window starting 0.6 s before the annotated onset (needs a
                              detector that finds the onset; the upper reference)

It also reports how many annotated songs the scan finds at all (some overlapping window says
"song"), and how much of the audio far from every annotated song is called "song".

**Without the annotation** (`detect_songs`, what a service would run): every window that hears a
song votes for the moment its temporal attention peaks, i.e. where it believes the ending is.
Windows that see the same song vote for the same moment, so songs appear as peaks of the vote
curve; each peak becomes one song, labelled by the weighted mean of the windows that voted for it.
Reported: how many annotated songs get a detection, how those are labelled, and how many
detections per minute match no annotated song (unannotated background birds included).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from config import CLASSES, DIALECT_IDX, NONE_IDX, SR, UNFINISHED_IDX, WS
from training.data import read_table
from training.inference import TorchEnsemble

WINDOW_SECONDS = 4.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SR)


# ------------------------------------------------------------------------------------ scanning
@torch.no_grad()
def scan_recording(models, npy, hop_seconds, device):
    """Classify every window of a recording.

    Returns:
        (starts, probabilities, attention_peak): window start times (W,), class probabilities
        (W, 7) averaged over `models`, and the time inside each window where the temporal
        attention is highest (W,), in seconds.
    """
    audio = torch.from_numpy(np.load(npy).astype(np.float32) / 32768.0)
    hop = int(hop_seconds * SR)
    windows = audio.unfold(0, WINDOW_SAMPLES, hop)  # (W, samples)
    starts = np.arange(len(windows)) * hop_seconds

    probabilities, peaks = [], []
    for first in range(0, len(windows), 256):
        batch = windows[first : first + 256].to(device)
        batch = batch / batch.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * 0.5
        batch_probabilities, batch_attention = 0, 0
        for model in models:
            outputs = model(batch)
            batch_probabilities = batch_probabilities + torch.softmax(outputs['logits'], 1)
            batch_attention = batch_attention + outputs['att']
        probabilities.append((batch_probabilities / len(models)).cpu().numpy())
        frame_seconds = WINDOW_SECONDS / batch_attention.shape[1]
        peaks.append((batch_attention.argmax(1).float() * frame_seconds).cpu().numpy())
    return starts, np.concatenate(probabilities), np.concatenate(peaks)


# --------------------------------------------------------------------------------------- rules
def centred_weight(attention_peak):
    """1 when the attention peak is 1.2-3.2 s into the window, falling to 0 at 0.2 s and 3.9 s."""
    rising = np.clip((attention_peak - 0.2) / 1.0, 0, 1)
    falling = np.clip((3.9 - attention_peak) / 0.7, 0, 1)
    return np.minimum(rising, falling)


def song_answers(probabilities, attention_peak):
    """The answer of every rule for one song, given the windows that overlap it."""
    answers = {}
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    is_song = predicted != NONE_IDX

    if is_song.any():
        best = np.flatnonzero(is_song)[confidence[is_song].argmax()]
        answers['most confident window'] = predicted[best]
    else:
        answers['most confident window'] = NONE_IDX

    answers['mean'] = probabilities.mean(0).argmax()

    p_song = 1 - probabilities[:, NONE_IDX]
    answers['song-weighted mean'] = (probabilities * p_song[:, None]).sum(0).argmax()

    weights = p_song * centred_weight(attention_peak)
    if weights.sum() > 1e-6:
        answers['centred-weighted mean'] = (probabilities * weights[:, None]).sum(0).argmax()
    else:
        answers['centred-weighted mean'] = answers['song-weighted mean']

    dialect_evidence = probabilities[:, DIALECT_IDX]
    if dialect_evidence.max() >= 0.5:
        answers['any complete window'] = DIALECT_IDX[dialect_evidence.max(0).argmax()]
    elif is_song.any():
        answers['any complete window'] = UNFINISHED_IDX
    else:
        answers['any complete window'] = NONE_IDX
    return answers


# ------------------------------------------------------------------------- blind detection
def detect_songs(starts, probabilities, attention_peak, hop_seconds, min_votes=None):
    """Find songs in a scanned recording from the windows alone.

    Returns:
        list of (ending time in seconds, class index) for every detected song.
    """
    if min_votes is None:
        min_votes = (
            0.6 / hop_seconds
        )  # three full-weight windows at a 0.2 s hop, fewer when sparser
    p_song = 1 - probabilities[:, NONE_IDX]
    weights = p_song * centred_weight(attention_peak) * (p_song > 0.5)
    vote_time = starts + attention_peak
    n_bins = int(np.ceil((starts[-1] + WINDOW_SECONDS) / hop_seconds)) + 1
    bins = np.round(vote_time / hop_seconds).astype(int)

    votes = np.bincount(bins, weights=weights, minlength=n_bins)
    class_votes = np.zeros((n_bins, probabilities.shape[1]))
    np.add.at(class_votes, bins, probabilities * weights[:, None])

    half_width = max(
        1, int(round(0.4 / hop_seconds))
    )  # ~0.8 s: one song's votes fall into one bump
    kernel = np.ones(2 * half_width + 1)
    kernel[[0, -1]] = 0.5
    smoothed = np.convolve(votes, kernel, mode='same')
    smoothed_classes = np.stack(
        [
            np.convolve(class_votes[:, index], kernel, mode='same')
            for index in range(class_votes.shape[1])
        ],
        1,
    )

    detections = []
    reach = int(round(1.0 / hop_seconds))  # a peak must be the highest point within +-1 s
    for index in np.flatnonzero(smoothed >= min_votes):
        neighbourhood = smoothed[max(0, index - reach) : index + reach + 1]
        if smoothed[index] >= neighbourhood.max() and (
            not detections or index * hop_seconds - detections[-1][0] > 1.0
        ):
            dialect_or_unfinished = smoothed_classes[index].copy()
            dialect_or_unfinished[NONE_IDX] = 0
            detections.append((index * hop_seconds, int(dialect_or_unfinished.argmax())))
    return detections


# ---------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        'model', help='a checkpoint, or a run directory (its fold models are averaged)'
    )
    parser.add_argument('--hop', type=float, default=0.2, help='seconds between window starts')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = TorchEnsemble(arguments.model, device=device).models
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']

    answers, labels = [], []
    found, n_windows_far, n_song_far, seconds = 0, 0, 0, 0.0
    blind = {'matched': 0, 'correct': 0, 'correct_bhbl': 0, 'n_bhbl': 0, 'unmatched_detections': 0}
    for npy, recording_songs in songs.groupby('npy'):
        starts, probabilities, attention_peak = scan_recording(models, npy, arguments.hop, device)
        seconds += len(starts) * arguments.hop
        ends = starts + WINDOW_SECONDS
        is_song = probabilities.argmax(1) != NONE_IDX

        near_a_song = np.zeros(len(starts), bool)
        for song in recording_songs.itertuples():
            overlap = np.minimum(ends, song.t1) - np.maximum(starts, song.t0)
            overlapping = overlap >= 0.5
            near_a_song |= (ends > song.t0 - 2.0) & (starts < song.t1 + 2.0)
            if not overlapping.any():
                continue

            song_result = song_answers(probabilities[overlapping], attention_peak[overlapping])
            aligned = np.abs(starts - (song.t0 - 0.6)).argmin()
            song_result['aligned window'] = probabilities[aligned].argmax()
            answers.append(song_result)
            labels.append(CLASSES.index(song.label))
            found += bool(is_song[overlapping].any())

        # --- the same recording without the annotation
        detections = detect_songs(starts, probabilities, attention_peak, arguments.hop)
        used = np.zeros(len(detections), bool)
        for song in recording_songs.itertuples():
            is_bhbl = song.label == 'BhBl'
            blind['n_bhbl'] += is_bhbl
            hits = [
                index
                for index, (time, _) in enumerate(detections)
                if song.t0 + 0.3 <= time <= song.t1 + 0.6 and not used[index]
            ]
            if hits:
                used[hits[0]] = True
                blind['matched'] += 1
                right = CLASSES[detections[hits[0]][1]] == song.label
                blind['correct'] += right
                blind['correct_bhbl'] += right and is_bhbl
        blind['unmatched_detections'] += int((~used).sum())

        n_windows_far += int((~near_a_song).sum())
        n_song_far += int((is_song & ~near_a_song).sum())

    answers = pd.DataFrame(answers)
    labels = np.array(labels)
    result = {
        'model': arguments.model,
        'hop_seconds': arguments.hop,
        'n_songs': len(labels),
        'hours_scanned': seconds / 3600,
        'songs_found_by_the_scan': found / len(labels),
        'windows_far_from_annotated_songs_called_song': n_song_far / max(n_windows_far, 1),
        'rules': {},
    }
    print(
        f'{len(labels)} annotated songs in {seconds / 3600:.1f} h of recordings, hop {arguments.hop} s'
    )
    print(f'songs found by the scan: {result["songs_found_by_the_scan"]:.3f}')
    print(
        'windows at least 2 s from every annotated song that are called "song": '
        f'{result["windows_far_from_annotated_songs_called_song"]:.3f} (includes unannotated songs)'
    )
    print(f'\n{"rule":24s} accuracy   BhBl recall   Unfinished recall')
    for rule in answers.columns:
        predicted = answers[rule].values
        accuracy = float((predicted == labels).mean())
        recall = {
            name: float((predicted[labels == CLASSES.index(name)] == CLASSES.index(name)).mean())
            for name in ('BhBl', 'Unfinished')
        }
        result['rules'][rule] = {'accuracy': accuracy, 'recall': recall}
        print(
            f'{rule:24s} {accuracy:.3f}      {recall["BhBl"]:.3f}         {recall["Unfinished"]:.3f}'
        )

    minutes = seconds / 60
    result['without_annotation'] = {
        'song_recall': blind['matched'] / len(labels),
        'label_accuracy_of_found_songs': blind['correct'] / max(blind['matched'], 1),
        'bhbl_found_and_right': blind['correct_bhbl'] / max(blind['n_bhbl'], 1),
        'detections_matching_no_annotated_song_per_minute': blind['unmatched_detections'] / minutes,
    }
    print('\nwithout the annotation (attention voting):')
    for name, value in result['without_annotation'].items():
        print(f'  {name}: {value:.3f}')

    with open(f'{WS}/results/whole_recording.json', 'w') as file:
        json.dump(result, file, indent=1)


if __name__ == '__main__':
    main()
