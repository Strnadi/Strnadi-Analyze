"""Find the songs of a recording from overlapping windows ("attention voting"). numpy only.

The model classifies 4 s windows. A recording is scanned with a window every `hop_seconds`; every
window that hears a song votes for the moment where its temporal attention peaks, which is where
the model believes the song's ending is. Windows that see the same song vote for the same moment,
so each song is a peak of the vote curve. A song's class probabilities are the weighted mean of
the windows that voted for it.

Why not "the most confident window": a window that cuts the ending of a complete song shows an
unfinished song, and says so with high confidence, out-voting the windows that see the ending. On
whole recordings of held-out males that rule recognises 57 % of the complete BhBl songs; the
weighted mean recognises 98.5 % with the same model, and the vote curve finds 99.5 % of all
annotated songs (docs/RESULTS.md, "From windows to songs").

Used by the service (`serving/pipeline.py`), by the evaluation on whole recordings
(`evaluation/whole_recording.py`) and by the harvest of new songs
(`processing/harvest_songs_by_scan.py`).
"""

import numpy as np

WINDOW_SECONDS = 4.0


def centred_weight(attention_peak):
    """How much a window's vote counts, from where its attention peaks inside the window.

    1 when the peak (the song's ending) is 1.2-3.2 s into the window, so that the whole song is
    inside; falling to 0 at 0.2 s and at 3.9 s, where the song is cut by the window's edge.
    """
    rising = np.clip((attention_peak - 0.2) / 1.0, 0, 1)
    falling = np.clip((3.9 - attention_peak) / 0.7, 0, 1)
    return np.minimum(rising, falling)


def detect_songs(starts, probabilities, attention_peak, hop_seconds, none_index, min_votes=None):
    """Find the songs in a scanned recording.

    Args:
        starts: (W,) start time of every window, seconds.
        probabilities: (W, n_classes) class probabilities of every window.
        attention_peak: (W,) time inside each window where the temporal attention is highest.
        hop_seconds: distance between window starts.
        none_index: index of the "no song" class.
        min_votes: total vote weight a peak needs; default 0.6 / hop_seconds (three full-weight
            windows at a 0.2 s hop).

    Returns:
        list of (ending time in seconds, class index, class probabilities) per detected song.
        The class is the most likely one other than "no song"; the probabilities are the weighted
        mean over the song's windows and still include "no song".
    """
    if min_votes is None:
        min_votes = 0.6 / hop_seconds
    p_song = 1 - probabilities[:, none_index]
    weights = p_song * centred_weight(attention_peak) * (p_song > 0.5)
    n_bins = int(np.ceil((starts[-1] + WINDOW_SECONDS) / hop_seconds)) + 1
    bins = np.round((starts + attention_peak) / hop_seconds).astype(int)

    votes = np.bincount(bins, weights=weights, minlength=n_bins)
    class_votes = np.zeros((n_bins, probabilities.shape[1]))
    np.add.at(class_votes, bins, probabilities * weights[:, None])

    half_width = max(1, int(round(0.4 / hop_seconds)))  # ~0.8 s: one song's votes form one bump
    kernel = np.ones(2 * half_width + 1)
    kernel[[0, -1]] = 0.5
    smoothed = np.convolve(votes, kernel, mode='same')
    smoothed_classes = np.stack(
        [np.convolve(column, kernel, mode='same') for column in class_votes.T], axis=1
    )

    detections = []
    reach = int(round(1.0 / hop_seconds))  # a peak must be the highest point within +-1 s
    for index in np.flatnonzero(smoothed >= min_votes):
        neighbourhood = smoothed[max(0, index - reach) : index + reach + 1]
        time = index * hop_seconds
        if smoothed[index] < neighbourhood.max() or (
            detections and time - detections[-1][0] <= 1.0
        ):
            continue
        mean_probabilities = smoothed_classes[index] / smoothed_classes[index].sum()
        song_classes = mean_probabilities.copy()
        song_classes[none_index] = 0
        detections.append((time, int(song_classes.argmax()), mean_probabilities))
    return detections
