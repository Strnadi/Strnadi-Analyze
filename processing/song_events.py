"""Song events and clean training windows (research log section 17.1-17.2).

The song index lists detected segments. Two defects of cutting a fixed window [onset-0.6, onset+3.8]
around each segment were found by looking at spectrograms on 2026-11-23:

- 10-20 % of the windows cut a neighbouring song or sound at their edge;
- the segmenter's song end is often the B element: the quiet terminal C / E element of BC / BE
  songs follows 0.3-1.1 s later and falls outside a tightly placed crop, which turns the song
  into "Unfinished".

This module merges segments into events and places one 4.4 s window per event such that

1. the event lies in the zone that every random 4 s training crop of the window contains,
2. preferably at least 1.1 s of audio follows the detected end inside every crop,
3. the window lies inside the file,
4. no other event of the same file is cut by a window edge (events fully inside are fine).

It also flags "borrowed" events: admitted only by the onset re-scoring (processing/rescore_yh.py)
while another event starts within 3 s, so the BirdNET score probably belongs to that next song.

    python -m processing.song_events [out.parquet]       # default: data/song_events.parquet

The table is read by processing/build_weak_set.py and processing/build_unlabeled.py.
"""

import sys

import numpy as np
import pandas as pd

from config import SR, WS

WINDOW = 4.4  # seconds cut per event; the trainer draws a random 4 s crop inside it
CROP = 4.0
TAIL_AFTER_END = 1.1  # seconds wanted after the detected end (terminal elements follow B)
MIN_SCORE = 0.3  # BirdNET Yellowhammer score a segment needs to count as a song


def events(segments, gap=0.6, max_len=3.8):
    """Merge consecutive segments of a file into song events.

    Two segments are merged when the gap between them is below `gap` seconds and the merged event
    stays within `max_len` seconds (the same rule as processing/realign_orig.py). The event keeps
    the columns of its first segment; `yh` and `yh_grid3` become the maximum over the members and
    `n_seg` counts them.
    """
    segments = segments.sort_values(['npy', 'seg_t0']).reset_index(drop=True)
    has_grid_score = 'yh_grid3' in segments
    merged = []
    for _, file_segments in segments.groupby('npy', sort=False):
        current = None
        for segment in file_segments.itertuples(index=False):
            if current is not None:
                close = segment.seg_t0 - current['seg_t1'] < gap
                new_end = max(segment.seg_t1, current['seg_t1'])
                if close and new_end - current['seg_t0'] <= max_len:
                    current['seg_t1'] = new_end
                    current['yh'] = np.nanmax([current['yh'], segment.yh])
                    current['n_seg'] += 1
                    if has_grid_score:
                        current['yh_grid3'] = np.nanmax([current['yh_grid3'], segment.yh_grid3])
                    continue
                merged.append(current)
            current = segment._asdict()
            current['n_seg'] = 1
        if current is not None:
            merged.append(current)
    return pd.DataFrame(merged)


def file_lengths(paths):
    """Length in seconds of each cached recording (reads only the .npy header)."""
    lengths = {}
    for path in paths:
        try:
            lengths[path] = np.load(path, mmap_mode='r').shape[0] / SR
        except Exception:
            lengths[path] = np.nan
    return lengths


def _first_uncrossed(candidates, other_starts, other_ends, window):
    """The first candidate window start whose two edges cut no other event, or None."""
    for start in candidates:
        end = start + window
        cut_at_start = (other_starts < start) & (other_ends > start)
        cut_at_end = (other_starts < end) & (other_ends > end)
        if not (cut_at_start | cut_at_end).any():
            return start
    return None


def place(song_events, lengths, window=WINDOW, pre=0.6, margin=0.1, step=0.05):
    """Choose the window start `win_t0` of every event (NaN when no clean placement exists).

    Candidates are tried in order of distance from the default start (onset - `pre`). Three ranges
    of window starts are tried, best first:

    1. the event is inside every 4 s crop and at least 1.1 s follows its detected end;
    2. the event is inside every 4 s crop (with 0.05 s to spare);
    3. the event is inside the window with `margin` to spare. Used only for songs too long for
       range 2, otherwise it would park short songs at the crop edge.

    Excerpts shorter than the window use the whole file, if the song lies fully inside it (the
    loader pads short files with background noise).

    Adds the columns `win_t0`, `moved` (placement differs from the default) and `borrowed`.
    """
    song_events = song_events.sort_values(['npy', 'seg_t0']).reset_index(drop=True)
    window_start = np.full(len(song_events), np.nan)
    borrowed = np.zeros(len(song_events), bool)
    slack = window - CROP  # how far a random crop can start after the window start

    for path, file_events in song_events.groupby('npy', sort=False):
        rows = file_events.index.values
        onsets = file_events.seg_t0.values
        ends = file_events.seg_t1.values
        file_length = lengths.get(path, np.nan)
        if 'yh_grid3' in file_events:
            grid_scores = file_events.yh_grid3.values
        else:
            grid_scores = np.full(len(file_events), 1.0)

        for k in range(len(rows)):
            onset, end = onsets[k], ends[k]
            others = np.ones(len(rows), bool)
            others[k] = False

            next_song_close = (others & (onsets > onset) & (onsets - onset <= 3.0)).any()
            if grid_scores[k] < MIN_SCORE and next_song_close:
                borrowed[rows[k]] = True

            if np.isfinite(file_length) and file_length < window:
                song_inside = onset >= margin and end <= file_length - margin
                other_cut = (others & ((onsets < 0) | (ends > file_length))).any()
                if song_inside and not other_cut:
                    window_start[rows[k]] = 0.0
                continue

            other_onsets, other_ends = onsets[others], ends[others]
            default_start = onset - pre
            latest_crop_safe = onset - slack - 0.05
            fits_crop_safe_zone = latest_crop_safe >= end - CROP + 0.05
            ranges = (
                (end - CROP + TAIL_AFTER_END, latest_crop_safe),
                (end - CROP + 0.05, latest_crop_safe),
                (end + margin - window, onset - margin),
            )
            for rank, (earliest, latest) in enumerate(ranges):
                if latest < earliest or (rank == 2 and fits_crop_safe_zone):
                    continue
                candidates = np.arange(earliest, latest + 1e-9, step)
                if earliest <= default_start <= latest:
                    candidates = np.append(candidates, default_start)
                inside_file = candidates >= 0
                if np.isfinite(file_length):
                    inside_file &= candidates + window <= file_length
                candidates = candidates[inside_file]
                candidates = candidates[np.argsort(np.abs(candidates - default_start))]
                chosen = _first_uncrossed(candidates, other_onsets, other_ends, window)
                if chosen is not None:
                    window_start[rows[k]] = chosen
                    break

    default_starts = song_events.seg_t0.values - pre
    song_events['win_t0'] = window_start
    song_events['moved'] = np.isfinite(window_start) & (
        np.abs(window_start - default_starts) > 1e-6
    )
    song_events['borrowed'] = borrowed
    return song_events


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else f'{WS}/data/song_events.parquet'
    segments = pd.read_parquet(f'{WS}/data/song_index.parquet')
    lengths = file_lengths(segments.npy.unique())
    placed = place(events(segments), lengths)
    placed['file_len'] = placed.npy.map(lengths)
    placed.to_parquet(out_path)

    songs = placed[((placed.seg_t1 - placed.seg_t0) <= 3.8) & (placed.yh >= MIN_SCORE)]
    usable = np.isfinite(songs.win_t0) & ~songs.borrowed
    summary = songs.assign(usable=usable).groupby('source')[['usable', 'moved', 'borrowed']].mean()
    print(summary.round(3).T)
    tail = (songs.win_t0 + CROP - songs.seg_t1)[usable & (songs.file_len >= WINDOW)]
    print(
        f'{int(usable.sum())} usable events of {len(songs)}; audio after the detected end that every '
        f'crop keeps: median {tail.median():.2f} s, at least 1.0 s for {(tail >= 1.0).mean():.0%}'
    )
    print('written', out_path)


if __name__ == '__main__':
    main()
