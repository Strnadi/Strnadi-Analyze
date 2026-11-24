"""Tables for training a song detector: "is a Yellowhammer singing in this window?".

    python -m processing.build_detector_tables

Writes  data/detector_clips.csv    the audited clips with two labels, for cross-validation
        data/detector_extra.csv    further training windows

Only labels that do not come from our own models are used, so a detector trained on these tables
is trained from the source data, not from a larger model's output:

    song       every audited clip that is not "None"; the expert-annotated focal songs; the
               synthetic ending swaps and truncations (all contain a song)
    no song    the audited "None" clips; windows without a detected song from the citizen and
               focal recordings; windows where BirdNET hears another species and no Yellowhammer

The trainer and its metrics know the seven Strnadi classes, so the two detector labels are written
with two of those names: `None`, and `Unfinished` standing for "song". A detector's score is
`1 - p(None)`. Songs with only a file-level label (BirdNET-detected, never checked) are left out:
whether such a window holds a song is exactly what is not known.
"""

import pandas as pd

from config import workspace_path
from training.data import read_table

SONG = 'Unfinished'  # stands for "any Yellowhammer song" in the detector tables


def as_detector_label(label):
    return 'None' if label == 'None' else SONG


def main():
    clips = read_table('data/clips_audited.csv')
    clips['label'] = clips.label.map(as_detector_label)
    clips.to_csv(workspace_path('data/detector_clips.csv'), index=False)

    weak = read_table('data/weak_v5.csv')
    no_song_windows = weak[weak.label == 'None']
    other_species = read_table('data/hardneg_v1.csv')
    expert_songs = read_table('data/vizann_v1.csv')
    synthetic_songs = read_table('data/cf_v4.csv')
    songs = pd.concat([expert_songs, synthetic_songs]).assign(label=SONG)

    extra = pd.concat([no_song_windows, other_species, songs], ignore_index=True)
    extra['sw'] = 1.0
    extra.to_csv(workspace_path('data/detector_extra.csv'), index=False)

    print('cross-validated clips:', clips.label.value_counts().to_dict())
    print('extra training windows:', extra.label.value_counts().to_dict())


if __name__ == '__main__':
    main()
