"""Apply the visual label audit to the labelled clips and rebuild their folds.

    python -m processing.apply_label_audit

Reads   data/orig_aligned.csv                  the 4,089 labelled clips and where each one sits in
                                               its source recording
        label_audit/clip_label_audit.csv       (in the repository) one row per reviewed clip
Writes  data/clips_audited.csv                 clips with the audited label, duplicate-free folds
        data/clips_unjudgeable.csv             reviewed clips whose content cannot be decided
        data/duplicate_recordings.csv          byte-identical recordings (all tables share folds)

Columns added: `label_original` (the label as delivered), `audit` (not_reviewed / confirmed /
relabelled / unjudgeable). `label` is the audited label. The rules of the audit are in
docs/LABEL_AUDIT.md. Pre-baked augmented copies (`aug`) are dropped.

**The window of a clip is the clip as it was labelled**, not the window `realign_orig.py` re-cut
around the detected song. A label describes what the labeller saw. The re-cut windows broke that in
two ways the second audit round made visible: some were placed on a neighbouring sound, after the
song (the models then answered "None", correctly), and some revealed a terminal element beyond the
labelled clip's end (an "Unfinished" label on a complete song). Where the clip was located in its
source recording, the table window is the clip with 0.2 s of real audio on either side, which gives
training its random time shift; evaluation uses the centre, i.e. exactly the labelled 4 s.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import WS, workspace_path
from processing.make_folds import RANK, duplicate_groups

CLIP_SECONDS = 4.0
CONTEXT_SECONDS = 0.2  # real audio kept on either side of a located clip

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_TABLE = os.path.join(REPOSITORY, 'label_audit', 'clip_label_audit.csv')


def audited_clips():
    """The labelled clips with the audit applied (still with the old folds)."""
    clips = pd.read_csv(workspace_path('data/orig_aligned.csv'), keep_default_na=False)
    clips = clips[~clips.aug].copy()
    audit = pd.read_csv(AUDIT_TABLE, keep_default_na=False).set_index('name')

    clips = labelled_windows(clips)

    clips['label_original'] = clips.label
    clips['audit'] = clips.name.map(audit.verdict).fillna('not_reviewed')
    relabelled = clips.audit == 'relabelled'
    clips.loc[relabelled, 'label'] = clips.name[relabelled].map(audit.label_audited)
    return clips


def labelled_windows(clips):
    """Point every row at the clip as it was labelled (see the module docstring).

    Rows located in their source recording with a correlation of at least 0.9 become a 4.4 s window
    of that recording centred on the clip; all others keep the delivered 4 s clip file.
    """
    # recording id (e.g. 'F000090') -> its decoded file, whatever the original extension was
    archive = {
        os.path.basename(path).split('.')[0].upper(): path
        for path in glob.glob(f'{WS}/audio32k/ncs/*.npy')
    }
    clip_files = pd.read_csv(workspace_path('data/orig_index.csv'), keep_default_na=False)
    clip_file_of = dict(zip(clip_files.name, clip_files.npy))

    files, starts, ends = [], [], []
    for clip in clips.itertuples():
        correlation = pd.to_numeric(clip.align_corr, errors='coerce')
        offset = pd.to_numeric(clip.src_off, errors='coerce')
        source = archive.get(clip.name.split('.')[0].upper())
        located = clip.src == 'ncs' and source is not None and correlation >= 0.9

        if located:
            files.append(source)
            starts.append(offset - CONTEXT_SECONDS)
            ends.append(offset + CLIP_SECONDS + CONTEXT_SECONDS)
        else:
            files.append(clip_file_of[clip.name])
            starts.append(0.0)
            ends.append(CLIP_SECONDS)

    clips['npy'], clips['t0'], clips['t1'] = files, np.round(starts, 4), np.round(ends, 4)
    clips['in_source'] = [end - start > CLIP_SECONDS for start, end in zip(starts, ends)]
    return clips


def assign_folds(clips, n_folds=5, n_tries=20):
    """Recording-grouped folds; byte-identical recordings count as one recording.

    Groups are stratified on their rarest label; of `n_tries` random splits the one with the most
    even class counts per fold is kept.
    """
    same_audio = duplicate_groups()
    duplicates = pd.Series(same_audio).rename('representative').rename_axis('recording')
    duplicates.to_csv(workspace_path('data/duplicate_recordings.csv'))

    is_app_clip = clips.src == 'birdnet_mobile'
    recording = clips.group.where(~is_app_clip, clips.group.str.replace(r'_\d+$', '', regex=True))
    clips['grp'] = recording.map(lambda name: same_audio.get(str(name).split('.')[0].upper(), name))

    rarest_label = clips.groupby('grp').label.agg(lambda labels: min(labels, key=RANK.get))
    groups, group_labels = rarest_label.index.values, rarest_label.values

    best_spread, best_folds = None, None
    for seed in range(n_tries):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_of_group = {}
        for fold, (_, held_out) in enumerate(splitter.split(groups, group_labels, groups)):
            fold_of_group.update({groups[index]: fold for index in held_out})
        counts = pd.crosstab(clips.label, clips.grp.map(fold_of_group))
        spread = ((counts.max(axis=1) - counts.min(axis=1)) / counts.mean(axis=1)).sum()
        if best_spread is None or spread < best_spread:
            best_spread, best_folds = spread, fold_of_group

    clips['fold'] = clips.grp.map(best_folds).astype(int)
    return clips


def main():
    clips = assign_folds(audited_clips())

    unjudgeable = clips.audit == 'unjudgeable'
    clips[unjudgeable].to_csv(workspace_path('data/clips_unjudgeable.csv'), index=False)
    clips[~unjudgeable].to_csv(workspace_path('data/clips_audited.csv'), index=False)

    print(clips.audit.value_counts().to_string())
    print(pd.crosstab(clips[~unjudgeable].label, clips[~unjudgeable].fold))


if __name__ == '__main__':
    main()
