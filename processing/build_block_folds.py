"""Folds that keep neighbouring archive recordings together: a stand-in for observer-grouped folds.

    python -m processing.build_block_folds [--block 25] [--out data/clips_audited_block_folds.csv]

Folds grouped by recording do not separate observers, and the same person often uploads several
recordings of the same birds. The honest check is to group the folds by observer, but the table
that links an archive recording to its observation (and so to its observer and place) covers only
about 630 recordings. As a stand-in, this script assumes that recordings with neighbouring archive
numbers were uploaded together: every block of `--block` consecutive numbers (F000000-F000024,
...) goes into one fold. Clips that are not from the archive keep their recording as the group,
and byte-identical recordings stay together.

Output: a copy of `data/clips_audited.csv` with new `fold` values and a `fold_group` column.
Nothing existing is modified.
"""

import argparse
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import workspace_path
from processing.make_folds import RANK
from training.data import duplicate_recordings, read_table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--block', type=int, default=25)
    parser.add_argument('--out', default='data/clips_audited_block_folds.csv')
    arguments = parser.parse_args()

    clips = read_table('data/clips_audited.csv')
    same_audio = duplicate_recordings()

    def group_of(recording):
        recording = same_audio.get(str(recording), str(recording))
        archive_number = re.fullmatch(r'F(\d+)', recording.split('.')[0].upper())
        if archive_number is None:
            return recording
        return f'block{int(archive_number.group(1)) // arguments.block:04d}'

    clips['fold_group'] = clips.grp.map(group_of)
    rarest_label = clips.groupby('fold_group').label.agg(
        lambda labels: min(labels, key=lambda label: RANK.get(label, 9))
    )
    best = None
    for seed in range(20):  # the most balanced of 20 stratified splits
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        groups, labels = rarest_label.index.values, rarest_label.values
        fold_of = {}
        for fold, (_, test) in enumerate(splitter.split(groups, labels, groups)):
            fold_of.update({groups[index]: fold for index in test})
        counts = pd.crosstab(clips.label, clips.fold_group.map(fold_of))
        imbalance = (counts.max(1) - counts.min(1)).div(counts.mean(1)).sum()
        if best is None or imbalance < best[0]:
            best = (imbalance, fold_of, counts)
    clips['fold'] = clips.fold_group.map(best[1])
    clips.to_csv(workspace_path(arguments.out), index=False)
    print(best[2])
    print(
        f'{clips.fold_group.nunique()} groups for {clips.grp.nunique()} recordings -> {arguments.out}'
    )


if __name__ == '__main__':
    main()
