"""Rewrite the absolute audio paths stored in the workspace tables after moving the workspace.

Every table row points at its audio with an absolute path in the `npy` column (some tables also have
`path`). After copying the workspace to another machine those paths still name the old location.
This tool replaces the old prefix with the new one in every table under `$STRNADI_WS`.

    python -m tools.relocate_workspace /old/place/strnadi-ml            # new prefix = $STRNADI_WS
    python -m tools.relocate_workspace /old/place/strnadi-ml /new/place --dry

Tables handled: `data/*.csv`, `data/*.parquet`, `data/jcu_index/*.parquet`, `audio32k/manifest.csv`.
A copy of each changed file is kept next to it with the suffix `.before_relocate`.
"""

import argparse
import glob
import os
import shutil

import pandas as pd

from config import WS

PATH_COLUMNS = ('npy', 'path', 'src_file')


def read(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False, na_values=[''], low_memory=False)


def write(table, path):
    if path.endswith('.parquet'):
        table.to_parquet(path)
    else:
        table.to_csv(path, index=False)


def relocate(path, old_prefix, new_prefix, dry):
    """Replace the prefix in the path columns of one table; return the number of rows changed."""
    table = read(path)
    changed = 0
    for column in PATH_COLUMNS:
        if column not in table.columns:
            continue
        values = table[column].astype('string')
        hit = values.str.startswith(old_prefix, na=False)
        if hit.any():
            table.loc[hit, column] = new_prefix + values[hit].str[len(old_prefix) :]
            changed += int(hit.sum())

    if changed and not dry:
        shutil.copy(path, path + '.before_relocate')
        write(table, path)
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('old_prefix', help='where the workspace used to be')
    parser.add_argument(
        'new_prefix', nargs='?', default=WS, help='where it is now (default: $STRNADI_WS)'
    )
    parser.add_argument('--dry', action='store_true', help='only report what would change')
    arguments = parser.parse_args()

    old_prefix = arguments.old_prefix.rstrip('/')
    new_prefix = arguments.new_prefix.rstrip('/')

    tables = sorted(
        glob.glob(f'{WS}/data/*.csv')
        + glob.glob(f'{WS}/data/*.parquet')
        + glob.glob(f'{WS}/data/jcu_index/*.parquet')
        + glob.glob(f'{WS}/audio32k/manifest.csv')
    )
    total = 0
    for path in tables:
        try:
            changed = relocate(path, old_prefix, new_prefix, arguments.dry)
        except Exception as error:
            print(f'skipped {os.path.relpath(path, WS)}: {error}')
            continue
        if changed:
            print(f'{os.path.relpath(path, WS):50s} {changed} paths')
            total += changed
    print(f'{"would change" if arguments.dry else "changed"} {total} paths in {len(tables)} tables')


if __name__ == '__main__':
    main()
