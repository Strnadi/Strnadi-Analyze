"""Shared constants and paths.

Paths come from environment variables, so nothing in the code is tied to one machine:

    STRNADI_WS    the workspace: audio cache, tables, runs, results, figures
    STRNADI_SRC   the raw source recordings
"""

import os

# ---------------------------------------------------------------------------------------- classes
CLASSES = ['BC', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XB']
C2I = {name: index for index, name in enumerate(CLASSES)}

DIALECTS = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
DIALECT_IDX = [C2I[name] for name in DIALECTS]  # [0, 1, 2, 3, 6]
NONE_IDX = C2I['None']
UNFINISHED_IDX = C2I['Unfinished']

# ------------------------------------------------------------------------------------------ audio
SR = 32000  # every model in this repository takes mono 32 kHz waveforms
WIN_S = 4.0  # default analysis window in seconds (a complete song is at most 3.8 s long)

# ------------------------------------------------------------------------------------------ paths
WS = os.environ.get('STRNADI_WS', os.path.expanduser('~/strnadi-ml'))
SRC = os.environ.get('STRNADI_SRC', os.path.expanduser('~/Strnadi-data'))
BIRDNET_DIR = os.environ.get(
    'BIRDNET_DIR', os.path.expanduser('~/.local/share/birdnet/acoustic-models')
)
PERCH_TF_DIR = os.environ.get(
    'PERCH_TF_DIR',
    os.path.expanduser(
        '~/.cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'
    ),
)


def workspace_path(path):
    """Resolve a path relative to the workspace (absolute paths are returned unchanged)."""
    return path if os.path.isabs(path) else os.path.join(WS, path)


def effective_run_name(run_name):
    """The run directory name: with STRNADI_SEED=n (n > 0) set, the suffix `_s<n>` is appended."""
    seed = os.environ.get('STRNADI_SEED', '')
    if seed and int(seed) != 0:
        return f'{run_name}_s{int(seed)}'
    return run_name


def dry_run():
    """True when STRNADI_DRY_RUN=1: experiment scripts then build their model and stop before training."""
    return os.environ.get('STRNADI_DRY_RUN', '') == '1'
