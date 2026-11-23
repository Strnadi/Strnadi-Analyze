"""Shared constants. Paths come from the environment so nothing in the code is tied to one machine."""
import os

CLASSES = ['BC', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XB']
C2I = {c: i for i, c in enumerate(CLASSES)}
DIALECTS = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
DIALECT_IDX = [C2I[c] for c in DIALECTS]          # [0, 1, 2, 3, 6]
NONE_IDX, UNFINISHED_IDX = C2I['None'], C2I['Unfinished']

SR = 32000            # every model in this repo takes mono 32 kHz waveforms
WIN_S = 4.0           # default analysis window (a complete song is <= 3.8 s)

WS = os.environ.get('STRNADI_WS', os.path.expanduser('~/strnadi-ml'))        # workspace: caches, tables, runs, results
SRC = os.environ.get('STRNADI_SRC', os.path.expanduser('~/Strnadi-data'))             # raw source recordings
BIRDNET_DIR = os.environ.get('BIRDNET_DIR', os.path.expanduser('~/.local/share/birdnet/acoustic-models'))
PERCH_TF_DIR = os.environ.get('PERCH_TF_DIR', os.path.expanduser(
    '~/.cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1'))
