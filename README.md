# Strnadi-Analyze

Machine learning for the Strnadi project (Yellowhammer dialects, citizen science): models, training,
evaluation, data preparation and the classification service. Everything is PyTorch; the service
runs an exported ONNX model.

Classes: `BC, BE, BhBl, BlBh, None, Unfinished, XB`. Every model takes a mono 32 kHz waveform.

## How the repository is organised

| Folder | What is in it |
|---|---|
| `experiments/` | **One self-contained script per experiment**, in notebook style (`# %%` cells): header with question / status / result, settings, `make_model()`, training, evaluation |
| `model/` | One model per file, assembled from layers. Each file can be run for a quick look: `python -m model.strnadnet` |
| `layers/` | One layer variant per file: front-ends, conv blocks, temporal modules, pooling, heads |
| `augment/`, `losses/` | One waveform augmentation / one loss per file |
| `training/` | What the experiment scripts call: `settings.py` (every setting with its default), `fit.py` (cross-validated training), `data.py`, plus `pretrain.py`, `unsupervised.py`, `bird_level.py`, `identity.py` |
| `evaluation/` | `report.py` = the shared evaluation functions (validation, metrics, confusion matrix, learning curves, error sheets); further evaluation scripts |
| `processing/` | Data preparation: decoding, song segmentation, table builders, BirdNET / Perch extraction |
| `serving/` | ONNX export, song-level inference pipeline, FastAPI app (`runner.py` is the entry point) |
| `data-generation/`, `vastai-serverless/` | The labelling web app; the Vast.ai worker in front of the service |
| `tests/`, `tools/`, `docs/` | Tests and checks; small utilities; `docs/REVIEW.md` (findings), `docs/STATUS.md` (what is verified) |

`config.py` holds the class list and the paths. Data lives outside the repository, under
`$STRNADI_WS` (audio cache, tables, runs, results, figures) and `$STRNADI_SRC` (raw recordings).

## An experiment, top to bottom

```python
# %% [markdown]
# # r2_factorized_weak
# What it is, the question it answers, its status and measured result.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'r2_factorized_weak'
settings = TrainSettings(
    labelled_table='data/orig_index.csv',
    extra_tables={'data/weak_v3.csv': 0.35, 'data/cf_v3.csv': 0.08, 'data/vizann_v1.csv': 0.07},
    noise_bank='data/noise_bank_audiomoth.npy',
    noise_probability=0.5,
    ...
)

# %% Model
def make_model():
    return StrnadNet(head='factorized')

# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)          # metrics, per class, per source, complete-vs-Unfinished breakdown
plot_run(RUN)               # learning curves + confusion matrix
plot_error_examples(RUN)    # spectrograms of the most confident errors
```

To start a new experiment, copy the closest script, change the header, the settings and the model.

## Running things

All commands run from the repository root.

```bash
python -m tools.list_experiments                      # every experiment with status and origin
python -m experiments.r2_factorized_weak              # train all folds, then evaluate
STRNADI_FOLDS=0,1 STRNADI_THREADS=5 python -m experiments.r2_factorized_weak   # some folds only
STRNADI_SEED=1 python -m experiments.r2_factorized_weak                        # stored as ..._s1
STRNADI_DRY_RUN=1 python -m experiments.frontend_pcen # build the model, stop before training

python -m evaluation.report <run>                     # the evaluation cells for any finished run
python -m evaluation.compare_runs <runA> <runB>       # paired recording bootstrap
evaluation/post_eval.sh <run>                         # external test sets as well
python -m serving.export_onnx <run_dir> model.onnx    # export + parity check
docker compose up --build                             # service on :8000 (POST /classify)
```

The scripts are also plain notebooks: open one in VS Code / PyCharm / Jupytext with the repository
root as working directory and run it cell by cell. A finished fold is skipped when the script is run
again, and an interrupted fold resumes from its last epoch.

Tests:

```bash
python -m tests.test_models        # every layer and model builds and runs (no data needed)
python -m tests.test_training      # losses, augmentations, every trainer path (synthetic data)
python -m tests.test_experiments   # every experiment script, in dry-run mode
python -m tests.check_names        # static check of all files
python -m tests.check_parity       # research checkpoints reproduce their stored predictions
python -m tests.check_serving model.onnx
```

Code style: `uvx ruff format .` (settings in `pyproject.toml`).

## Conventions

- Shapes: waveform `(B, N)`; spectrogram `(B, C, F, T)`; frame series `(B, D, T)`.
- A model returns a dict: `logits` (may be log-probabilities) and, where available, `emb`, `att`,
  `frames`, `frame_logits`, plus model-specific entries (`log_complete`, `element_logits`, ...).
- Tables are CSV rows of windows: `npy, t0, t1, label` (+ `fold, grp, rec, source, w, sw,
  song_t0, song_t1`). A label is a class, a candidate set (`BC|Unfinished|None`) or `soft`.
- Splits are always by recording (or coarser), never by clip; the test fold is never used to choose
  anything.
- Look at spectrograms (`plot_error_examples`, `evaluation/qa_sheets.py`) of every new table,
  augmentation or surprising number before building on it.

## Where things stand

Read `docs/REVIEW.md` first. In short: dialects are separated ~97 % correctly once a song is
recognised as complete; almost all remaining error is complete-vs-Unfinished, a good part of it
label noise; the experiments marked `proposed` are aimed at that.
