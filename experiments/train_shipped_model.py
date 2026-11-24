# %% [markdown]
# # train_shipped_model
#
# Trains the model that is deployed: the recipe of one cross-validated experiment, on the
# labelled clips of ALL folds, then exports it for the backend (ONNX).
#
# **Question.** None: this is production training, not an experiment. The expected accuracy of the
# result is the cross-validated accuracy of `RECIPE` (docs/RESULTS.md); nothing is held out here.
# The checkpoint is still chosen by the lowest loss on an inner validation set of recordings the
# model does not train on.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.train_shipped_model`. Writes
# `$STRNADI_WS/runs/<RECIPE>/final/` (weights, log, TensorBoard) and
# `$STRNADI_WS/export/strnadi_classifier.onnx`.

# %% Imports
import importlib
import os
import subprocess
import sys

from config import WS, dry_run
from training.fit import train_final_model

# %% Which recipe is shipped (an experiment module with `settings` and `make_model`)
RECIPE = os.environ.get('STRNADI_RECIPE', 'strnadnet_field_ready')
recipe = importlib.import_module(f'experiments.{RECIPE}')

# %% Train on all folds, then export and check the exported graph against PyTorch
if __name__ == '__main__':
    checkpoint = train_final_model(RECIPE, recipe.make_model, recipe.settings)
    if not dry_run():
        os.makedirs(f'{WS}/export', exist_ok=True)
        export = [
            sys.executable,
            '-m',
            'serving.export_onnx',
            checkpoint,
            f'{WS}/export/strnadi_classifier.onnx',
        ]
        subprocess.run(export, check=True)
