"""Every experiment script runs from top to bottom in dry-run mode.

    python -m tests.test_experiments

With STRNADI_DRY_RUN=1 the training functions build the experiment's model, push one random batch
through it and return, and the evaluation functions do nothing. That executes every line of every
script (imports, settings, `make_model`) without data, downloads or training.

Experiments whose model needs something external are only checked for syntax:
pretrained weights (torchvision / timm), the Perch port, or the EfficientAT clone.
"""

import ast
import glob
import os
import runpy

os.environ['STRNADI_DRY_RUN'] = '1'

import torch  # noqa: E402

EXPERIMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments'
)

#: Text in a script that means "building this model needs a download or external code".
NEEDS_EXTERNAL = ('pretrained=True', 'precomputed=False', 'EfficientATTemporal', 'TimmTemporal(')
#: Models that load pretrained weights by default.
PRETRAINED_BY_DEFAULT = ('DenseNet121AttnPool()', 'MobileNetV3Delta()')


def test_experiments():
    scripts = sorted(glob.glob(f'{EXPERIMENT_DIR}/*.py'))
    scripts = [path for path in scripts if not path.endswith('__init__.py')]
    assert len(scripts) > 40

    executed = 0
    for path in scripts:
        with open(path) as file:
            source = file.read()
        name = os.path.basename(path)

        ast.parse(source)
        assert source.startswith('# %% [markdown]'), f'{name}: no markdown header'
        assert '**Question.**' in source and '**Status:**' in source, f'{name}: incomplete header'

        if any(marker in source for marker in NEEDS_EXTERNAL + PRETRAINED_BY_DEFAULT):
            continue
        runpy.run_path(path, run_name='__main__')
        executed += 1

    print(f'{len(scripts)} experiment scripts parsed, {executed} executed in dry-run mode')


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_experiments()
