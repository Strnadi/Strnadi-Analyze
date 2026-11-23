"""One experiment per file. Each file defines `EXPERIMENT`, a dict of overrides for `training/defaults.py` plus:

  description   what it is, in plain words
  hypothesis    the question it answers
  origin        legacy-repo | research-session | review-fork | review-base
  status        done | running | queued | proposed   (as of 2026-11-23, when this structure was written)
  result        measured numbers, where they exist
  runner        cv (default) | ssl | unsupervised | mil | identity
  needs         data / code that must exist first

Run:   python -m experiments.run <file stem> [--folds 0,1] [--seed 1] [--set key=value ...]
List:  python -m experiments.run --list
Shared building blocks are in `experiments/recipes.py`.
"""
