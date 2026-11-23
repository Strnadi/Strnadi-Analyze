# Strnadi-Analyze

Machine learning for the Strnadi project (Yellowhammer dialects, citizen science): training, evaluation, data
preparation and the classification service. Everything is PyTorch; the service runs an exported ONNX model.

Classes: `BC, BE, BhBl, BlBh, None, Unfinished, XB`. Every model takes a mono 32 kHz waveform.

## Layout

| Folder | What is in it | Rule |
|---|---|---|
| `layers/` | building blocks: front-ends, conv blocks, temporal modules, pooling, heads | one layer variant per file |
| `model/` | complete models assembled from layers | one model per file; `model.build_model(name, ...)` |
| `augment/` | waveform augmentations and the pipeline that chains them | one augmentation per file |
| `losses/` | losses | one loss per file |
| `training/` | data loading, sampler, metrics, EMA, checkpoints, the grouped-CV trainer, SSL / MIL / identity trainers | |
| `experiments/` | every experiment, past and proposed, as a small config file with its hypothesis, status and result | one experiment per file |
| `processing/` | data preparation: decoding, song segmentation, table builders, BirdNET / Perch extraction, JCU download | |
| `evaluation/` | evaluation, comparison, calibration, explanation, QA sheets | |
| `serving/` | ONNX export, song-level inference pipeline, FastAPI app (`runner.py` is the entry point) | no torch at run time |
| `data-generation/` | the labelling web app | |
| `vastai-serverless/` | Vast.ai worker in front of the service | |
| `tests/` | smoke tests, parity checks, static name check | |
| `docs/` | `REVIEW.md` (findings and what was changed), `STATUS.md` (what is verified, what is not), `RESEARCH_LOG.md` | |

`config.py` holds the class list and the paths. Data lives outside the repository, under `$STRNADI_WS`
(caches, tables, runs, results) and `$STRNADI_SRC` (raw recordings).

## Running things

All commands run from the repository root.

```bash
python -m experiments.run --list                          # every experiment with origin and status
python -m experiments.run r2_factorized_weak --folds 0,1  # train (recording-grouped CV)
python -m experiments.run frontend_pcen --set epochs=12   # ad-hoc override
python -m evaluation.summarize_cv <run>                   # metrics incl. per-source and completeness breakdown
python -m evaluation.compare_runs <runA> <runB>           # paired recording bootstrap
evaluation/post_eval.sh <run>                             # the whole evaluation suite
python -m serving.export_onnx <run_dir> model.onnx        # export + parity check
docker compose up --build                                 # service on :8000 (POST /classify)

python -m tests.test_models      # every layer and model builds and runs (no data needed)
python -m tests.test_training    # losses, augmentations, every trainer path (synthetic data)
python -m tests.check_names      # static check of all files
python -m tests.check_parity     # research checkpoints reproduce their stored predictions (needs the workspace)
python -m tests.check_serving model.onnx
```

## Conventions

- waveform `(B, N)`; spectrogram `(B, C, F, T)`; frame series `(B, D, T)`.
- A model returns a dict: `logits` (may be log-probabilities), and where available `emb`, `att`, `frames`,
  `frame_logits`, plus model-specific entries (`log_complete`, `element_logits`, `measurements`, ...).
- Tables are CSV rows of windows: `npy, t0, t1, label` (+ `fold, grp, rec, source, w, sw, song_t0, song_t1`).
  A label is a class, a candidate set (`BC|Unfinished|None`) or `soft`.
- Splits are always by recording (or coarser), never by clip; the outer fold is never used to choose anything.
- Look at spectrograms (`evaluation/qa_sheets.py`) of every new table, augmentation or surprising number
  before building on it.

## Where things stand

Read `docs/REVIEW.md` first. In short: dialects are separated ~97 % correctly once a song is recognised as
complete; almost all remaining error is complete-vs-Unfinished, a good part of it label noise; the experiments
marked `proposed` are aimed at that.
