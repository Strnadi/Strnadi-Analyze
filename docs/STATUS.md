# What is verified, what is not

**Update 2026-11-24.** Since the list below was written: the trainer was ported to the GPU and has
trained ~25 five-fold experiments (`docs/RESULTS.md`); `tests/test_training.py`,
`tests/test_models.py`, `tests/test_experiments.py` and `tests/check_names.py` pass on the GPU
machine; `data/noise_bank_clean.npy`, duplicate-free folds and `data/species_bank.npy` were built;
ONNX export was checked on the new checkpoints (ONNX Runtime vs PyTorch 2e-7). Still not run here:
TFLite conversion of the new models, the Docker images, the Perch and EfficientAT wrappers,
`training/pretrain.py`, `unsupervised.py`, `bird_level.py`.

The rest of this file is the state of 2026-11-23.

## Verified by running it

| Check | How | Result |
|---|---|---|
| Every layer and model builds, runs forward and backward | `tests/test_models.py` (random audio) | 21 model variants and all front-end variants pass |
| The layer split changed nothing numerically | `tests/check_parity.py`: research checkpoints (`base_v3`, `r2_s0`) loaded into the refactored models and compared with their stored out-of-fold probabilities | max difference 5e-5, identical predictions |
| The readability rewrite changed nothing numerically | each rewritten file in `layers/`, `model/`, `augment/`, `losses/` compared against its committed predecessor on the same inputs, weights and random seeds | identical outputs (see the commit message for the exact scope) |
| Mel filterbank without librosa | compared with `librosa.filters.mel` | < 1e-6 |
| Training (FFT) and export (conv) STFT paths agree | `tests/test_models.py`, `serving/export_onnx.py` | < 1e-3 / 4e-6 |
| Trainer, all optional paths | `tests/test_training.py` on synthetic data: weak pools, factorised weak loss, FixMatch, bag loss, frame loss, cached and online distillation, logit distillation, cosine margin, cached-embedding inputs, element and measurement losses, waveform-input models, no inner validation, resume after a kill, rebuilding a model from its checkpoint, the report functions | pass |
| Trainer on real data | 2 epochs x 4 steps of the R2 recipe on the real tables | runs; same shares of draws (0.50 / 0.35 / 0.08 / 0.07) and parameter count (900,585) as the research runs |
| Every experiment script | `tests/test_experiments.py`: all 54 parsed; 44 executed top to bottom in dry-run mode (model built, one batch pushed through) | pass; the other 10 need pretrained weights or external code |
| Shared evaluation functions on a real run | `python -m evaluation.report r2_s0` | reproduces macro-F1 0.774 and the per-class numbers; both figures inspected |
| Run comparison | `evaluation/compare_runs.py r2_s0 base_v3` | reproduces +0.029 [+0.012, +0.046] |
| ONNX export and the service pipeline | `r2_s0/fold0` exported (6.3 MB), `tests/check_serving.py` on a held-out Vizina recording | ONNX vs PyTorch 7e-7; 12 of 16 annotated songs found, all labelled correctly; silence and sub-window clips handled |
| Static check | `tests/check_names.py` | 0 undefined names / broken imports |
| Song events, weak and unlabelled tables | built from the real song index with `processing/song_events.py`, `build_weak_set.py`, `build_unlabeled.py` | identical to the research session's `song_events.parquet`, `weak_v5.csv` and `unlabeled_v4.csv` |
| Loader repairs (padding, digital silence) | `processing/audio_repair.py` through `training.data.load_seg` against the research loader on 240 real windows | identical samples |
| TFLite export | `serving/export_tflite.py` on `r2_s0/fold0`, then `tests/check_tflite.py` on 200 clips | float32 and float16: same class as PyTorch on every clip; int8: 99.5 % |

## Not run (written or moved; compiles; static check passes)

- **Full trainings of anything.** No experiment was trained in this pass; the accuracy of every
  `proposed` model is unknown. The refactored trainer has not reproduced a research result end to
  end (about 4 CPU hours per fold).
- **`training/pretrain.py`, `training/identity.py`, `training/unsupervised.py`,
  `training/bird_level.py`**: rewritten as importable functions; only their dry-run path (model
  construction) is exercised by the tests. `pretrain.py` and `identity.py` are readable rewrites of
  the research session's scripts and were not executed on data.
- **Relocated research scripts** in `processing/` and most of `evaluation/` (and `tools/sched.py`):
  imports and paths were rewritten, the formatter was applied, a few call sites were adapted by hand.
  They were not executed, because they write into the live workspace or run for a long time.
  **These files were only formatted, not rewritten for readability**: there is no test that would
  catch a behaviour change in a script that builds training data, so a hand rewrite was not worth the
  risk. They still use the research session's terse variable names.
- **New scripts:** `processing/clean_noise_bank.py`, `build_species_bank.py`, `build_disputed.py`,
  `make_folds.py` (rewritten), `cut_songs.py`; `evaluation/research_session/dialect_prior_shift.py`,
  `embed_songs.py`, `rescore_oof.py`.
- **`layers/perch_encoder.py`** (PyTorch port of Perch 2.0) and **`model/efficientat_temporal.py`**:
  written from the projects' READMEs; they need a download / a clone. Run `tests/check_perch.py`
  first. Fine-tunability of the Perch port is an assumption until that check passes.
- **Docker images, the FastAPI app under uvicorn, the Vast.ai worker, the label app.** The pipeline
  class the app calls is tested; the containers were not built.
- **Pretrained-weight downloads** (torchvision / timm ImageNet weights): tests use random weights.

## Data that must be (re)built before the corrected experiments run

| Needed | Command |
|---|---|
| `data/noise_bank_clean.npy` | `python -m processing.clean_noise_bank --model runs/r2_s0`, then look at `figs/qa/noise_bank_dropped.png` |
| folds without duplicate recordings | `python -m processing.make_folds --table data/orig_aligned.csv --dedupe` (changes the folds: old and new runs are then not comparable) |
| `data/song_events.parquet` | `python -m processing.song_events` (exists in the workspace; rebuild after the song index changes) |
| weak table with song spans (frame targets, ending-referenced noise) | `python -m processing.build_weak_set data/weak_v5.csv`: the same rows as the research table plus `song_t0` / `song_t1`. On the CPU server write to another name while the research jobs read `weak_v5.csv`. `build_jcu_set.py` for the JCU tables |
| `data/unlabeled_v4.csv` | `python -m processing.build_unlabeled` (exists in the workspace) |
| `data/species_bank.npy` | `python -m processing.build_species_bank` |
| `data/orig_aligned_disputed.csv` | `python -m processing.build_disputed base_v3 r1_s0 r2_s0` |
| Perch embeddings for weak tables | `processing/extract_foundation_emb.py` (TensorFlow environment) |

## Known small issues, not fixed

Found while rewriting `augment/` and `losses/`; none affects a finished result.

- `augment/noise_mix.py`: the `ending` reference still follows the loud phrase unless the faint
  ending supplies more than half of the active frames in the last 0.7 s (hence the raised SNR range).
- `losses/factorized_weak.py`: normalises by the weight of all weak rows, including rows without a
  dialect candidate (kept as in the research run R2, for comparability).
- `losses/soft_cross_entropy.py`: with `focal_gamma > 0` the focal factor applies to every class
  term with a non-zero smoothed target, not only the true class.
- `augment/codec_roundtrip.py`: a missing or failing ffmpeg silently turns the augmentation off.
- `augment/noise_mix.py::draw_chunk`: noise chunks shorter than the window are tiled.
- `losses/fixmatch.py`: thresholds live on the CPU (fine on this server, would fail on a GPU).

## Not materialised

- Element-frame table for the element tagger (needs the annotators' key from Vizina phrase types /
  components to B / C / D / E / X) and the measurement-target table (Table S1's frequency columns
  must be checked first).
- A recording -> observer table for observer-grouped folds.
- Unpadded originals of the BirdNET-app clips (source shortcut, review item 11).
- Full-integer (int8) quantisation with a calibration set, and latency on a real phone. The
  float16 TFLite export exists and is checked (`serving/export_tflite.py`).
- A learned detector for the end of a song. The energy rule often stops at the B element; the
  1.1 s tail kept by `processing/song_events.py`, `serving/pipeline.py` and `model/ending_zoom.py`
  is a patch (research log 17.4).

## Removed from the repository (all in git history)

Keras / TFLite code replaced by PyTorch equivalents: `model/{densenet,mobilenet,densenet_simclr,
autoencoder,perch_based,_perch}.py`, `layers/{arcface,attentive_stats_pool,densenet,global_gem2d,
mel_to_magma,spec_augument}.py`, `experiments/perch.py`, `convert.py`,
`evaluation/{confusion-matrix,gradcam,history-plot,pca_tsne,tsne_tflite,file_names}.py`,
`processing/{split,encode}.py`, `vastai-serverless/runner.py`, the per-folder `requirements.txt`.
Replacements: `model/*`, `layers/*`, `experiments/legacy_*.py`, `serving/export_onnx.py`,
`evaluation/{report,explain,map2d}.py`, `processing/cut_songs.py`, `serving/app.py`.

The research session's `research/` folder (untracked, in the main checkout) was copied from, not
moved: its training jobs were still running from it.
