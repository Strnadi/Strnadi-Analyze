# What is verified, what is not (2026-11-23)

## Verified by running it

| Check | How | Result |
|---|---|---|
| Every layer and model builds, runs forward and backward | `tests/test_models.py` (random audio) | 21 model variants + all front-end variants pass |
| The layer split changed nothing numerically | `tests/check_parity.py`: research checkpoints (`base_v3`, `r2_s0`) loaded into the refactored models, compared with their stored out-of-fold probabilities | max difference 5e-5, identical predictions |
| Mel filterbank without librosa | compared with `librosa.filters.mel` | < 1e-6 |
| Training (FFT) and export (conv) STFT paths agree | `tests/test_models.py`, `serving/export_onnx.py` | < 1e-3 / 4e-6 |
| Trainer, all optional paths | `tests/test_training.py` on synthetic data: weak pools, factorised weak loss, FixMatch, bag loss, frame loss, cached and online distillation, logit distillation, cosine margin, cached-embedding inputs, element and measurement losses, waveform-input models, no-inner-validation, resume after a kill | pass |
| Trainer on real data | 2 epochs x 4 steps of the R2 recipe on the real tables | runs; same pool shares (0.50 / 0.35 / 0.08 / 0.07) and parameter count (900,585) as the research runs |
| Relocated evaluation scripts | `summarize_cv`, `compare_runs` on the research runs | reproduce 0.774 and +0.029 [+0.012, +0.046] |
| ONNX export and the service pipeline | `r2_s0/fold0` exported (6.3 MB), `tests/check_serving.py` on a held-out Vizina recording | ONNX vs PyTorch 7e-7; 12 of 16 annotated songs found, all labelled correctly; silence and sub-window clips handled |
| Static check | `tests/check_names.py` | 0 undefined names / broken imports in all files |

## Not run (written or moved, compiles, static check passes)

- **Full trainings of anything.** No experiment was trained in this pass; the accuracy of every `proposed`
  model is unknown. The refactored trainer has not reproduced a research result end to end (4 CPU hours per fold).
- **Relocated scripts other than the two above** (`processing/*`, most of `evaluation/*`, `training/ssl_pretrain.py`,
  `training/train_identity.py`, `tools/sched.py`): imports and paths were rewritten mechanically and a few call
  sites by hand. They were not executed, because they write into the live workspace or run for a long time.
  `evaluation/explain.py` and `evaluation/qa_sheets.py` had the most hand edits.
- **New scripts:** `processing/clean_noise_bank.py`, `build_species_bank.py`, `build_disputed.py`,
  `make_folds.py` (rewritten), `cut_songs.py`; `evaluation/dialect_prior_shift.py`, `embed_songs.py`,
  `rescore_oof.py`, `plot_run.py`; `training/train_mil.py`, `train_unsupervised.py`.
- **`layers/perch_encoder.py`** (PyTorch port of Perch 2.0) and **`model/efficientat_temporal.py`**: written from the
  projects' READMEs; they need a download / a clone. Run `tests/check_perch.py` first. Fine-tunability of the
  Perch port is an assumption until that check passes.
- **Docker images, the FastAPI app under uvicorn, the Vast.ai worker, the label app.** The pipeline class the app
  calls is tested; the containers were not built. `vastai-serverless/main.py` still says "TFLite" in comments.
- **Pretrained-weight downloads** (torchvision / timm ImageNet weights): tests use random weights.

## Data that must be (re)built before the corrected experiments run

| Needed | Command |
|---|---|
| `data/noise_bank_clean.npy` | `python -m processing.clean_noise_bank --model runs/r2_s0`, then look at `figs/qa/noise_bank_dropped.png` |
| folds without duplicate recordings | `python -m processing.make_folds --table data/orig_aligned.csv --dedupe` (changes folds: old and new runs are then not comparable) |
| tables with song spans (frame targets) | re-run `processing/build_weak_set.py`, `build_jcu_set.py` |
| `data/species_bank.npy` | `python -m processing.build_species_bank` |
| `data/orig_aligned_disputed.csv` | `python -m processing.build_disputed base_v3 r1_s0 r2_s0` |
| Perch embeddings for weak tables | `processing/extract_foundation_emb.py` (TensorFlow environment) |

## Not materialised

- Element-frame table for the element tagger (needs the annotators' key from Vizina phrase types / components
  to B / C / D / E / X) and the measurement-target table (Table S1 frequency columns must be checked first).
- A recording -> observer table for observer-grouped folds.
- Unpadded originals of the BirdNET-app clips (source shortcut, review item 11).
- TFLite export for phones (ONNX only; `onnx2tf` + a parity check would be the next step).

## Removed from the repository (all in git history)

Keras / TFLite code replaced by the PyTorch equivalents: `model/{densenet,mobilenet,densenet_simclr,autoencoder,
perch_based,_perch}.py`, `layers/{arcface,attentive_stats_pool,densenet,global_gem2d,mel_to_magma,spec_augument}.py`,
`experiments/perch.py`, `convert.py`, `evaluation/{confusion-matrix,gradcam,history-plot,pca_tsne,tsne_tflite,
file_names}.py`, `processing/{split,encode}.py`, `vastai-serverless/runner.py`, the per-folder `requirements.txt`.
Replacements: `model/*`, `layers/*`, `experiments/legacy_*.py`, `serving/export_onnx.py`,
`evaluation/{plot_run,explain,map2d,summarize_cv}.py`, `processing/cut_songs.py`, `serving/app.py`.

The research session's `research/` folder (untracked, in the main checkout) was copied from, not moved: its
training jobs were still running from it.
