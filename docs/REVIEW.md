# Review findings and what was done about them

Sources: two independent reviews of the research session of 2026-11-22/23 (its log is `RESEARCH_LOG.md`), each with
its own analysis of the out-of-fold predictions and spectrograms, plus a code/data audit and a literature check.
Numbers are recording-grouped 5-fold CV on the 3,744 human-labelled clips unless stated.

## 1. What the results say

- **Dialects are not the problem.** When label and prediction are both a complete song, the dialect is right
  97-98 % of the time. BhBl and BlBh are confused 1-2 times in 3,744 clips.
- **Completeness is.** Nearly every error is dialect <-> Unfinished: 59-69 of 144 BhBl clips are predicted
  Unfinished. Complete-vs-Unfinished AUC: BhBl 0.79-0.83, BlBh 0.84-0.89, BC 0.93, BE 0.95-0.97, XB 0.98.
- **Much of it looks like label noise.** ~310 errors are shared by four independently trained recipes; in 163
  clips their average confidently contradicts the label (236 clips get < 10 % for their own label over three runs).
  Spectrograms of those clips mostly show a first phrase and one terminal element, which the project's own rule
  calls Unfinished. Labellers seem to have tagged the bird's dialect rather than the window's content. Treating
  them as label errors moves macro-F1 from 0.78 to 0.85-0.88 (a sensitivity bound, not a result).
- **Recipes hit a ceiling early.** Held-out F1 is flat from epoch 4-8 while training loss keeps falling; the fold
  matters more than the recipe (0.72-0.80 across folds; the best recipe gains +0.029).
- **Only one recipe change is significant:** the factorised head (R2, +0.029 [+0.012, +0.046] over the baseline).

## 2. Defects, and their state in this repository

| # | Defect | State |
|---|---|---|
| 1 | Noise bank contains Yellowhammer songs (36 of 400 sampled chunks flagged; clear songs among them), mixed into half of all training clips | `processing/clean_noise_bank.py`; corrected recipes use `noise_bank_clean.npy` (**must be built**) |
| 2 | Noise level referenced to the loud first phrase; the terminal element is fainter | `augment/noise_mix.py` reference `ending` + SNR range raised to 6-26 dB. Partial: the energy rule finds the first terminal element, not the faintest one |
| 3 | Factorised-head weak loss gives no completeness supervision | `losses/bag_complete.py` + `experiments/bag_completeness.py`; `training/bird_level.py` |
| 4 | Checkpoint choice on a ~380-clip inner set is noise (argmax 0.757 vs last epoch 0.760) and costs 15 % of training data | default rule is the 3-epoch smoothed one; `experiments/fixed_schedule_no_inner_val.py` |
| 5 | Evaluation window ended 3.4 s after onset (documented as 3.8 s): long songs lost their ending | `eval_window='center'` (default); research runs reproducible with `'start'` |
| 6 | `compare_runs` mixed runs evaluated on different clip tables / windows; one-seed intervals read as significance | `evaluation/compare_runs.py` refuses mixed tables, reports completeness AUC, prints the seed caveat; `evaluation/rescore_oof.py` |
| 7 | Pretrained backbones trained with from-scratch settings in the architecture sweep | `backbone_lr_factor` (0.1 in the `arch_*` experiments) |
| 8 | Per-frequency median subtraction removes song energy when the song fills the window | `layers/log_percentile_norm.py`, `layers/pcen.py` + front-end experiments |
| 9 | Byte-identical recordings under different F-ids (107 groups, 81 across folds, some with conflicting labels) | `processing/make_folds.py --dedupe` (**folds must be rebuilt**) |
| 10 | Folds do not separate observers / sites (observer predicts dialect at 0.81 vs 0.69) | `make_folds.py --group_map`; needs a recording -> observer table that does not exist yet |
| 11 | BirdNET-app clips (395; 310 of 551 XB) are never Unfinished/None: source shortcut; they stay unaligned and synthetically padded | visible now (`metrics()` reports recall by source). **Not fixed**: needs the unpadded originals |
| 12 | "Passive recorder => BhBl" shortcut (all passive weak data is BhBl-only; the JCU metric rewards it) | `evaluation/dialect_prior_shift.py` measures it. **Not fixed** |
| 13 | Speed augmentation by linear interpolation (treble loss, images) | polyphase resampling (`augment/speed_perturb.py`) |
| 14 | Factorised head froze the batch size into the exported ONNX graph | fixed (`layers/factorized_head.py`); export check now uses a second batch size |
| 15 | Export checked only ONNX vs the export path | `serving/export_onnx.py` also checks export path vs training path. TFLite is not produced |
| 16 | Service padded short files with 10x more noise than training; `runner_v2` bout logic never evaluated | `serving/pipeline.py` = the evaluated song-level pipeline, training's padding rule |
| 17 | Vizina "held-out males" and JCU "held-out recorders" are not independent of weak training data (neighbours within 200 m, shared territories, recorders 120-170 m apart) | documented in `evaluation/post_eval.sh` output and the experiment files. **Not fixed** |
| 18 | 225 of 464 Vizina test "Unfinished" songs are first-phrase-only, which training treats as ambiguous | documented. **Not fixed** |
| 19 | Disputed clips trained as hard dialect labels | `processing/build_disputed.py` + `experiments/disputed_as_candidates.py` (stop-gap until expert relabelling) |
| 20 | The song detector often ends a song at the B element; the quiet terminal C / E follows 0.3-1.1 s later. Windows, crops and tests placed relative to the detected end lose it (found by the research session, log 17.4) | `processing/song_events.py` keeps 1.1 s after the end in every training crop; `serving/pipeline.py` starts long songs' windows later; `model/ending_zoom.py` crops to 1.2 s after the end. A patch: a learned end-of-song detector is still missing |
| 21 | About 11 % of weak-label windows cut a neighbouring song; BirdNET scores borrowed from the next song (research log 17.1-17.2) | `processing/song_events.py`; tables `weak_v5`, `unlabeled_v4` (reproduced exactly with this code) |
| 22 | Digital silence and near-silent padding in the model input (research log 17.3) | `processing/audio_repair.py`, used by the loader and by the service |
| 23 | `onnx2tf` overwrites the ONNX file it converts with a one-clip graph, which the backend cannot run in batches | `serving/export_tflite.py` converts a copy; `tests/check_tflite.py` checks the result |

Legacy code defects fixed in the port: SpecAugment masked to 0 dB (the loudest value after colour mapping) with
one mask per batch; ImageNet backbones fed un-normalised input; random by-file splits with a time-seeded split
and pre-baked augmented copies on both sides; SimCLR "views" that were the same waveform twice; autoencoder
decoder hard-coded to one input size; `fullPredictions` always `{}` and IndexError on empty results in
`runner.py`; `layers/arcface.py` empty; label app crashing on clips that are not exactly 4 s; the Vast.ai image
copying files that were not in the repository; time-stretching and synthetic-noise padding in `split.py`.

## 3. What would make it state of the art (not done yet)

1. **Expert relabelling** of the disputed clips (`results/disputed_for_experts.csv`) and a ~500-clip gold test
   set, with the protocol "label what is in the window; record the bird's dialect separately".
2. **A target-domain test set:** confirmed segments from the app database. There is still no fair comparison
   with the production model on phone audio.
3. **A strong teacher:** Perch 2.0 fine-tuned end to end (`experiments/perch_finetune_teacher.py`); frozen, with a
   temporal head, it already equals the from-scratch baseline (0.741 vs 0.745).
4. **Distillation into the small model** on identical augmented views over a long schedule
   (`experiments/distill_function_matching.py`), pseudo-labels only from that teacher.
5. **Completeness-specific modelling:** multi-resolution input, long-kernel stem, high-resolution zoom on the
   ending, element-sequence output, bag loss (the `proposed` experiments).
6. **A learned song detector** (end to end, missed detections are the largest loss: song recall 0.875).
7. **Bird-level output as the headline** (0.957 accuracy at 0.89 coverage for the baseline), aggregated over
   complete songs only, with abstention.
8. Deprioritise: self-supervised pretraining from scratch on CPU, the from-scratch patch transformer, more JCU
   batches (one dialect, one site).

Ideas recorded but not materialised as code: truncation counterfactuals for all dialects from citizen songs
(needs element positions: do after the element tagger); half-speed input to Perch (finer mel resolution in the
song band; a one-hour frozen-embedding test); probing earlier Perch layers; Xeno-canto (3,758 Yellowhammer
recordings, needs an API key) and the Pipek / Petruskova dialect labels (not public; ask).
