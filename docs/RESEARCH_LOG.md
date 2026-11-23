# Strnadi / BioAcu ML research log

Workspace: `$HOME/strnadi-ml` (data caches, runs, embeddings). Code: `research/`.
All numbers are **recording-grouped 5-fold CV** on the labelled clip set unless stated otherwise
(groups = NCS recording F-id or BirdNET-app device; pre-baked `aug_*` copies excluded from evaluation).

## 0. Findings about the existing pipeline (2026-11-22)

| Issue | Impact |
|---|---|
| All historical splits were random **by file** (`train_test_split(..., stratify=labels)`), never by recording. 429 recordings contribute clips to >1 class, 2337 groups for 4089 clips. | Reported val F1 (~85%) is optimistic. |
| 345 `aug_1_*` files are pre-baked augmented copies (random noise prefix) of other clips. | Copies of the same clip end up in train and test. |
| 70% of XB clips are BirdNET-app 3-s clips padded to 4 s with quiet synthetic white noise. | A model can learn "padding ⇒ XB" (source shortcut). |
| `processing/split.py` time-stretches clips to 5 s (rubberband). | Distorts syllable/element durations, which carry dialect information. |
| Training used peak-normalised librosa audio + noise padding; inference uses ffmpeg loudnorm + band-pass + zero padding. | Train/inference mismatch. |
| `runner.py`: `zip` iterator consumed by `max()`, so `fullPredictions` is always `{}`; IndexError on empty result. | Bugs. |
| Only stored result (`evaluation/results/DenseNet.png`, leaky split): acc ≈ 82.5%, macro recall ≈ 64.5% (BlBh 1/8, BhBl 8/17). | Baseline reference. |

## 1. Data inventory

| Source | Content | Labels | Hours |
|---|---|---|---|
| `new-data.zip` (Drive) | 4089 × 4 s clips, 7 classes | strong, human-revised per clip | 4.5 |
| NCS recordings `ncs_nahravky` (2011–2022 citizen science) | 3899 decoded files | recording-level dialect for 3333 (`dialects_filtered.csv`), segment table `nahravky.csv` (38k BirdNET cuts) | 155 |
| Vizina focal recordings 2019–2026 | 1123 WAV | all complete songs BhBl; 2434 songs annotated (62 males, phrase types A–F, components) | 98 |
| AudioMoth Vizina/Brdy sample (of the ~21 TB set) | 101 × 55 min | site-level: BhBl or unfinished only | 88 |
| BirdNET (TU Chemnitz) Yellowhammer detections 2024-10..2025-09 | 51,944 clips | none (BirdNET YH detections; CC BY-NC-SA) | 89 |
| Italian Alps passive (on NAS, not here) | 1–2k 5 s fragments | none yet | ~2–3 |

## 2. Frozen foundation embeddings (what the BioAcu web head-training uses)

Perch v2 (`perch_v2_cpu`, 5 s @32 kHz, clip centred & zero-padded to 5 s, peak 0.25):

| Head | macro-F1 | acc | bal-acc | dialect-F1 | BhBl rec | BlBh rec |
|---|---|---|---|---|---|---|
| logistic (pooled 1536-d) | 0.662 | 0.799 | 0.665 | 0.754 | 0.51 | 0.42 |
| cosine prototype (pooled) | 0.618 | 0.731 | 0.654 | 0.705 | 0.47 | 0.59 |
| MLP (pooled) | 0.691 | 0.833 | 0.663 | 0.761 | 0.47 | 0.44 |
| temporal conv+attention on `spatial_embedding` (16×1536) | **0.741** | **0.863** | 0.699 | **0.812** | 0.48 | 0.58 |

→ Confirms Ghani et al. 2023 (Sci Rep 13:22876): pooled bird-ID embeddings mix Yellowhammer dialects; keeping
Perch's time axis recovers +5–8 macro-F1 points. Pooling discards the element *order* that defines BhBl vs BlBh.

## 3. Production model on data it never saw (Vizina held-out individuals)

`good-model-2.tflite` (DenseNet121, 30.5 MB, 4 s @48 kHz). Vizina annotated songs, 17 held-out males
(201 complete BhBl, 469 unfinished). *aligned* = one 4 s window from 0.4 s before onset; *slide* = max over 5 windows.

| Model | protocol | BhBl recall | BhBl → Unfinished | Unfinished recall | acc |
|---|---|---|---|---|---|
| production (good-model-2) | aligned | 0.68 | 27% | 0.93 | 0.855 |
| production (good-model-2) | slide | 0.73 | 22% | 0.93 | 0.870 |

→ reproduces the field report: clear complete BhBl songs called "unfinished" about a quarter of the time.

## 4. NCS recording-level labels vs human-verified clip labels

Within NCS recordings labelled D, verified segments are: same dialect 33–58 %, Unfinished 38–62 %, another dialect
0–10 % (BlBh recordings: 10 %), None 0–5 %. 36 % of verified BhBl clips (21 % of BlBh) come from recordings whose
file label is a *different* dialect.  ⇒ file labels are usable only as noisy candidate sets
{D, Unfinished, None} + an escape mass ε for other dialects (robust partial-label loss), never as truth.

## 5. Knowledge-based parsing (own idea; mirrors expert measurement protocol of Diblíková et al. 2023)

Element tracker (dominant-peak tracking on per-frequency-denoised 5 ms spectrogram) + rules on the last two long
elements. Medians of measured features match the acoustic definitions (XB: X 0.22 s @7.3 kHz → B @4.7 kHz;
BhBl: 2nd element lower; BlBh: 2nd higher by ~400 Hz; BC/BE: 2nd element at ~7 kHz, falling/rising).
v0 hand rules: 62 % coverage but only 34 % agreement with verified labels (XB mis-parsed as BhBl because long X
elements are "high then low"). Parse noise is large → rules not yet usable as labels; kept as an analysis tool and
for element localisation.

## 6. Counterfactual song surgery (own idea)

From 297 complete BhBl songs of Vizina *training* males (annotated component boxes snapped to tracked elements):
`swap` (Bh↔Bl → synthetic **BlBh**), `trunc2` (2nd element replaced by level-matched post-song background →
**Unfinished**, B only), `trunc12` (both removed → Unfinished), plus `orig`. Targets the main confusion (BhBl/BlBh)
and the BhBl→Unfinished error directly; BlBh has only 81 real clips.

## 7. First trained model: StrnadNet baseline (strong labels only)

StrnadNet v2: 0.90 M params; log-mel 128 bins 1.5–11 kHz (32 kHz, 10 ms hop, per-frequency median normalisation);
stride-2 stem + 4 residual stages (16-32-64-128 ch) → flatten remaining 8 freq bins into channels (absolute frequency
kept) → 3 dilated temporal conv blocks (≈1.5 s receptive field) → attention statistics pooling → 128-d embedding.
AdamW 2e-3, cosine, EMA 0.998, label smoothing 0.05, sqrt class balancing, SpecAugment (narrow), ±2 % speed.

| Model | CV folds | macro-F1 | acc | dialect-F1 | BhBl rec | BlBh rec | None rec |
|---|---|---|---|---|---|---|---|
| base_v2 (final EMA, old protocol) | 0–1 | 0.756 | 0.877 | 0.812 | 0.44 | 0.58 | 0.31 |

External: Vizina held-out males (never trained on any Vizina data):

| Model | BhBl recall | Unfinished recall | acc |
|---|---|---|---|
| production good-model-2 (30.5 MB) | 0.68 | 0.93 | 0.855 |
| base_v2 2-fold ensemble (≈3.6 MB fp32) | **0.95** | **0.98** | **0.970** |

Production model on the *other* 41 Vizina males: BhBl recall 0.48 (aligned) / 0.52 (slide); base_v2: 0.895 / 0.890
(Unfinished 0.94, acc 0.925). Over all 58 males: BhBl recall production 0.55 vs base_v2 0.91. Across all
58 males the deployed model labels ~45 % of complete BhBl songs "Unfinished".

**Protocol change (2026-11-22, requested):** from base_v3 on, each fold carves a grouped 15 % *inner validation*
set out of its training groups; EMA weights are evaluated every epoch, the best inner-val checkpoint is kept
(`model.pt`, plus `last.pt`) and only that checkpoint is scored on the outer fold. Outer-fold curves are logged for
diagnostics only (never used for selection).

## 8. BioAcu: frozen Perch v2 as a per-project backbone

Few-shot (n clips/class, grouped, 5 folds × 3 seeds), macro-F1:

| n/class | 4 | 8 | 16 | 32 | 64 | all |
|---|---|---|---|---|---|---|
| logistic head | 0.37 | 0.44 | 0.47 | 0.49 | 0.52 | 0.65 |
| prototype head | 0.36 | 0.48 | 0.52 | 0.55 | 0.58 | 0.61 |

2-D maps of pooled Perch embeddings (fit on train folds, held-out fold shown): kNN dialect-F1 in 2-D —
PCA 0.15, LDA 0.34, UMAP 0.42, supervised UMAP 0.34; dialect silhouette ≈ 0 for all; HDBSCAN finds **no**
clusters. ⇒ An expert could not "draw a circle" around dialects on a Perch map; the platform needs a
within-species-aware embedding for such projects.

## 9. Data-usage ledger (kept current; nothing of value should stay unused)

| Source | Size | Status | Used as |
|---|---|---|---|
| `new-data.zip` labelled clips | 3,744 (+345 aug copies dropped) | ✅ | strong labels (CV) |
| `dataset-v2.zip`, `kuba-dataset.zip` (Drive) | 3,229 / 758 clips | ✅ checked | duplicates of new-data (≥99 %); nothing new |
| NCS recordings + `dialects_filtered.csv` | 155 h, 3,333 file labels | ✅ | weak candidate sets (39k songs), SSL, pseudo-labels |
| Diblíková 2023 Table S1 + WAVs (Zenodo 8092674) | 430 measured males | ✅ | expert BE/BlBh labels override file labels; 332-song expert OOF test; element timings |
| Vizina focal 2019–2026 + annotations | 98 h, 2,434 annotated songs, 62 males | ✅ | weak BhBl, held-out-male test, counterfactual surgery, identity (diarization) |
| AudioMoth sample (bhbl) | 88 h | ✅ | weak BhBl, noise bank, None windows |
| JCU AudioMoth share (Vižina/Brdy 2025, 35 recorders) | 76,037 files, 23 TB, 66k h | 🔄 batch 1 (1,412 dawn/dusk files) | weak BhBl, SSL, far-field robustness, None windows |
| BirdNET (Chemnitz) YH detections 2024-25 | 52k clips | ✅ | SSL, pseudo-labels, Perch-distillation targets |
| BioDCASE 2025 Task 3 YH (Zenodo 15228365) | 2.6 h, 209 songs × 7 distances | ⬇️ downloaded | far-field transfer-function augmentation; detection negatives |
| strnadi.cz observation export | 4,113 georeferenced dialects | ✅ | (spatial prior option); F-file↔ID anchors (~630 recordings geolocated) |
| Italian Alps passive data | 1–2k 5 s fragments | ⛔ need NAS location | open-set / domain-shift test, BN adaptation, pseudo-labels |
| App/production DB 2025–26 confirmed segments | ? | ⛔ need export | target-domain strong labels |
| Chemnitz offer: ~98k YH files (1.5 yr) | — | ❓ ask | more SSL data |
| Xeno-canto European Yellowhammers | — | ❓ needs API key | domain diversity for SSL |

## 10. New data & tooling (2026-11-22 evening, after server reboot)

- **JCU AudioMoth share** reached via the Synology Drive public link (read-only API): 76,037 hourly WAVs, 22.96 TB,
  66.4k h, 35 recorders, 2025-02-25…07-25. Batch pipeline (`scripts/jcu_pipeline.py`): 4 download streams (~7.6 MB/s
  each) → tonal-SNR event proposals → BirdNET only on candidate windows → keep song windows [onset−1.5, +4.5 s] +
  None windows (compact int16 store) → delete raw. Batch 1 = first 3 + last 1 file of every 14th day per recorder
  (1,412 files, 382 GB): ≈50 songs per dawn hour.  7 recorders held out for a label-free passive-domain test
  (**wrong-dialect rate**: any BC/BE/BlBh/XB prediction is an error at this BhBl-only site).
- **Diblíková 2023 Table S1**: 430 expert-measured males (BE 309, BlBh 116, transitional 5 after re-evaluation; BD
  dissolved, mostly → BlBh) incl. exact terminal-element timing and frequency contour. Used for: expert overrides of
  file labels (99 recordings had no file label), a 332-song **expert BE/BlBh OOF test** (windows anchored on the measured
  terminal element). F-file numbers are monotonic in strnadi.cz observation IDs (342/342 anchors) → ~630 recordings
  geolocated (partial geographic CV possible).
- **weak_v2**: weak-only NCS recordings now get a hashed CV fold (not "always train") → every weakly labelled
  recording is evaluable out-of-fold → **bird-level accuracy** on ~3.8k NCS recordings (`eval_recordings.py`).
- **Far-field profiles** from BioDCASE 2025 playback (same 209 songs at 6.5–200 m, forest/grassland): median song-band
  level −2 → −44 dB, SNR 30 → 3 dB, spectral tilt +2…+7 dB/kHz (sd 2–4) → `--p_far` augmentation.
- **Evaluation suite** (`post_eval.sh`): grouped-CV OOF (strong labels) · Vizina held-out males · expert BE/BlBh songs ·
  bird-level NCS · JCU held-out recorders (label-free).
- **Model zoo** (`strnadi_ml/zoo.py`): StrnadNet (conv/GRU/attention temporal), any timm CNN (ImageNet weights, OS=16,
  shared temporal head), patch transformer, 1-D TCN over pitch-contour features. Per-step train cost (1 thread, bs 8):
  contour 0.4 s · StrnadNet-0.3M 0.9 s · StrnadNet 2.0 s · ConvNeXt-atto 2.8 s · MobileNetV3 3.5 s · ResNet18 4.1 s ·
  EfficientNet-B0 5.6 s · patch-TF 6.0 s · DenseNet121 25 s.
- **Losses/options**: robust partial-label (ε escape mass for NCS), soft labels (noisy student), Perch-v2 embedding
  distillation, focal, sub-center cosine margin head, far-field augmentation, `torch.compile`.
- **Label-free pretraining** (`ssl_pretrain.py`): same-bird SupCon (Vizina positives = same ring-coded male across
  recordings/years), data2vec-style masked-frame prediction, Perch-2.0-style source prediction (42.7k sources),
  M2D-X-style denoising teacher. 112k songs, Vizina test males excluded.
- **Scheduler** (`scripts/sched.py` + `queue/jobs.txt`): FIFO, packs jobs onto free cores within the 45-core budget;
  every queued block is a stated hypothesis (H1…H10).

## 11. Independent review (2026-11-22 late) — verified issues and fixes

| # | Issue (verified) | Fix |
|---|---|---|
| V1 | JCU "held-out" recorder AUD27 = half of the local AudioMoth sample (weak train, noise bank, SSL) | JCU test recorders re-drawn from devices used nowhere else: AUD2, 3, 6, 12, 16, 22, 26 |
| V2/V3 | weak-only NCS songs/None windows had fold −1 in `song_index` → pseudo-labels/None windows could leak expert & bird-level test recordings | `fix_splits.py`: hashed folds written into song/none index (single source of truth); `weak_v3` |
| V4 | Vizina male codes non-canonical (`24.21(?)`, `?`, `unknown`); 177 uncoded files (12.7 h) defaulted to train | canonical codes; uncoded files → `exclude` (no training, no SSL); per-male bootstrap CI in end-to-end eval |
| V5 | counterfactual splice artefacts only in synthetic classes; cf dominated BlBh draws; hard-labelled extras inflated the strong pool | `cf_v3`: identical null splices in `orig`; random null splices (`--p_splice`) on all classes; explicit per-table sampling pools (`--pool_share`) |
| V6 | `runner_v2` bout aggregation merges sessions, max-over-windows turns Unfinished bouts into dialects; never evaluated | song-level inference N1 (`eval_e2e.py songlevel`), end-to-end evaluation on full recordings |
| V7 | SSL same-recording positives pull complete & unfinished songs together (the main error); corpus contained CV audio | `ssl_v2`: same-song positives, no source prediction, NCS-free corpus |
| V8 | BD mapped to BE\|BhBl (experts re-assign BD mostly to BlBh) | BD → BE\|BlBh |
| V9 | ~1.6k annotated Vizina songs of training males unused as labels | `vizann_v1` table (complete → BhBl, single-B → Unfinished, first-phrase-only → BhBl\|Unfinished) |
| V10 | low power (≈15 BlBh / fold), single seeds, best-of-many selection | 5 folds × 2 seeds for decisive comparisons; `compare_runs.py` paired recording-group bootstrap |
| minor | reflect padding (time-reversed audio) for windows at file edges; gain augmentation was a no-op | windows shifted inside the file, noise padding only for short files; gain removed |

Correction to §7: the Vizina BhBl gain (0.91 vs 0.55) holds on *clean focal* songs; on citizen clips OOF BhBl recall
is still 0.44 (95 % CI 0.31–0.57; 31/59 BhBl clips → Unfinished). The dialect problem on phone data is not solved yet.

**End-to-end (full recordings, Vizina held-out males, 663 annotated songs)** — song-level pipeline N1 + base_v2
(2 folds): song detection recall 0.875, BhBl end-to-end recall 0.88 (male-bootstrap 95 % CI 0.79–0.94), Unfinished
0.84, 1.65 false alarms/min (incl. possibly unannotated real songs). Most loss = missed detections, not labels.

**New factorised head (N2)**: p(song)·p(complete|song)·q(dialect|complete); weak recording labels train only q, weighted
by the detached p(complete) → weak labels can no longer drag songs into "Unfinished".

Revised queue (P1): base (5 folds × 2 seeds) · R1 = strong + weak_v3 + cf_v3 + vizann + noise + null splices ·
R2 = R1 + factorised head · R3 = R2 + far-field · seeds 1 for R1/R2/base · ssl_v2 (fixed); fine-tunes from ssl_v2
enqueued when it finishes; identity model at idle priority.

## 12. Results so far under the corrected protocol

**base_v3** (strong labels only, folds 0–2, inner-val checkpointing): macro-F1 0.745, acc 0.863, bal-acc 0.708,
dialect-F1 0.817; recall BC 0.82 · BE 0.85 · BhBl 0.49 · BlBh 0.54 · None 0.40 · Unfinished 0.93 · XB 0.92.
Reliability (cross-fitted): ECE 0.020 (0.019 after temperature scaling); split-conformal 90 % sets: coverage 0.90,
mean size 1.12, 88 % singletons; 95 %: size 2.0. Auto-accept capacity: 51 % of clips at 95 % accuracy, 26 % at 98 %.

**VQ tokens, stage 0** (k-means K=100 on 40 ms log-mel frames): song-ending unit signatures carry no dialect
information (ARI ≈ 0, kNN dialect-F1 0.22 = chance) → units must be learned (HuBERT iteration) — queued.

## 13. Own research backlog (ranked; executed without waiting for prompts)

Running/queued: R1 (fixed weak+cf+annotations), R2 (factorised head), R4 (FixMatch/FreeMatch on 72k unlabelled
songs), R5 (hard negatives + phone-codec round-trips), R3 (far-field), SSL: ssl_v2 (same-song contrastive + data2vec),
ssl_hub (HuBERT units), ssl_mae (masked AE), ssl_ae (bottleneck AE) → each followed by an R2 fine-tune + VQ discovery
on its features; seeds for the decisive comparisons; identity/diarization model.

Next (in order): (1) JCU batch-1 integration (weak BhBl + None + unlabelled pool, 7 held-out recorders);
(2) noisy student from the best fold teachers (fold-correct); (3) architecture shortlist on the winning recipe
(StrnadNet small/default/big, GRU/attention, contour-TCN, MobileNetV3, EfficientAT, DenseNet121 control);
(4) frozen backbones for BioAcu: BirdNET 3.0 (extracting), Bird-MAE, AVES + prototype probing, few-shot & 2-D maps;
(5) HuBERT iteration 2 + VQ discovery on learned units; (6) distillation of the best ensemble into one small student,
TFLite fp16/int8 export with parity tests, front-end cost reduction (on-device STFT / n_fft 512), streaming variant;
(7) label audit list for experts; (8) SAM optimiser, test-time BN adaptation, DTW-kNN contour baseline, recording-level
attention-MIL, spatial prior (backend, optional); (9) codec/IR/Freq-MixStyle device-robustness sweep.
Blocked on access: Italian Alps data (NAS location), app/production DB confirmed segments (best target-domain test).

## 14. First recipe results, 5 folds each (2026-11-23 morning)

| run | what changes | macro-F1 | acc | BhBl | BlBh | None | Unfin. | XB |
|---|---|---|---|---|---|---|---|---|
| base_v3 | strong labels only | 0.745 | 0.868 | 0.50 | 0.49 | 0.43 | 0.93 | 0.93 |
| r1_s0 | + weak_v3 / cf_v3 / vizann pools, AudioMoth noise, null splices | 0.754 | 0.867 | 0.50 | 0.53 | 0.48 | 0.93 | 0.92 |
| **r2_s0** | R1 + **factorised head** (weak rows train only q(dialect), weighted by p(complete)) | **0.774** | **0.874** | **0.55** | **0.65** | **0.50** | 0.92 | **0.95** |

Paired recording-group bootstrap (2088 recordings, 1000 resamples):
- **R2 vs base:** macro-F1 +0.029 [+0.013, +0.046], BlBh +0.15 [+0.07, +0.24], BhBl +0.05 [+0.01, +0.10], None +0.07, XB +0.02. All have P > 0.97.
- **R1 vs base:** +0.009 [−0.009, +0.026], not significant.
- **R2 vs R1:** +0.020 [+0.007, +0.034].

So the weak file-level data only pays off when it is not allowed to teach "this clip is a complete song of dialect X". The factorised head is the mechanism. Unfinished recall drops by 0.01, which is the price of that.

**End-to-end on full recordings of the 15 held-out Vizina males** (663 annotated songs; detection, alignment and labelling all included):

| pipeline | song recall | BhBl correct | Unfinished correct | false alarms/min |
|---|---|---|---|---|
| production good-model-2 + runner logic | 0.774 | 0.678 [0.53, 0.80] | 0.690 | 1.20 |
| song-level N1 + base_v2 (StrnadNet 0.9 M) | **0.875** | **0.879 [0.79, 0.94]** | **0.843** | 1.65 |

Two production males get 0 % BhBl. The worst male under N1 gets 50 %.

**VQ discovery on supervised features** (k-means K=100 on base_v3 fold-0 frame features): kNN dialect-F1 0.30 against 0.22 for log-mel units, ARI 0.02. Supervised frame features tokenise slightly better, but bag-of-ending-units still discards element order. Units need sequence modelling (HuBERT iteration 2 / n-gram signatures).

**Checkpoint-selection meta-analysis** (18 finished folds, outer F1 at the chosen epoch):

| rule | outer F1 |
|---|---|
| inner argmax (current) | 0.757 |
| centred 3-epoch moving average of inner F1 | 0.761 |
| last epoch | 0.760 |
| oracle | 0.776 |

Argmax occasionally picks a fluke early epoch (r1 fold 4, epoch 1: 0.684 against 0.757 smoothed). New runs now also write `model_s3.pt` / `oof_s3.npz` with the smoothed rule. Comparisons keep using `oof.npz`, the argmax rule, until the rule is validated on fresh runs.

**Engineering fixes:**
- R4 crashed at epoch 10 because the unlabelled loader was exhausted. It now uses a replacement sampler.
- The trainer writes `resume.pt` every epoch and resumes automatically after a crash or reboot (smoke-tested).
- The BirdNET 3.0 extraction via the `birdnet` package deadlocked for 6 h in its multiprocessing session. It was replaced by direct onnxruntime.

**JCU batch 1 integrated** (1412 files, 59k candidates, 22 GB compact store):
- `jcu_train.csv`: 38.4k weak songs `BhBl|Unfinished|None` (YH ≥ 0.3, full song inside the 4.4 s window) plus 3.3k None windows, from 28 recorders.
- `jcu_test.csv`: 12.9k songs from 7 held-out recorders.
- `unlabeled_v2.csv`: 77.5k songs (+5.2k JCU).
- R4 re-queued on v2. New R6 = R2 + JCU weak pool.
- Architecture sweep on the R2 recipe, 3 folds each, ImageNet init for timm models: ConvNeXt-atto, MobileNetV3-L, contour-TCN, RegNetY-004, StrnadNet-wide (3.5 M), EfficientNet-B0, DenseNet121 (production architecture control), PatchTF.

Training cost per step at batch 16 on 2 cores: contour 0.7 s · StrnadNet-1.5M 1.7 s · ConvNeXt-atto 2.6 s · MobileNetV3 3.5 s · RegNetY 3.6 s · StrnadNet-wide 4.1 s · PatchTF 6.3 s · EffNet-B0 7.0 s · DenseNet121 9.8 s. Single 4 s inference: 0.04 to 0.23 s.

**Expert evaluation was broken and is now fixed** (2026-11-23).
- **What was wrong.** `expert_eval.csv` placed each song window at Table S1 `time_01_b2` − 2.6 s. But `time_01_b2` is a 6.000 s placeholder in 384 of 430 rows. The spectrograms show the real songs 0.5–2 s earlier, so the windows cut off the song starts. That produced the earlier "BE/BlBh recall 0.25", which measured the window placement, not the model. The table is renamed `*.INVALID_placeholder_times.csv`.
- **What replaced it.** `eval_expert_birds.py` runs a bird-level out-of-fold test. Every detected complete song of each expert recording is scored with 3 song-aligned windows. The dialect is the mean over complete songs, and folds are asserted consistent with `weak_v3`. Training was not affected: the weak override rows always came from detected songs, never from the S1 times.

Results on 332 birds:

| run | BE recall (226) | BlBh recall (102) | BE/BlBh acc | binary BE-vs-BlBh |
|---|---|---|---|---|
| base_v3 | 0.951 | 0.765 | 0.893 | 0.945 |
| r1_s0 | 0.965 | 0.775 | 0.905 | 0.945 |
| **r2_s0** | **0.969** | **0.873** | **0.939** | **0.957** |
| original NCS file label | — | — | 0.863 | — |

R2 agrees with the experts more often than the original NCS file labels do.

**Baseline on other held-out sets** (base_v3, 5-fold ensemble):
- Vizina held-out males, aligned clips: BhBl recall 0.96, Unfinished 0.96. The production model scores 0.68 / 0.73 on the same males.
- NCS bird-level OOF: accuracy 0.957 at coverage 0.89 (BlBh 0.81).
- JCU held-out recorders: 2.1 % of songs get an impossible dialect. Of all songs, 78 % are predicted Unfinished and 18 % BhBl.

**BirdNET 3.0 as a frozen backbone** (same protocol as Perch, §8):

| head | BirdNET 3.0 | Perch v2 |
|---|---|---|
| logistic | 0.496 | 0.662 |
| MLP | 0.519 | 0.691 |
| temporal | 0.523 | 0.741 |

BirdNET 3.0 recalls BlBh only 0.10–0.15. Few-shot at 64 shots gives 0.35 against 0.58 for Perch. HDBSCAN finds no clusters. This is consistent with Ghani et al. 2023: species-classifier embeddings blur within-species variation. Caveat: with 3 s windows the temporal head sees only 3 steps. For BioAcu, Perch v2 is the frozen default, with our own encoders as the alternative.

**BirdNET step size** (the user's question):

| windows | recall @0.3 | recall @0.2 | windows run |
|---|---|---|---|
| 3 s grid (corpus pass `birdnet24/`) | 0.748 | 0.807 | 2226 |
| 0.5 s grid (old birdnet-segment branch) | 0.879 | 0.922 | 13293 |
| one onset-aligned window (JCU pipeline) | 0.733 | 0.822 | 663 |

The 0.5 s grid finds clearly more songs, so my earlier claim that one aligned window was as good was wrong. A multi-offset and false-positive check is running. Re-scoring of the 63k low-score segments and a JCU pipeline fix follow from it.

## 15. Visual QA of model inputs (2026-11-23), prompted by "have you inspected real spectrograms?"

Until now almost every decision rested on metrics. `qa_sheets.py <sheet>` now renders exactly what the model sees (log-mel 1.5–11 kHz, 128 bins, median-normalised, 4 s) for each data source, augmentation and error type. Sheets are in `figs/qa/`. The first pass found six real defects, each invisible in the metrics.

| # | where | what the sheet showed | fix |
|---|---|---|---|
| 1 | expert test set | Table S1 `time_01_b2` is a 6.000 s placeholder in 384/430 rows, so windows cut song starts | bird-level re-detected test (`eval_expert_birds.py`, §14) |
| 2 | `cf_v3` truncations (in R1/R2 training) | removed endings filled with near-black patches. The filler was RMS-matched over the full band (dominated by low-frequency wind) against the tiny inter-element gap, which gives a shortcut cue "black hole = Unfinished" | `cf_v4`: filler EQ'd per STFT bin to the window's own pre-song background (median power). The sheet shows the patches gone |
| 3 | strong NČS clips | 663 of 3349 (20 %) start mid-song and 52 lose the ending: the old cuts sat on BirdNET's 0.5 s grid, not on songs. Many R2 "errors" were these | `realign_orig.py`: clip located in its source recording (normalised correlation, median r = 1.000), merged tonal event found, window re-cut at onset − 0.6 s, 4.4 s long. 2,740 windows moved, row order kept (`orig_aligned.csv`, `--strong`) |
| 4 | 121 "Unfinished" strong clips | the song continues past the clip end in the source, sometimes into a full ending (F003070, F000320). "Unfinished" describes the cut, not the bird | kept on the original clips (true of the window), listed for expert audit |
| 5 | noise mixing (every run with `p_noise`) | SNR was relative to the whole clip's power. Faint songs disappeared under noise-bank birds while the label stayed a dialect | `--snr_mode band` (default): song-active power (loudest 20 % frames) vs noise power, both in 1.5–11 kHz. `clip` is kept for the R2-comparable ablations (R6 folds 1–4, R2 seed 1) |
| 6 | far-field aug (unused so far) | random tilt sign boosted highs by up to 30 dB. The measured "slope" is not monotonic in distance (noise-floor confound) | far-field v2: distance-scaled high-frequency loss above 3 kHz, exponential-decay reverberation (RT 0.1–0.8 s forest / 0.1–0.4 s grassland, D/R falls with distance), measured SNR floored at 5 dB, profiles ≤ ~50 m only (beyond that the song is unreadable, so the label becomes noise) |

Minor: speed perturbation and null splices padded short clips with `np.resize`, which wraps the clip start onto its end. They now pad with low-level noise. 58 strong clips (1.5 %) have zero-padded tails, not class-correlated.

Checked and fine:
- weak_v3 windows (full songs inside);
- JCU songs (mostly phrase-only, so the 78 % "Unfinished" prediction looks real);
- noise / codec / splice / speed visuals;
- cf swap variants (natural looking).

**R7** = R2 + fixes 2, 3 and 5 (+ far-field v2 for R3) is queued. The architecture sweep now runs on the R7 recipe with R7 StrnadNet as its control. R2 fold models will be re-scored on the aligned windows for a like-for-like comparison.

**BirdNET window placement** (pipeline-faithful: segmenter event first, then the YH score; false positives = tonal events inside 196 other-species windows):

| windows | recall @0.3 | FP @0.3 | windows run |
|---|---|---|---|
| 3 s grid | 0.741 | 0.051 | 2226 |
| 0.5 s grid | 0.855 | 0.235 | 13293 |
| 1 onset window | 0.739 | 0.041 | 631 |
| 3 onset windows (−1, −0.5, 0 s) | 0.816 | 0.071 | 1893 |
| 5 onset windows | 0.833 | 0.122 | 3155 |

The 0.5 s grid's extra recall costs 5× more false positives. 3 onset windows is the chosen scheme for re-scoring the 63k low-score corpus segments and for JCU batch 2+.

## 16. What the model looks at: Grad-CAM, occlusion, attention (R2, out-of-fold, 25 clips/class)

Tool: `explain.py maps <run>` and `explain.py patch <runs>`. Sheets are in `figs/explain/<run>_<class>.png` (spectrogram | Grad-CAM | occlusion | temporal attention, song span dashed). Occlusion uses 16 mel bins × 0.2 s patches filled with the clip's per-bin median. For the factorised head, the complete-vs-unfinished and the dialect decisions are also explained separately. Grad-CAM at the last 2-D stage is coarse and spills onto background, so **occlusion is the primary tool**.

**Findings:**
- **Dialect decisions read the ending.** Occlusion importance per second on the ending (last 0.7 s of the song) is 1.6–2.0× the clip average for BC/BE/BhBl/BlBh/XB, and 1.24 for Unfinished, which has no ending. In correct BhBl/BlBh examples the red patches sit on the Bh and Bl elements, and the temporal attention peaks on the final element.
- **Unfinished** is decided at the end of the phrase and the region right after it, where an ending would be. That is the right evidence.
- **[SUPERSEDED by §17.4: the 'outside' region contained the faint C/E terminal elements the segmenter misses]** **Context is used heavily.** Importance density on the whole song span is ~1× the clip average, so outside-song regions matter as much as the phrase. Part of that is legitimate (silence after the ending = complete). Errors lean on context more: 59 % of importance outside the song for wrong predictions against 48 % for correct ones. The sheets show other birds and second songs driving errors.
- **Low band (< 2.5 kHz, background):** 2–6 % of importance for dialect classes, 12 % for None and 8.5 % for Unfinished.
- **Zero-padded clip example:** F001790.4, BlBh predicted BC. Grad-CAM's hottest spot is inside the digital-silence tail. Realignment (§15) removes such windows.

**Black-patch shortcut test** (complete, correctly classified dialect songs; 0.5 s after the song end attenuated; base_v3 never saw counterfactuals, R2 trained on the cf_v3 dark-patch artefact). Flips to Unfinished:

| patch starts after song end | base −12 dB | base −20 dB | R2 −12 dB | R2 −20 dB |
|---|---|---|---|---|
| 0.05 s | 16 % | 26 % | 12 % | 24 % |
| 0.35 s | 7 % | 15 % | 7 % | 15 % |
| 0.65 s | 3 % | 5 % | 5 % | 8 % |
| control | 0 % | 0 % | 0 % | 0 % |

The decay with distance shows most of the effect is the faint tail of the last element being cut, which makes "Unfinished" the legitimate answer. R2 ≈ base, so the cf_v3 artefact was **not** learned as a shortcut to a measurable degree. Only a small residual (5–8 % at 0.65 s, −20 dB) reacts to unnatural dark patches.

**Actions:**
1. Re-run maps and the patch test for R7 (cf_v4 + aligned windows) and the best sweep architectures.
2. Add a context-dependence test: mask everything outside the song and measure the accuracy drop.
3. R8 candidate: other-species background mixing (hard-negative windows added at −5…+10 dB below the song) to reduce context-driven errors.

### 16.x Corpus re-scoring with onset3 BirdNET windows (2026-11-23)
`scripts/rescore_yh.py`: 63,238 song_index segments that the 3 s-grid pass scored < 0.3 were re-scored with 3 windows at onset −1.0/−0.5/0.0 s (max).
7,784 now pass yh ≥ 0.3: ncs 4,110, vizina 3,176, audiomoth_bhbl 498. `yh = max(old, new)`, old value kept in `yh_grid3`; backup `song_index.pre_rescore.parquet`.
Done in §17: QA found borrowed scores; rebuilt as weak_v5 / unlabeled_v4 via song_events.py.


## 17. Data hygiene round 2 + context test (2026-11-23 afternoon)

### 17.1 Re-scored songs: QA found borrowed scores
QA sheet `figs/qa/rescored_s{0,1}.png` (newly admitted songs): mostly real, complete Yellowhammer songs, but in about a
quarter of the windows the song starts 2-3 s into the clip.  The segment at the onset is another sound, and its score
was borrowed from the *next* song, which the onset+3 s BirdNET window reaches.  Measured: a following segment
starts within 3 s of the onset for 20.6 % of newly admitted segments, against 9.6 % of segments that already passed.
So the existing weak set also had windows cutting a neighbouring song at the edge.

### 17.2 `scripts/song_events.py` (events + clean window placement), used by build_weak_set / build_unlabeled
- **Events:** segments merged when gap < 0.6 s and the event stays <= 3.8 s (as realign_orig); yh = max over members.
- **Placement:** a 4.4 s window that
  - contains the event inside the crop-safe zone [w0+0.4, w0+4.0] (train_cv draws a random 4 s crop);
  - preferably keeps >= 1.1 s after the detected end (terminal elements, §17.4);
  - lies inside the file;
  - is crossed by no other event.

  Candidates are ranked by distance from the default onset-0.6 s; short excerpts (< 4.4 s) use the whole file only
  if the song is fully inside it.  Unplaceable events are dropped.
- **'borrowed'** (admitted only by the onset re-score while another event starts within 3 s): dropped.
- **Usable events (yh >= 0.3):** ncs 95.2 %, vizina 96.2 %, audiomoth 97.5 %, birdnet_yh 49.6 % (the rest of the
  BirdNET excerpts cut the song at their edges; they stay in SSL, which needs no complete songs).  Windows moved:
  15-16 %.  Guaranteed in-crop time after the detected end: median 1.5 s, >= 1.0 s for 78 %.
- QA `figs/qa/moved_s{0,1,2}.png`: the first version parked songs at the window edge (shift -2 s), fixed by the
  crop-safe zone; the final version keeps all terminal elements in view.
- **Output:**

  | Table | Rows | Note |
  |---|---|---|
  | `weak_v5.csv` | 50,788 | v3 had 47,773; more BhBl and BlBh |
  | `unlabeled_v4.csv` | 75,427 | |

  The queue was switched to weak_v5 for r7_s0, the architecture sweep and r3_s0; R7 folds 0/1 (started on older
  tables) were restarted.  r2_s1 and r6 keep weak_v3 for like-for-like comparison.

### 17.3 Loader hygiene (`train_cv.load_seg`, also used by SSL/explain/QA)
- **Short-file padding:** files shorter than the window were padded with white noise at 0.05 x std, about 26 dB
  below the file, which drew a dark block (black-patch-like cue) after 41 % of birdnet_yh excerpts.  Now `bg_noise`:
  noise with the file's 25th-percentile per-bin STFT magnitude (background even in song-dense excerpts).
  QA `figs/qa/pad_old_new{,2}.png`.
- **Digital-zero runs** (>= 10 ms) are now filled with the same background noise.  Prevalence by source:

  | Source | Files with runs | Where | Run length |
  |---|---|---|---|
  | vizina | 72 % | file start | 13 ms (MP3 decoder delay) |
  | ncs | 29 % | start/end | median 0.125 s, up to 0.64 s |
  | birdnet_yh | 11 % | interior | median 19 ms, up to 10 s |

  Real phone recordings never contain these; Grad-CAM had found a prediction driven by digital silence (§16).

### 17.4 Context-dependence test (`explain.py context`) and a correction of §16
Method: OOF song clips (all errors plus as many correct, dialect/Unfinished labels) re-scored with (a) everything
outside [onset-0.15, end+POST] replaced by the clip's own background noise ('song only') and (b) the song replaced by
background ('context only').

**First run (POST = 0.35 s):** R2 lost 48 % of its correct BC predictions in song-only, almost all becoming
Unfinished, and 26 % of context-only clips kept their label.  The sheet (`figs/explain/r2_s0_context.png`) shows why.
The quiet, short terminal **C element of BC** (and BE's E) follows the B whistle by 0.3-1.1 s and is below the tonal
segmenter's threshold and minimum duration, so the "song span" stopped at B.  The model was right: without C the song
*is* unfinished, and C alone still says BC.  A low-threshold faint-element extension caught only some of these, so
the test now keeps a fixed POST = 1.2 s.

Results with POST = 1.2 s (n = 761-792 clips):

| | base_v3 | R2 |
|---|---|---|
| context-only clip predicted None | 0.48 | **0.90** |
| context-only clip still predicted as its label | 0.34 | **0.07** |
| errors fixed by removing context | 0.054 | 0.078 (Unfinished 0.17) |
| correct predictions broken by removing context | 0.037 | 0.046 |

**Conclusion:** R2 reads the song itself; the weak/partial-label data cut the baseline's context use (34 % → 7 %).
Only ~8 % of R2's errors are context-driven, so R8 (other-species background mixing) is downgraded.  The occlusion
"outside" share in §16 is confounded by the same missed terminal elements; the occlusion maps themselves are fine.

**Pipeline caution (production too):** the energy/tonal segmenter's song end is often the B element.  Any cutting of
songs (app, backend, weak windows, realignment) must keep >= ~1.1 s after the detected end, or BC/BE songs turn into
"Unfinished".  realign_orig windows extend to onset+3.8 s (fine for songs <= 2.7 s); song_events now prefers this
margin.

### 17.5 R4 (FixMatch, unlabeled_v2) and R5 (hard negatives + phone codecs) vs R2, 5 folds, paired bootstrap
- **R4:** macro-F1 0.775 (R2 0.774), diff -0.001 [-0.015, +0.013].  R2 is ahead on BhBl (+0.035, P 0.89) and BlBh
  (+0.051, P 0.91).  Expert birds: acc 0.924 (R2 0.939), BlBh recall 0.835 (R2 0.873).  Vizina held-out acc 0.988.
  **FixMatch gives no gain at this scale; not adopted.**
- **R5:** macro-F1 -0.013 vs R2 [-0.002, +0.028], P(R2 better) 0.95.
  - None improves +0.048 (the hard negatives work).
  - Dialects get worse: BC -0.015, XB -0.022, BhBl -0.035.
  - The codec part needs the device-robustness test (the strong OOF set has no phone audio).

  Hard negatives are worth keeping only at a lower pool share; to retest inside the R7 recipe.

### 17.6 Calibration (calibrate.py, cross-fitted)
| | ECE raw | ECE after T-scaling | conformal 90 %: singleton / dialect singleton | auto-accept at 95 % acc |
|---|---|---|---|---|
| base_v3 | 0.020 | 0.019 | 0.882 / 0.908 | 0.511 |
| R2 | 0.036 | 0.037 | **0.919 / 0.941** | 0.513 |
| R4 | 0.030 | 0.028 | 0.921 / 0.945 | 0.463 |

Temperature scaling does not fix the factorised head (its output is a product of probabilities, not a softmax).
For production, use conformal sets or per-class thresholds rather than raw confidence (the backend's fixed 70 % cut).

### 17.7 R2 re-scored on realigned windows (`oof_rescore.py`): like-for-like baseline for R7, and a terminal-element check
R2 fold models on their own OOF rows of orig_aligned.csv (eval crop = window start = onset-0.6 s):
- macro-F1 0.770 (on the original windows: 0.774); accuracy on moved clips 0.869 → 0.861, kept clips 0.890 → 0.890.
- **R7 must be compared with 0.770** (same windows).
- Unfinished clips that were moved: 1693, of which 23 flip to a dialect (12 with conf > 0.8).  Sheet
  `figs/qa/unf_flips.png`: in most of them the new window shows the C/E stroke after B that the original clip cut off.
  The label was right for the old clip but is wrong for the new window.
- 318 moved Unfinished clips had their detected end < 1.1 s before the old clip edge, but only these 23 change prediction.
- Kept for R7 (0.6 % of rows; R2-on-aligned sees the same windows, so the comparison is fair).  Listed in
  `results/expert_audit.csv` (144 rows: 121 unf_cut + 23 unf_reveal), to be fixed in the next data generation.

### 17.8 Production pipeline audit (repo `main`, read-only; nothing changed)
- **Windowing:** no segmenter.  4 s windows on a 0.5 s grid over the whole recording at 48 kHz.  Argmax per window;
  "None" is dropped and confidence < 70 % becomes "Unknown"; overlapping windows are merged keeping the most
  confident (`runner.py:31-41, 79-85, 114-139`).
- **Not affected:** the "detected end stops at B" issue (§17.4) does not apply here.
- **Issues found:**
  1. Windows past the end of the file are **zero-padded** (digital silence; §16/§17.3 showed it drives predictions).
  2. Training peak-normalises (`model/densenet.py:126`) but inference does not: a train/serve mismatch.
  3. `predicted = zip(...)` is consumed by `max(...)`, so `fullPredictions` in the API response is always `{}`.
  4. Grid windows ignore song onsets.  bn_hop_check showed onset-anchored windows recover more songs; our models
     are trained on onset-anchored windows (onset at 0.2-0.6 s), so serving should anchor on onsets too, or average
     3 grid windows around each onset.
- An unmerged fix `codex/fix-classifier-window-bounds` aligns the last window to the file end.  It branches before
  016fe98, so merging it as-is would revert the multi-representative change.
- Branch `pytorch-restructure` (not deployed) serves an ONNX model with onset windows (-0.6/-0.4/-0.2 s): the
  right direction; its segmenter settings match ours.

### 17.9 Mobile export path verified (R2 fold 0: StrnadNet + factorised head, 1.52 M params)
`export_model.py` (waveform → probs/emb/att; STFT as conv1d; peak normalisation inside the graph), then
`export_model.py --tflite` (onnx2tf), then `tflite_parity.py`.

Two fixes were needed:
1. `FactorHead` built its output with `torch.empty(len(z), 7)`, which the tracer froze to batch 2.  Now `torch.stack`
   (identical values and gradients; verified).
2. onnx2tf's default `flatbuffer_direct` backend wrote graphs that fail to allocate (reshape 51200 != 128).  Now a
   static `waveform:1,128000` input and `-tb tf_converter`.

Parity on 400 fold-0 OOF clips (1 thread, x86):

| | size | max \|Δp\| vs torch | argmax agree | acc | ms / 4 s clip |
|---|---|---|---|---|---|
| torch | – | – | – | 0.8875 | 86 |
| ONNX | 6.3 MB | 5e-7 | 1.000 | – | – |
| TFLite fp32 | 6.5 MB | 0.0017 | 1.000 | 0.8875 | 84 |
| **TFLite fp16** | **3.3 MB** | 0.040 | 1.000 | 0.8875 | 83 |
| TFLite dyn-int8 | 2.0 MB | 0.53 | 0.985 | 0.8875 | 180 (x86; ARM kernels differ) |

Production good-model-2 is 30.5 MB (DenseNet121, 48 kHz).  **fp16 is the deploy format**: one graph serves the app
(TFLite) and the backend (ONNX, dynamic batch).  Int8 needs full-integer quantisation with a calibration set
before use.

Recording level (`eval_recordings`, NCS recordings, OOF):

| | coverage | acc | BC | BE | BhBl | BlBh | XB |
|---|---|---|---|---|---|---|---|
| base_v3 | 0.893 | 0.957 | 0.973 | 0.936 | 0.944 | 0.810 | 0.997 |
| R2 | **0.908** | **0.973** | 0.980 | 0.964 | 0.962 | **0.898** | 1.000 |
| R4 | – | 0.970 | 0.978 | 0.961 | 0.954 | 0.882 | 1.000 |

R2 remains the reference.
