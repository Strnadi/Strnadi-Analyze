# Continuation notes for the next agent (GPU server)

State as of 2026-11-23, about 13:30 UTC. Written at the end of the session that restructured the
repository. Read this file first, then `docs/REVIEW.md` (findings) and `docs/STATUS.md` (what is
verified). `docs/RESEARCH_LOG.md` is the research session's own experiment log, section by section.

This repository is public. Server paths, access links and anything from private correspondence are
deliberately not in it: the project owner has a private companion note (`HANDOVER_PRIVATE.md`, kept
in the workspace, not in git). Ask for it.

---

## 1. The project in one page

**Strnadi** is a citizen-science project on Yellowhammer (*Emberiza citrinella*) song dialects.
A dialect is defined by the last one or two elements of a song. The classifier assigns each song one
of 7 classes: the dialects `BC, BE, BhBl, BlBh, XB`, plus `Unfinished` (the bird left the ending
out) and `None` (no Yellowhammer song).

Two lines of work, in priority order:

1. **Strnadi classifier.** Best accuracy with a small model, ideally one model for both the phone
   app (offline, near real time) and the backend. A deployable, fine-tuned model comes before
   further research. The deployed model (Keras DenseNet121, "good-model-2") calls about 45 % of
   complete BhBl songs "Unfinished" on recordings it never saw.
2. **BioAcu.** Generalise to a platform where experts create their own projects: one shared
   embedding model with a small head per project, and a 2-D "draw a circle around the dialect" tool.

What the project owner has asked for, across all sessions (treat as standing instructions):

- Do not drop a strategy because of compute; on the old server there was no GPU, now there is.
- A cut-out must contain the whole song, otherwise "Unfinished" cannot be judged.
- Use the weakly labelled and unlabelled bulk; relabelling by hand is the bottleneck.
- The citizen (NCS) file-level dialect labels are not to be trusted much.
- Bring your own ideas from current research; do not just implement the owner's suggestions, and do
  not wait to be prompted. Own the backlog.
- Look at real spectrograms of every dataset, augmentation, test set and of the model's errors
  before trusting a number. Metric-only work was called out as "operating on vibes".
- Monitor training per epoch, keep checkpoints, do not ship an overfitted model.
- Explain experiments in plain words. Leave no data of value unused.
- Code style: each experiment a self-contained notebook-style script; layers and models one per
  file; readable, expanded code.

---

## 2. Where things stand: the short version

- **Dialects are already separated.** Given a song that label and prediction both call complete,
  the dialect is right 97-98 % of the time. BhBl and BlBh are confused 1-2 times in 3,744 clips.
- **The error is "is the ending there?"** 59-69 of 144 BhBl clips are predicted Unfinished.
  Complete-vs-Unfinished AUC: BhBl 0.79-0.83, BlBh 0.84-0.89, BC 0.93, BE 0.95-0.97, XB 0.98.
- **A large part of that is label noise.** About 310 errors are shared by four independently
  trained recipes; 163 clips are confidently contradicted by their average. Their spectrograms
  mostly show a first phrase and ONE terminal element, which the project's own rule calls
  Unfinished. Nobody has yet had an expert adjudicate them. This is the highest-value hour of human
  time available.
- **Only one training change has a significant measured gain:** the factorised head (R2).
- **Nothing has been trained with the code in this repository.** It is a verified port and
  rewrite of the research code plus new, untrained proposals.
- **The research session's jobs are still running on the CPU server** from its own untracked code
  (`research/`), unaware of the review findings.

---

## 3. The three sessions

### 3a. The research session (the original; code in `research/` on the CPU server, not in git)

It ran about 20 hours on a 40-core CPU server (cap: 45 threads). What it built, all carried over
into this repository:

- A 32 kHz decoded cache of ~430 h of audio and a set of tables (section 5).
- StrnadNet (0.9 M parameters), a model zoo, a job scheduler, the weak-label machinery, an
  evaluation suite, QA sheets and explanation tools.
- Fixes after an independent review it commissioned (V1-V11 in `RESEARCH_LOG.md` section 11) and
  after a visual QA pass (section 15, six defects that metrics had hidden).

**Results** (recording-grouped 5-fold CV on the 3,744 labelled clips, one seed each; numbers
recomputed from the stored predictions with `python -m tools.project_state`):

| Run | What | Folds done | Macro-F1 | BhBl recall | BlBh recall | AUC complete BhBl / BlBh |
|---|---|---|---|---|---|---|
| base_v3 | labelled clips only | 5 | 0.745 | 0.50 | 0.49 | 0.79 / 0.84 |
| r1_s0 | + weak pools, flat head | 5 | 0.754 | 0.50 | 0.53 | 0.80 / 0.85 |
| **r2_s0** | + factorised head | 5 | **0.774** | 0.55 | 0.65 | 0.83 / 0.89 |
| r5_s0 | R2 + hard negatives + codecs | 5 | 0.761 | 0.51 | 0.59 | 0.82 / 0.87 |
| r4_s0 | R2 + FixMatch | 3 of 5 | 0.777 (folds 0-2) | 0.49 | 0.66 | 0.79 / 0.88 |
| r6_s0 | R2 + JCU weak pool | 1 of 5 | 0.775 (fold 0) | 0.47 | 0.75 | 0.81 / 0.86 |

R2 vs base: +0.029 [+0.012, +0.046] (paired recording bootstrap; seed variance not included).
R2 on folds 0-2 alone is about 0.78, so R4 and R6 show no gain so far.

Other measured facts:

- Frozen Perch 2.0: logistic 0.662, MLP 0.691, temporal head on the spatial embedding 0.741.
  Frozen BirdNET 3.0: 0.50-0.52. Few-shot and 2-D maps on pooled Perch embeddings do not separate
  dialects; HDBSCAN finds no clusters.
- Held-out Vizina males, aligned clips: BhBl recall 0.96 (base_v3) vs 0.68 (production).
- End to end on full recordings of those males: song recall 0.875, BhBl 0.88, 1.65 false alarms
  per minute (production logic: 0.774, 0.68).
- Expert bird-level BE/BlBh test (332 birds): R2 0.939, base 0.893, the NCS file labels 0.863.
- NCS bird-level accuracy of base_v3: 0.957 at 0.89 coverage.
- Label-free pretraining from scratch (`ssl_v2`): linear-probe macro-F1 0.32 after 4,000 of 12,000
  steps and ~6 CPU hours (supervised from scratch: 0.745; frozen Perch: 0.66). As both reviews
  predicted, it is a poor use of compute next to a pretrained teacher.
- Parked: rule-based labelling from an element tracker (0.52 macro-F1, parser too noisy);
  bag-of-units VQ tokens (carry no dialect information without sequence modelling).

**What was running on the CPU server at the time of writing:** R4 folds 3-4, R6 fold 1, `ssl_v2`,
`ssl_hub`, JCU download batch 2. **Queued behind them:** R7 (R2 + visual-QA fixes), the
8-architecture sweep, `ssl_mae`, `ssl_ae`, second seeds of R2 / R1 / base, R3 (far-field), the
identity model. The session itself had been idle for three hours, waiting for R7, and had been
compacted twice.

**Read its results with these caveats** (found by the reviews; the research session was never told):

- Every run that mixes noise used a noise bank that contains Yellowhammer songs.
- R1-R6 trained on the black-patch counterfactual filler and clip-power SNR; R7 fixes those.
- R7's evaluation windows differ from R2's; compare only after re-scoring R2
  (`evaluation/rescore_oof.py`). Its architecture sweep trains pretrained backbones at the
  from-scratch learning rate, on 3 folds and one seed: it cannot resolve differences below ~0.03.
- Vizina "held-out males" and JCU "held-out recorders" are not independent of the weak training
  data used from R1 / R6 on (neighbouring territories, recorders 120-170 m apart).
- Its exporter has the batch-size bug fixed here (section 3c).

**Still open from that session, and only a human can unblock them:**

1. Where the Italian Alps passive recordings are (they need cutting with `processing/cut_songs.py`).
2. An export of confirmed segments from the app's production database: the only target-domain
   (phone) test set, and the only fair comparison with the production model.

### 3b. The base review session

Asked to review the research session as a senior ML engineer. It analysed the stored predictions,
looked at error spectrograms, and ran a code/data audit and a literature check. Its findings are
merged into `docs/REVIEW.md`; the ones that matter most:

- The labelling problem above (236 clips get < 10 % for their own label over three runs).
- Checkpoint selection on the inner validation set is noise; runs are twice as long as needed.
- 107 groups of byte-identical NCS recordings under different file ids, some with conflicting
  labels; observer identity predicts dialect (0.81 vs 0.69).
- The per-frequency median normalisation subtracts song energy from itself when the song fills the
  window (worst for BlBh).
- Literature (checked online): PyTorch ports of Perch 2.0 exist (Apache-2.0); the BirdCLEF 2026
  winner distilled small students into Perch 2.0's embedding space; pseudo-labels help only from a
  strong teacher; EfficientAT is MIT; ESP-AVES2 and NatureLM are CC-BY-NC-SA.

It ended with two offers that were never taken up: send the review to the research session, and
build a labelling page for the disputed clips plus a gold test set. Its error sheets and duplicate
list lived in a temporary folder and may be gone; both are reproducible
(`evaluation/report.py::plot_error_examples`, `processing/make_folds.py --dedupe`).

### 3c. The restructuring session (this branch)

Its own review agreed with the base review and added:

- The noise bank contains Yellowhammer songs (36 of 400 sampled chunks flagged by a trained model;
  clear songs among the 12 viewed).
- Noise level was referenced to the loud first phrase, not to the faint ending.
- BirdNET-app clips (395; 310 of the 551 XB clips) are never Unfinished or None: a source shortcut.
- All passive-recorder training songs are BhBl-only: "passive acoustics => BhBl" is learnable and
  the JCU metric rewards it; this is the main risk for the Italian recordings.
- With the factorised head the weak songs give no completeness supervision at all.

Then it produced this branch (three commits on `pytorch-restructure`):

- The Keras repository ported to PyTorch; the research code split into `layers/`, `model/`,
  `augment/`, `losses/`, `training/`; 54 experiments as notebook-style scripts; a shared
  `evaluation/report.py`; ONNX export and a song-level service; tests.
- Corrections listed in `docs/REVIEW.md` section 2, among them: stable bag loss, ONNX batch-size bug
  in the factorised head, centred evaluation window, comparison tool that refuses mixed tables,
  backbone learning-rate factor, polyphase speed augmentation, SpecAugment fill value.
- Verified: research checkpoints reproduce their stored predictions in the refactored models
  (max difference 5e-5); every rewritten file equals its predecessor numerically; all trainer paths
  run on synthetic data; a short real-data training run; export and service on a real recording.

Not done there: any full training; execution of most relocated data scripts; the Perch and
EfficientAT wrappers against the real downloads; Docker builds. See `docs/STATUS.md`.

---

## 4. First steps on the GPU server

1. **Environment.** Python 3.12; `pip install -r requirements.txt` with a CUDA build of torch.
   The CPU server used torch 2.15, torchvision 0.29, timm 1.0.31.
   `export STRNADI_WS=<workspace>` and `STRNADI_SRC=<raw sources>`.
2. **Data.** Copy the workspace (section 5), then
   `python -m tools.relocate_workspace <old workspace path>`: the tables store absolute audio paths.
3. **Check the port on the new machine** (all CPU, minutes):
   `python -m tests.test_models`, `tests.test_training`, `tests.test_experiments`,
   `tests.check_names`, and `tests.check_parity` (needs `runs/base_v3`, `runs/r2_s0`).
   `python -m tools.project_state` should print the table of section 3a.
4. **Make the trainer use the GPU. It never has.** `training/fit.py` was written and tested on
   CPU only; nothing is moved to a device. To do:
   - one `device` (settings field, default `cuda` if available); move the model, the EMA copy, the
     distillation projection, the teacher ensemble and every batch tensor;
   - `classification_loss` builds `torch.zeros(len(logits))` on the CPU: create it on the logits'
     device; same for `losses/fixmatch.py` (its thresholds live on the CPU);
   - `training/validation.py`, `training/inference.py` and `evaluation/report.py` call `.numpy()` on
     model outputs: add `.cpu()`;
   - `layers/stft_power.py` caches its conv kernels per device (fine) and uses `torch.stft` (fine);
   - mixed precision (`torch.autocast`) and `channels_last` for the CNNs; larger batches; the
     waveform augmentation runs in DataLoader workers on the CPU and will become the bottleneck:
     raise `data_workers`, or move noise mixing and the STFT to the GPU;
   - `training/pretrain.py`, `unsupervised.py`, `identity.py`, `bird_level.py` need the same.
   Re-run `tests/test_training.py` on both devices afterwards, and `tests/check_parity.py`.
5. **Reproduce R2 with this code** (`python -m experiments.r2_factorized_weak`), expecting about
   0.774 macro-F1. On a GPU this is minutes per fold. It is the missing end-to-end check of the
   refactored trainer; do it before trusting any new number.
6. **Then follow section 6.**

---

## 5. Data

Everything lives in the workspace (`$STRNADI_WS`); raw sources are only needed to rebuild the cache.

| Part of the workspace | Size | What |
|---|---|---|
| `audio32k/ncs` | 34 GB | citizen recordings 2011-2022, 155 h, decoded to 32 kHz int16 `.npy` |
| `audio32k/vizina` | 21 GB | focal recordings of ring-coded males at one site (all complete songs BhBl), 98 h |
| `audio32k/audiomoth_bhbl` | 19 GB | AudioMoth sample from the same area, 88 h |
| `audio32k/birdnet_yh` | 20 GB | 52k BirdNET Yellowhammer detections (Germany), unlabelled, 89 h |
| `audio32k/jcu` | 24 GB | songs cut from JCU AudioMoth batch 1 (and batch 2 while it runs) |
| `audio32k/orig`, `cf_v3`, `cf_v4` | 1.6 GB | the 4,089 labelled clips; counterfactual songs |
| `data/` | 2.1 GB | all tables (below), noise bank, far-field statistics |
| `emb/` | 0.25 GB | Perch 2.0 and BirdNET 3.0 embeddings of the labelled clips |
| `runs/` | 0.5 GB | research checkpoints, logs and out-of-fold predictions |

About 120 GB in total. Not in the workspace: the remaining ~23 TB of JCU AudioMoth audio (76k
hourly files; download with `processing/jcu_pipeline.py`, access through `JCU_NAS_*` environment
variables, values from the project owner), the Italian recordings, the app database export.

Key tables in `data/`:

| Table | Rows | Use |
|---|---|---|
| `orig_index.csv` | 4,089 | labelled 4 s clips with folds (`aug` copies excluded: 3,744 used) |
| `orig_aligned.csv` | same rows | the same clips re-cut around the whole song (R7 onward) |
| `weak_v3.csv` | 47,773 | songs with candidate-set labels from file / site labels, plus None windows |
| `cf_v3.csv`, `cf_v4.csv` | 1,188 | counterfactual songs (swap -> BlBh, truncations -> Unfinished); v4 has the fixed filler |
| `vizann_v1.csv` | 1,598 | annotated Vizina songs of training males |
| `jcu_train.csv`, `jcu_test.csv` | 41.7k / 12.9k | JCU songs, 28 / 7 recorders |
| `hardneg_v1.csv` | 6,865 | other-species windows |
| `unlabeled_v2.csv` | 77.5k | songs for semi-supervised training |
| `song_index.parquet`, `none_index.parquet` | ~180k | every detected segment with source, fold, split, BirdNET score |
| `vizina_eval.csv`, `vizina_files.csv` | | held-out-male test set; file -> male mapping |
| `noise_bank_audiomoth.npy` | 1,212 x 6 s | background noise. **Contaminated with songs: rebuild** |
| `farfield_stats.csv` | | level / SNR by distance from the BioDCASE 2025 playback data |

Class counts in the labelled set: Unfinished 2,086, XB 551, BC 469, BE 291, BhBl 144, None 124,
BlBh 79. External, public: BioDCASE 2025 Task 3 Yellowhammer data (Zenodo 15228365), the
measurements of Diblikova et al. 2023 (Zenodo 8092674; its song-time column is a 6.000 s placeholder
in 384 of 430 rows).

**Rebuild before training anything new** (commands in `docs/STATUS.md`): the clean noise bank
(and look at the sheet it writes), folds without duplicate recordings, tables with song spans, the
species bank, the disputed-clip table, Perch embeddings for the weak tables.

---

## 6. Plan, in order

The GPU changes what is sensible. On the CPU server a recipe took 15-20 fold-hours; now every
comparison can be 5 folds x 3 seeds, and large pretrained encoders can be fine-tuned.

1. **Honest measurement first.**
   - Expert adjudication of the disputed clips (`processing/build_disputed.py` writes the list);
     a ~500-clip gold test set labelled by two or three experts, with the protocol "label what is in
     the window; record the bird's dialect separately". Until then train disputed clips as
     candidate sets (`experiments/disputed_as_candidates.py`).
   - Headline metrics: complete-vs-Unfinished AUC per dialect, dialect accuracy given complete,
     bird-level accuracy. Macro-F1 over 7 classes rewards the None/Unfinished boundary, which has
     no scientific value.
   - Folds grouped by duplicate-free recording; an observer- or site-grouped check; three seeds;
     keep one test set untouched (the app export, when it exists).
2. **Fix the data defects** and re-run R2 on the corrected recipe
   (`experiments/clean_noise_ending_snr.py`). Measure the passive-recorder shortcut with
   `evaluation/dialect_prior_shift.py` before using any JCU-trained model on passive data.
3. **A strong teacher.** Fine-tune pretrained encoders with the weak-label recipe (section 7A).
   Rank frozen encoders by probing first (minutes each), then fine-tune the best two or three.
4. **Completeness-specific modelling** (section 7B-7E), each as an ablation against step 2.
5. **Distil** the teacher ensemble into the phone-size model on identical augmented views over a
   long schedule, with teacher pseudo-labels on all ~180k unlabelled songs
   (`experiments/distill_function_matching.py`).
6. **Detector and end-to-end.** A learned frame-level song detector; whole-recording evaluation;
   bird-level aggregation with abstention as the product metric.
7. **Deploy.** ONNX export with the parity checks, a TFLite build for the phone with the same
   checks, latency on a real device, the service pipeline on phone recordings.
8. **BioAcu** on top of the winning encoder: per-project heads, few-shot curves, a fixed 2-D map,
   open-set flags. Do not start it before step 5 is done.

Stop doing: self-supervised pretraining from scratch on CPU-sized models, the from-scratch patch
transformer, further JCU downloads for dialect training (one dialect, one site; useful only as
unlabelled audio and for robustness).

---

## 7. Architectures: what exists here, and what else could work

Everything in A-E that says "in repo" is implemented and smoke-tested but untrained. Literature
claims marked (verify) come from memory and were not re-checked online.

### A. Pretrained encoders to fine-tune (teachers)

| Encoder | Why | State |
|---|---|---|
| Perch 2.0 (EfficientNet-B3, 12 M, 32 kHz) | best bird encoder; frozen + temporal head already equals the from-scratch baseline | in repo: `model/perch_temporal.py`, `experiments/perch_finetune_teacher.py`; run `tests/check_perch.py` first |
| BirdNET 3.0 (EfficientNetV2-S, 32 kHz) | second bird teacher with different training data; weak frozen (0.52) but untested fine-tuned | not in repo; needs a PyTorch route from its ONNX weights |
| EfficientAT mn10 / dymn (AudioSet) | strong small audio CNN; teacher and phone candidate | in repo, wrapper untested: `model/efficientat_temporal.py` |
| Bird-MAE (ViT-B/L, 32 kHz) | best on BirdSet when fine-tuned; needs prototypical pooling | not in repo; `layers/prototype_head.py` is the matching head |
| BEATs, EAT, Dasheng, AVES / BirdAVES (16 kHz) | strong general audio transformers | not in repo. 16 kHz cuts the band at 8 kHz: feed half-speed audio (3-9 kHz -> 1.5-4.5 kHz), which also doubles mel and time resolution. Check licences: ESP-AVES2 and NatureLM are non-commercial |
| ConvNeXt (BirdSet) / AudioProtoPNet | strong supervised bird CNN; interpretable prototypes | not in repo |

How to use them on 3.7k labelled clips without overfitting: layer-wise learning-rate decay
(`backbone_lr_factor` is the coarse version), low-rank adapters (LoRA) or training only the last
blocks for the ViTs, the weak pools and the factorised head from the start, three seeds. Rank all
of them cheaply first with frozen features and three probes (linear, attentive, prototype) on the
completeness AUC, not on species-style accuracy.

### B. Models aimed at the faint ending (in repo)

- `layers/multires_frontend.py` + `model/strnadnet_tonal.py`: 16 / 32 / 128 ms windows as channels,
  long horizontal kernels in the stem, percentile noise floor.
- `model/ending_zoom.py`: coarse network + a 27 Hz / 5 ms network on the 1.5 s around the song end.
- `layers/freq_dynamic_conv.py` + `model/fdy_crnn.py`; `layers/ridge_contour.py` +
  `model/contour_tcn.py` (sub-bin pitch contour).
- Front-end ablations: `experiments/frontend_{percentile_floor,pcen,linear_hires,multires}.py`.

### C. New ideas for the ending, enabled by a GPU

- **Instantaneous-frequency channel.** Add the time derivative of the STFT phase as a second input
  channel. For a tonal element it gives the frequency to a fraction of a bin, so 100-250 Hz criteria
  are resolved without a longer window. Cheap; pairs with the multi-resolution front-end.
- **Sharper time-frequency pictures of thin tones:** reassigned or synchrosqueezed spectrograms,
  superlets, multitaper estimates, or a constant-Q transform with 48-96 bins per octave over
  2-10 kHz. All are fixed front-ends: one ablation each.
- **Fine-patch transformer on the zoom crop.** A ViT with tall, thin patches (about 4 frequency x 2
  time bins) on the 1.5 s ending crop only, initialised from a pretrained audio ViT with
  interpolated patch embedding. The crop keeps the token count small.
- **Relational head for the last two elements.** Dialects such as BhBl vs BlBh are a comparison
  (which element is higher). Encode the two terminal elements separately and classify from their
  difference; or regress the pitch offset and bandwidth as ordinal targets
  (`model/strnadnet_measure.py` is the simple version).
- **Completeness as detection, not classification.** Train "is there a terminal element after the
  phrase?" as its own detector on synthetic positives: paste real terminal elements after real
  first phrases at controlled levels from +10 to -25 dB relative to the phrase, with soft labels
  near the audibility limit. This yields unlimited exact supervision for the one decision that
  fails, and a psychometric curve (detection vs level) as an evaluation.

### D. Element sequences without frame labels

`model/element_tagger.py` needs per-frame element labels that do not exist yet. Two ways around it:

- **CTC over element tokens from clip labels.** A clip labelled BC has the target sequence
  `phrase B C`; XB `phrase X B`; BhBl `phrase B B`. "Unfinished" is the set
  {`phrase`, `phrase B`, `phrase X`}; a weak label is the union of its candidates' sequences. CTC
  (or its multi-target form: maximise the summed likelihood of the allowed sequences) trains the
  tagger with no alignment at all, and the forced alignment afterwards gives the frame labels.
- **Set prediction of elements** (DETR-style, as in the bioacoustic box detectors of the Voxaboxen
  line (verify)): predict (class, start, end, centre frequency) per element.

Both give, as by-products, element positions for truncation counterfactuals in every dialect, the
ending location for the zoom model, and a natural "unknown dialect" output (an unseen sequence).

### E. Making the weak and unlabelled data teach completeness

- Recording-level bag loss (`losses/bag_complete.py`, `experiments/bag_completeness.py`) and the
  bird-level bag model (`model/recording_mil.py`).
- **Progressive disambiguation of candidate sets** (PRODEN-style (verify)): start from uniform
  weights over a song's candidates and sharpen them with the model's own predictions, song by song.
- **Two-network noise-robust training** (co-teaching / DivideMix-style (verify)) on the labelled
  set, where an estimated 5-10 % of labels describe the bird and not the window.
- **Noisy student with a real teacher:** the fine-tuned ensemble labels all ~180k songs with soft,
  sharpened, confidence-filtered targets; the student trains on them with strong augmentation.
- **Mean-teacher sound-event detection** with strong + weak + unlabelled data, the standard recipe
  of the DCASE detection task (verify): the data here has exactly that mix (Vizina boxes, file
  labels, raw audio), and its output is the frame-level detector the pipeline lacks.

### F. Domain-adaptive pretraining (now affordable)

Not from scratch: continue the self-supervised objective of an already pretrained encoder (masked
prediction in the style of EAT / Bird-MAE, or the denoising variant of M2D) on the Yellowhammer
audio, then fine-tune. The research session's corpus rules apply (training split only; no pairing
of complete with unfinished songs). The JCU archive finally has a use here.

### G. Small models for the phone

- Distillation targets: StrnadNet (0.9 M), `model/bc_resnet.py` (0.15 M), EfficientAT mn04 (1 M).
- With a GPU: a latency-constrained search over width, depth and strides of the StrnadNet family
  with the teacher's soft labels as the target (a once-for-all style supernet (verify) or plain
  random search); quantisation-aware training; a causal variant for live feedback.
- Pick from an accuracy vs on-phone-latency curve. Size is not the constraint: 3-5 M int8
  parameters run comfortably on phones.

### H. Around the classifier

- **Bout context.** A male repeats his song: smooth per-song posteriors over a bout with a small
  sequence model or an HMM, in addition to the bag model.
- **Unknown dialects** (Italian data): distance-based rejection on the teacher's embedding, and
  category discovery by clustering ending embeddings, with experts naming clusters.
- **Audio-language models as a second annotator** for the disputed clips (analysis only; check the
  licence). Low expectation, cheap to try.
- Speculative, low prior: state-space (Mamba-style) audio encoders; classification from neural
  codec tokens (codecs probably destroy faint tones).

### What I would bet on, in order

1. Clean labels and data (no architecture fixes those).
2. A fine-tuned Perch 2.0 teacher, plus one transformer teacher fed half-speed audio.
3. Completeness as detection with pasted terminal elements (7C), and the CTC element model (7D).
4. Instantaneous-frequency + multi-resolution input on the student.
5. Function-matching distillation into StrnadNet, then quantisation.

---

## 8. Traps that have already cost time

- Tables contain the literal label `None`: always read with `keep_default_na=False`
  (`training.data.read_table` does).
- Never pad with zeros or by reflection; never time-stretch; no pitch shift beyond 2 %; no mixing of
  two dialects; SpecAugment masks must stay narrower than a terminal element.
- A window must contain the whole song. Songs longer than 3.8 s are dropped, not cut.
- Site-level labels (Vizina, AudioMoth, JCU) say "BhBl or Unfinished", never which.
- The Vizina annotation "song" rows are fixed 3 s detector boxes; real boundaries are in the
  component rows.
- The expert table's song-time column is a placeholder; re-detect songs instead.
- A metric that looks plausible can sit on a broken test set. Render the spectrograms.
- BirdNET on a 3 s grid finds ~75 % of songs; three onset-aligned windows find ~82 % at a third of
  the false positives of a 0.5 s grid.
- An exported graph must be checked at a batch size other than the traced one.
- One seed and a bootstrap over recordings does not include training-seed variance.
- This repository is public: no share links, tokens, server paths or private correspondence in
  commits. `processing/jcu_nas.py` reads its access details from the environment for that reason.

---

## 9. Open questions for the project owner

1. Location of the Italian Alps recordings.
2. Export of confirmed segments from the app database (recording, interval, dialect, device).
3. Expert time: the disputed clips and a gold test set. Should a labelling page be built?
4. Should the research session on the CPU server be told about the review findings, or stopped?
   Its queued jobs (R7, the architecture sweep, the pretraining runs) repeat known defects.
5. May the folds be rebuilt without duplicate recordings? Old and new runs are then not comparable.
6. The key from Vizina phrase types / components to the element names B, C, D, E, X.
7. A recording -> observer table (for observer-grouped folds).
8. Unpadded originals of the BirdNET-app clips.
9. A Xeno-canto API key (3,758 Yellowhammer recordings) and the larger set of BirdNET detections
   offered by the Chemnitz group; the dialect labels of the Pipek / Petruskova / Ghani datasets.
10. Which licences are acceptable for a teacher model (some strong encoders are non-commercial).

---

## 10. Where to look

| Question | File |
|---|---|
| What can be set for a training run? | `training/settings.py` |
| How does training work? | `training/fit.py` |
| What does an experiment look like? | `experiments/r2_factorized_weak.py` |
| Which experiments exist, and their status? | `python -m tools.list_experiments` |
| State of the runs in a workspace? | `python -m tools.project_state` |
| Evaluation, plots, error sheets | `evaluation/report.py` |
| Findings and what was fixed | `docs/REVIEW.md` |
| What is verified, what is not; data to rebuild | `docs/STATUS.md` |
| The research session's log | `docs/RESEARCH_LOG.md` |
