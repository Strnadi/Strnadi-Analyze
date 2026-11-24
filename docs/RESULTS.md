# Results on the audited labels (2026-11-24)

This is the current state of the Strnadi classifier. It supersedes the numbers in
`RESEARCH_LOG.md` and `CONTINUATION.md`, which were measured against the original clip labels.

## In one paragraph

The old ceiling (accuracy ~0.87, macro-F1 ~0.77 for every recipe and architecture) was set by the
labels, not by the models. After a visual audit of every clip the models disputed
(`LABEL_AUDIT.md`: 220 relabelled, 140 removed as unjudgeable, of 3,744), a **0.9 M-parameter**
network trained from scratch reaches **TABLE_REFERENCE_ACCURACY accuracy / TABLE_REFERENCE_F1
macro-F1** on recordings it never saw, and a **0.19 M-parameter** one is within a point of it. The
deployed DenseNet121 (30.5 MB) scores 0.911 on the same clips although it was trained on most of
them. No distillation, no teacher: every model here is trained directly on the source data.

## How the numbers are measured

- **Clips:** `data/clips_audited.csv`, 3,604 human-labelled 4 s clips after the audit
  (Unfinished 2,212, XB 524, BC 399, BE 260, BhBl 80, None 66, BlBh 63).
- **Protocol:** 5-fold cross-validation grouped by recording; byte-identical recordings filed under
  different ids count as one. Every clip is predicted by a model that never saw its recording, nor
  any weakly labelled song of that recording.
- **Window:** exactly the 4 s the labeller saw (`eval_window='center'`).
- **Checkpoint:** the epoch with the lowest loss on an inner validation set (15 % of the training
  recordings); the test fold never chooses anything. Test-fold curves are logged to TensorBoard for
  diagnosis only.
- **Accuracy** is the headline ("overall success"); macro-F1 weights the three rare classes
  (BhBl, BlBh, None: 60-80 clips each) as much as the 2,212 Unfinished clips and is noisier.
- One seed per row unless stated; the spread between seeds is given below.

## All runs

Produced by `python -m tools.results_table --markdown`. "fold sd" is the standard deviation of
accuracy across the five folds.

TABLE_ALL_RUNS

## What each comparison says

TABLE_FINDINGS

## The model that ships

TABLE_FINAL_MODEL

## Is it valid? Checks that do not depend on the audit

| Check | Result |
|---|---|
| **Independent expert labels.** 663 songs of 15 males at the Vizina site, annotated by ornithologists in Raven (complete BhBl vs Unfinished); these males are in no training table | reference 0.992 accuracy (BhBl recall 0.985, Unfinished 0.996). Trained on the audited clips alone, with no audio from that site at all: 0.985. The deployed DenseNet121: 0.855 (BhBl recall 0.68) |
| **Unjudgeable clips counted as errors.** The audit removed 140 clips it could not decide; a sceptic can count every one of them against the model | 0.927 |
| **Control sample.** 75 random clips on which models and labels agree, viewed as spectrograms | 74 labels right |
| **No overfitting.** Test-fold loss at the kept checkpoint vs at the last epoch | within 0.01 in every fold; training loss stays above validation loss (augmentation) |
| **Test-time shifts.** The same clips moved by +-0.1 / +-0.2 s | accuracy 0.957-0.963; averaging shifted windows changes nothing: the model does not depend on where the song sits in the window |
| **Deployed model on the same clips**, although it was trained on most of them | 0.911 accuracy, macro-F1 0.840. It reproduces the label errors it memorised: 75 one-element songs called BC, 46 called BhBl, 38 called BE |

Confidence is usable: predictions with confidence >= 0.7 are 94 % of all clips and 98.1 % correct;
>= 0.9: 77 % of clips, 99.0 % correct.

## What the model looks at

`python -m evaluation.explain_model <run>` writes, per class, the spectrogram of example clips with
three occlusion maps (one per question of the factorised head: is there a song? is it complete?
which dialect?) and the temporal attention, plus a deletion test. Figures:
`figs/explain/<run>/`, and the TensorBoard images under `runs/<run>/explain`.

- The evidence for "complete" sits on the terminal elements: on the second B of BhBl / BlBh, on the
  falling C of BC, on the rising E of BE. The temporal attention peaks on the last element.
- **Deletion test** (400 correctly classified complete songs, reference model). Hiding the ONE
  most important 16-bin x 0.2 s patch (2.5 % of the picture) turns 15.5 % of the answers into
  "Unfinished"; hiding a random patch: 0.0 %. Hiding the three most important patches: 45.5 %
  (BlBh 87 %, BhBl 75 %, BE 59 %, XB 42 %, BC 33 %); three random patches: 0.75 %. So the
  completeness decision rests on a few small patches at the ending, as it should, and the maps are
  faithful to the model.
- The errors seen on the sheets are of three kinds: a terminal element fainter than the model
  will accept (BC with a barely visible C); two songs in one window, where the label follows one
  song and the attention the other; and unusual variants (a C that falls from 6.5 to 5.5 kHz
  instead of 9 to 6.5).

## The embedding as a map (BioAcu: "draw a circle around a dialect")

`python -m evaluation.embedding_map <run>`: the 128-d clip embedding of held-out clips, projected to
2-D by UMAP without labels.

| | Frozen Perch 2.0, pooled (research log, section 8) | This model, held-out clips |
|---|---|---|
| kNN accuracy in the embedding | - | 0.967 |
| kNN in the 2-D map | dialect-F1 0.15-0.42 | accuracy 0.953 |
| silhouette of the classes | ~ 0 | 0.77 (embedding), 0.44 (2-D) |
| HDBSCAN on the 2-D map, no labels | no clusters | 6 clusters, 95 % pure |

Each dialect is its own island, and "Unfinished" is one large region between them. An expert could
outline every dialect by hand on this map. The practical conclusion for the platform: a generic
bird-species embedding does not carry within-species structure, but a small project-specific
encoder (under 1 M parameters, minutes of GPU training on a few thousand labelled clips) does, and
its map is drawable.

## What is still open

- **An expert pass over the audit** (`label_audit/clip_label_audit.csv`). The audit applied the
  written rule strictly but was not done by an ornithologist.
- **Phone recordings.** Every number here is on archive recordings (dictaphones, 2011-2022) and
  BirdNET-app clips. A set of confirmed segments from the app database is still the missing test.
- **X-only and B-only songs.** "Unfinished called XB" is the largest remaining confusion (28
  clips): songs with an X element and no B. Four such clips were found filed under XB among the
  40 least confident agreeing XB clips; more may hide among the confident ones.
- **Two songs in one window.** The label rule ("a complete song anywhere in the window?" vs "the
  last song?") is not defined by the project; the service avoids the case by cutting one window
  per detected song.
- Everything listed under "blocked on the owner" in `CONTINUATION.md` section 9.
