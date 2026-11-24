# Results on the audited labels (2026-11-24)

This is the current state of the Strnadi classifier. It supersedes the numbers in
`RESEARCH_LOG.md` and `CONTINUATION.md`, which were measured against the original clip labels.

## In one paragraph

The old ceiling (accuracy ~0.87, macro-F1 ~0.77 for every recipe and architecture) was set by the
labels, not by the models. After a visual audit of every clip the models disputed
(`LABEL_AUDIT.md`: 224 relabelled, 140 removed as unjudgeable, of 3,744), a **0.9 M-parameter**
network trained from scratch reaches **0.96 accuracy / 0.92
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

<!-- runs table: begin -->
| run | folds | params (M) | accuracy | fold sd | macro-F1 | BC | BE | BhBl | BlBh | None | Unfinished | XB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| size_strnadnet_2900k | 5 | 2.91 | 0.962 | 0.007 | 0.918 | 0.95 | 0.97 | 0.94 | 0.87 | 0.80 | 0.97 | 0.98 |
| strnadnet_audited_labels | 5 | 0.90 | 0.961 | 0.005 | 0.914 | 0.96 | 0.97 | 0.91 | 0.81 | 0.79 | 0.97 | 0.97 |
| size_strnadnet_430k | 5 | 0.43 | 0.961 | 0.008 | 0.916 | 0.95 | 0.98 | 0.90 | 0.84 | 0.80 | 0.97 | 0.98 |
| strnadnet_audited_labels_s1 | 5 | 0.90 | 0.961 | 0.007 | 0.921 | 0.96 | 0.98 | 0.93 | 0.89 | 0.79 | 0.96 | 0.97 |
| augment_real_world | 5 | 0.90 | 0.959 | 0.008 | 0.916 | 0.95 | 0.98 | 0.90 | 0.86 | 0.82 | 0.96 | 0.98 |
| ablation_clip_labels_only | 5 | 0.90 | 0.958 | 0.005 | 0.905 | 0.96 | 0.98 | 0.88 | 0.76 | 0.73 | 0.97 | 0.98 |
| strnadnet_field_ready | 5 | 0.91 | 0.957 | 0.010 | 0.914 | 0.95 | 0.97 | 0.90 | 0.83 | 0.86 | 0.96 | 0.98 |
| ablation_flat_head | 5 | 0.90 | 0.956 | 0.010 | 0.897 | 0.94 | 0.96 | 0.89 | 0.76 | 0.79 | 0.97 | 0.97 |
| size_strnadnet_190k | 5 | 0.19 | 0.955 | 0.007 | 0.907 | 0.95 | 0.96 | 0.91 | 0.84 | 0.83 | 0.96 | 0.97 |
| expanded_machine_labels | 5 | 0.90 | 0.955 | 0.004 | 0.910 | 0.95 | 0.98 | 0.90 | 0.86 | 0.82 | 0.96 | 0.98 |
| backbone_mobilenetv3_large_imagenet | 5 | 4.97 | 0.954 | 0.007 | 0.901 | 0.97 | 0.98 | 0.85 | 0.78 | 0.79 | 0.96 | 0.96 |
| element_sequence_ctc_weak_rows_in_sequence_loss | 5 | 0.91 | 0.954 | 0.008 | 0.903 | 0.94 | 0.97 | 0.89 | 0.84 | 0.80 | 0.96 | 0.97 |
| element_sequence_ctc | 5 | 0.91 | 0.953 | 0.008 | 0.902 | 0.94 | 0.97 | 0.93 | 0.86 | 0.79 | 0.96 | 0.96 |
<!-- runs table: end -->

Scored against the labels after audit round 3; numbers quoted in the text below were partly
measured before its four relabels and can differ from the table by 0.002.

The queue was still running when this was written (front-ends, the other architectures, the
second labelling round); `python -m tools.results_table --write-doc` refreshes the table.

## What each comparison says

Two seeds of the reference give 0.963 and 0.961 accuracy (macro-F1 0.915 and 0.921): differences
below about half a point of accuracy, or one point of macro-F1, are noise.

- **Labels were the ceiling.** The same recipe went from 0.874 / 0.774 (original labels) to
  0.96 / 0.92 (audited labels and labelled-clip windows).
- **Size barely matters.** 0.19 M parameters: 0.956; 0.43 M: 0.961; 0.90 M: 0.962; 2.9 M: 0.962.
  The race to the bottom is open well below one million parameters.
- **ImageNet pretraining does not help.** MobileNetV3-large (5.0 M with its head, pretrained): 0.955, below the
  0.19 M network trained from scratch.
- **The factorised head matters** (song x complete x dialect): the plain 7-class head loses 0.7
  points of accuracy and 2 of macro-F1.
- **The weak file-level labels now add little**: clip labels alone reach 0.958 / 0.905. They mostly
  help the rare classes (BhBl 0.88 -> 0.91, BlBh 0.76 -> 0.81).
- **Machine-labelled songs, field augmentation and the element-sequence output** each cost
  between nothing and 0.8 points on this test set, and together half a point; see their sections
  for why they are in the shipped recipe anyway.

## The model that ships

`experiments/strnadnet_field_ready.py` (the recipe) and `experiments/train_shipped_model.py`
(training on all folds, export).

| | |
|---|---|
| Network | StrnadNet, 0.90 M parameters, trained from scratch: 128-bin log-mel 1.5-11 kHz -> 4 residual stages -> dilated temporal convolutions -> attentive pooling -> factorised head, plus the element-token head |
| Input / output | 4 s of 32 kHz mono audio -> 7 class probabilities, a 128-d embedding, the element sequence |
| Trained on | audited clips (40 % of draws), machine-labelled songs (25 %), weak file-level songs (20 %), synthetic ending swaps / truncations (8 %), expert-annotated focal songs (7 %) |
| Augmentation | speed +-2 %, clean background noise, far-field simulation, other species in the background, device EQ, null splices, SpecAugment |
| Cross-validated | **accuracy 0.957, macro-F1 0.914** (reference recipe without the field additions: 0.962 / 0.918) |
| Independent expert songs (Vizina, 663) | shipped single model 0.979 (BhBl recall 0.985); the five fold models together 0.991 |
| Checkpoint | epoch 18 of 25, the lowest inner-validation loss (0.203; last epoch 0.207: no overfitting) |
| Files | `runs/strnadnet_field_ready/final/model.pt`; `export/strnadi_classifier.onnx`, 6.3 MB, ONNX Runtime = PyTorch to 1e-7 |
| Explanation | deletion test: hiding the 3 most important patches turns 33.5 % of complete songs into "Unfinished" (random patches 0.25 %); head and element sequence agree on 89 % of clips, which are 97.1 % correct |

Why this one and not the highest number: the reference is half a point better on archive
recordings, but it has never heard a simulated far-field song, another species over the song, a
phone-detected song from another country, and it cannot say which elements it heard. The app's
recordings are phone and field recordings. If a pure-accuracy model is preferred, train it with
`STRNADI_RECIPE=strnadnet_audited_labels python -m experiments.train_shipped_model`.

**Size.** Of the 6.3 MB ONNX file about 2.5 MB is the spectrogram front-end written as
convolutions (so that the graph uses only standard operators), not the network. The phone-size
candidate is `strnadnet_field_ready_190k` (0.19 M parameters; see the table). A TFLite build of
the new models has not been made in this session (`serving/export_tflite.py`, needs TensorFlow).

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

## Element sequences: the model says which elements it heard

`losses/element_ctc.py`, `experiments/element_sequence_ctc.py`, `evaluation/element_sequences.py`.
Besides the class, the model emits a sequence of song elements, e.g. `phrase X B`. It is trained
with CTC from the clip labels alone (no frame annotation): a BC clip must contain `phrase B C`
somewhere, an Unfinished clip `phrase`, `phrase B` or `phrase X`, a None clip nothing.

**The table.** Nothing in the decoder is specific to Yellowhammers. A project (one species, with
all its dialects or call types) supplies one table, `ElementGrammar`: its element names and every
dialect written as its ending (`BC = B C`, `XB = X B`, ...). Training targets and decoding are
derived from it: the decoded tokens are split into songs at each `phrase`, an ending found in the
table is that dialect, a proper prefix of one is "Unfinished", anything else is reported as
`other` with the raw sequence for an expert. Adding `XBC = X B C` to the table is enough to make
`phrase X B C` decode as XBC (checked), though the model has never seen such a song. Two songs in
one window each get their own answer.

**Results** (5 folds, 0.9 M parameters):

| Run | Class head | Sequence, greedy | Sequence, likelihood | Head and sequence agree | Accuracy where they agree |
|---|---|---|---|---|---|
| sequence loss on all rows, weak candidate sets included | 0.955 | 0.864 | 0.892 | 0.907 | 0.968 |
| `element_sequence_ctc`: sequence loss on hand-labelled clips only | 0.953 | 0.921 | 0.930 | 0.941 | 0.966 |
| `element_sequence_discriminative`: classes compete | queued; see the runs table and `python -m evaluation.element_sequences element_sequence_discriminative` | | | | |

- The first version read most BC / BE songs as `phrase B`: a candidate set such as
  `BC|Unfinished|None` lets CTC always take the easy reading. Using only hard labels fixed it
  (BC recall of the sequence 0.66 -> 0.91, BE 0.51 -> 0.94).
- The sequence is not a better classifier than the head, and the auxiliary loss does not improve
  the head (0.953-0.955 against 0.961-0.963 for two seeds of the reference: a small real cost).
- What it buys: a readable answer, and a free "not sure" flag. Where head and sequence disagree
  (6 % of clips) the head is right only 75 % of the time; where they agree, 96.6 %.

**Limits of "generic".** The table fits species whose categories are an ordered sequence of a few
discrete, nameable elements. It does not fit graded or continuous calls, large improvised
repertoires, or classes defined by rhythm or absolute pitch; such projects would use the plain class
head on the same encoder and the embedding map for exploration. Element names do not transfer
between species (nothing knows that C is "a falling sweep"); a shared vocabulary of shape
primitives (flat tone, up-sweep, down-sweep, burst, trill, plus relative pitch) is the idea that
could, and it is untested. **No second species has been tried: there is no such data here.**
Measured attributes (short / long X) are not implemented; the token would locate the element and
its duration would be measured on the spectrogram.

## Growing the labelled set with the model in the loop

`processing/machine_label_songs.py`, `experiments/expanded_machine_labels.py`.

- 70,037 detected songs had no human label (citizen archive 25.5k, BirdNET detections 25.3k, focal
  recordings 15.3k, passive recorders 4.0k). The fold models labelled them; a label was kept only
  if it was confident (>= 0.9), allowed by the recording's file- or site-level label, and
  confirmed by the decoded element sequence.
- **42,655 songs passed with all teachers** (59,052 with at least one fold's teachers): 28.5k
  Unfinished, 4,337 BhBl, 3,569 BC, 2,376 BE, 2,262 None, 1,250 XB, 328 BlBh.
- Spot check by eye, 40 accepted songs of the rare classes from both the archive and the BirdNET
  detections: 36-37 clearly right, the others faint rather than plainly wrong.
- Leakage is excluded by construction: fold k trains on labels written by teachers that never saw
  fold k (`label_fold<k>`); only the final model uses the labels of all teachers.
- Nothing existing was modified; the result is the new table `data/songs_machine_labelled.csv`.

| Run | Accuracy | Macro-F1 |
|---|---|---|
| reference (two seeds) | 0.963, 0.961 | 0.915, 0.921 |
| `expanded_machine_labels` (round 1) | 0.956 | 0.910 |
| `expanded_machine_labels_round2` | queued; see the runs table | |

On this test set the extra labels do not help: the labelled clips already cover the archive, and a
model cannot learn much from songs it was already confident about. Their value is elsewhere and is
not measurable here: 25k of them are BirdNET-app detections, i.e. phone recordings from other
countries, the domain the app works in. They are therefore part of the shipped recipe, where the
cross-validated cost of the whole recipe is reported.

**Continuous cutting of the passive-recorder archive** (never storing the raw audio) needs the raw
recordings, which are not on this machine, and a detector with very few missed songs. The stored
end-to-end number (song recall 0.875 on focal recordings) is not good enough for "cut and delete";
the model's own p(song) on a dense grid of windows is the natural detector to test first.

## Augmentation

| Run | What is added | Accuracy | Macro-F1 |
|---|---|---|---|
| reference | speed +-2 %, noise 0-20 dB, null splices, SpecAugment | 0.963, 0.961 | 0.915, 0.921 |
| `augment_real_world` | far-field simulation, other species in the background, device EQ | 0.960 | 0.916 |
| `augment_background_songs` | a second, quieter Yellowhammer (label-safe form of two-song mixup) | queued; see the runs table | |

The field augmentations cost nothing on the archive test set, which contains little far-field or
phone audio and so cannot show their benefit; they are in the shipped recipe. Two songs are only
mixed in the label-safe form: the added song is unfinished, 5-20 dB quieter, and never added to a
"None" clip. Mixing two dialects is still excluded, because the label of the mixture is undefined.

## Backlog: ideas that are experimentally sound and not yet run

Ordered by what I would try next.

1. **Completeness as detection with pasted endings.** Real terminal elements pasted after real
   phrases at controlled levels (+10 to -25 dB): unlimited exact supervision for the one decision
   that still fails (faint C / E), and a psychometric curve as an evaluation.
2. **Shape-primitive tokens** shared across species (see "Limits of generic"); needs a second
   species to mean anything.
3. **Fusing the two readings.** The sequence likelihoods as an input to the class head, or the head
   trained to agree with them, instead of two outputs that are only compared afterwards.
4. **Per-song answers in training.** Windows with two songs are labelled as one; the decoder
   already splits songs, so the CTC target could allow `song song` sequences.
5. **Individual identity for "diarization".** `experiments/identity_embedding.py` (same ring-coded
   male across years as positives), then clustering songs per recording; a synthetic test by
   interleaving songs of held-out males. Queued last; the code was never trained before.
6. **A learned song detector** from the frame-level outputs (p(song) per frame), evaluated end to
   end on whole recordings; required for the "cut and delete" pipeline above.
7. Instantaneous-frequency or constant-Q input channels; a relational head comparing the last two
   elements; quantisation-aware training and int8 export; hyper-parameters (learning rate,
   epochs, pool shares, sequence-loss weight) were never tuned.

Not repeated on the new models, although the scripts exist under `evaluation/research_session/`:
bird- and recording-level accuracy, the expert BE / BlBh bird test, end-to-end evaluation on
whole recordings, calibration and conformal sets, TFLite parity and phone latency.

## What is still open

- **An expert pass over the audit** (`label_audit/clip_label_audit.csv`). The audit applied the
  written rule strictly but was not done by an ornithologist.
- **Phone recordings.** Every number here is on archive recordings (dictaphones, 2011-2022) and
  BirdNET-app clips. A set of confirmed segments from the app database is still the missing test.
- **X-only and B-only songs.** "Unfinished called XB" is the largest remaining confusion (28
  clips): songs with an X element and no B. Four such clips were found filed under XB among the
  40 least confident agreeing XB clips and relabelled (audit round 3); more may hide among the
  confident ones. The element sequence makes these cases visible (`phrase X`).
- **Two songs in one window.** The label rule ("a complete song anywhere in the window?" vs "the
  last song?") is not defined by the project; the service avoids the case by cutting one window
  per detected song.
- Everything listed under "blocked on the owner" in `CONTINUATION.md` section 9.
