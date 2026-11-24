# Visual audit of the clip labels (2026-11-24)

The earlier reviews suspected that a large part of the classifier's "errors" were wrong labels.
Nobody had looked. This audit did: every clip on which the models disagreed with its label was
rendered as a spectrogram and judged by eye.

**The audit was done by the ML agent, not by an ornithologist.** It applies the project's own
written rule strictly (below). An expert pass over the same clips is still worth having:
`label_audit/clip_label_audit.csv` is the list, with the original label, the verdict and what the
models said.

## What was reviewed

- Seven independently trained recipes of the research session (`base_v3`, `r1`, `r2`, `r4`-`r7`)
  were averaged per clip, out of fold.
- Their average disagreed with the label for **441 of 3,744 clips** (11.8 %). All 441 were reviewed.
- Each clip was drawn at 62 Hz / 4 ms resolution, 2-10 kHz (3-13 kHz for a second look at songs
  with an X element), **with 1.5 s of the source recording before and 2 s after the clip**, so that
  an ending cut by the clip edge can be told from an ending the bird left out.
- As a control, 75 random clips on which models and label agree were viewed the same way:
  74 labels were right (one "XB" clip shows an X element and no B).

## The rule

A song is a phrase of repeated syllables followed by terminal elements. The dialect is the
arrangement of the last two elements:

| Class | After the phrase |
|---|---|
| BC | a B whistle (~5 kHz), then a C element: a faint, long, falling whistle from ~9 to ~6.5 kHz, 0.3-1.1 s after B |
| BE | B, then an E element: a short rising whistle around 7-8 kHz |
| BhBl | a B whistle, then a second, lower one (3.5-4.5 kHz) |
| BlBh | a B whistle, then a second, higher one (6-7 kHz, flat) |
| XB | an X element (short, 7-8 kHz, several harmonic bands), then B |
| Unfinished | a Yellowhammer phrase with no terminal element, or with only ONE (a lone B or a lone X) |
| None | no Yellowhammer song in the window |

The label describes **what is in the window**, not the bird. A bird whose other songs are BC still
gives an "Unfinished" clip when this song stops after B.

## Verdicts

| Verdict | Clips | Meaning |
|---|---|---|
| relabelled | 209 | the spectrogram plainly shows another class |
| confirmed | 119 | the label is right; the models were wrong |
| unjudgeable | 113 | cannot be decided: the terminal element would lie under noise or another bird, the clip cuts the song, two birds overlap, or the song is too faint. Removed from training and evaluation |

Where the 209 corrections went:

| Original label | Clips reviewed | Relabelled | To |
|---|---|---|---|
| BC | 82 | 54 | Unfinished (phrase + B, no C) |
| BE | 37 | 23 | Unfinished |
| BhBl | 70 | 50 | 48 Unfinished (one B only), 1 BlBh, 1 None |
| BlBh | 31 | 15 | 14 Unfinished, 1 None |
| None | 62 | 35 | Unfinished (a Yellowhammer phrase is visible) |
| Unfinished | 129 | 14 | 6 XB, 3 BE, 2 BC, 3 None |
| XB | 30 | 18 | Unfinished (B only, or X only) |

So the dominant label error is the one the reviews predicted: **a first phrase plus one terminal
element, filed under the bird's dialect.** It is 152 of the 209 corrections, and it hit the rare
classes hardest: a third of all BhBl clips (48 of 144) were single-B songs.

## Effect on the numbers

The same stored predictions, scored against the original and the audited labels:

| Run | Accuracy, original labels | Accuracy, audited | Macro-F1, original | Macro-F1, audited |
|---|---|---|---|---|
| base_v3 (clip labels only) | 0.868 | 0.937 | 0.745 | 0.851 |
| r2 (weak pools + factorised head) | 0.874 | 0.949 | 0.774 | 0.894 |
| average of the seven runs | 0.882 | 0.965 | - | 0.924 |

These models were *trained on* the wrong labels and are scored here on a set from which 113
hard clips were removed, so the audited columns are not a result: they show that the label noise,
not the architecture, set the old ceiling. Results of models trained on the audited labels are in
`docs/RESULTS.md`.

## Caveats

- Only disagreements were reviewed. Errors that every model reproduces (e.g. a consistent habit of
  the labellers) are invisible to this method; the 75-clip control puts their rate near 1 %.
- Removing 113 unjudgeable clips makes the test set easier. They are kept in
  `data/clips_unjudgeable.csv`, and results are also reported with them counted as errors.
- "Unfinished" now includes many faint or distant songs whose terminal element may exist below the
  noise. For the scientists this is the right default (an unfinished song is never used to assign
  a dialect), but it means the class is "no dialect visible", not "the bird left it out".

## Reproduce

    python -m processing.apply_label_audit      # writes data/clips_audited.csv with new folds

Review sheets are produced by `evaluation/review_sheets.py`.
