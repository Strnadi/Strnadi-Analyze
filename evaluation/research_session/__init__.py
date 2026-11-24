"""Evaluation scripts of the research session of 2026-11-22/23, kept as they were.

They were written against the original clip labels and tables (`orig_index.csv`, `weak_v3.csv`, the
runs `base_v3`, `r1_s0`, `r2_s0`, ...) and produced the numbers in `docs/RESEARCH_LOG.md`. They
are terse research code: only formatted, never rewritten. The maintained tools are one level up:

    evaluation/report.py          metrics, confusion matrix, learning curves of a run
    evaluation/review_sheets.py   spectrogram sheets for judging labels by eye
    evaluation/explain_model.py   occlusion maps and the deletion test
    evaluation/compare_runs.py    paired bootstrap between two runs
"""
