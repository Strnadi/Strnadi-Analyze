"""One layer variant per file. Conventions:

  waveform      (B, N)        float32, mono, 32 kHz
  spectrogram   (B, C, F, T)  frequency before time
  frame series  (B, D, T)

Composition helpers live in `layers/registry.py`; nothing here imports from `model/`.
"""
