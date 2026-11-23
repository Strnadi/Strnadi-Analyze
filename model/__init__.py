"""One model per file. Every model:

- takes a mono 32 kHz waveform (B, N), or an already computed spectrogram (B, C, F, T) from its own
  `.frontend` (the trainer applies spectrogram augmentation in between); models with `wave_input = True`
  need the waveform itself;
- returns a dict with `logits` (B, n_classes) (may be log-probabilities) and, where it has them, `emb`
  (clip embedding), `att` (frame weights), `frames` (B, D, T), `frame_logits`, plus model-specific extras.

`build_model(name, **kwargs)` instantiates by registry name; `experiments/*.py` refer to models by that name.
"""
import importlib

MODELS = {
    # research session
    'strnadnet': ('strnadnet', 'StrnadNet'),
    'timm_temporal': ('timm_temporal', 'TimmTemporal'),
    'patch_transformer': ('patch_transformer', 'PatchTransformer'),
    'contour_tcn': ('contour_tcn', 'ContourTCN'),
    # legacy repo (ported from Keras)
    'densenet121_attnpool': ('densenet121_attnpool', 'DenseNet121AttnPool'),
    'densenet_small': ('densenet_small', 'DenseNetSmall'),
    'mobilenetv3_delta': ('mobilenetv3_delta', 'MobileNetV3Delta'),
    'densenet_crnn_embed': ('densenet_crnn_embed', 'DenseNetCRNNEmbed'),
    'conv_autoencoder': ('conv_autoencoder', 'ConvAutoencoder'),
    'mobile_autoencoder': ('mobile_autoencoder', 'MobileAutoencoder'),
    'perch_mlp': ('perch_mlp', 'PerchMLP'),
    # review proposals
    'perch_temporal': ('perch_temporal', 'PerchTemporal'),
    'efficientat_temporal': ('efficientat_temporal', 'EfficientATTemporal'),
    'strnadnet_tonal': ('strnadnet_tonal', 'TonalStrnadNet'),
    'strnadnet_measure': ('strnadnet_measure', 'MeasurementStrnadNet'),
    'bc_resnet': ('bc_resnet', 'BCResNet'),
    'fdy_crnn': ('fdy_crnn', 'FDYCRNN'),
    'ending_zoom': ('ending_zoom', 'EndingZoomNet'),
    'element_tagger': ('element_tagger', 'ElementTagger'),
    'song_detector': ('song_detector', 'SongDetector'),
    'recording_mil': ('recording_mil', 'RecordingMIL'),
}


def build_model(name, **kwargs):
    module, cls = MODELS[name]
    return getattr(importlib.import_module(f'model.{module}'), cls)(**kwargs)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
