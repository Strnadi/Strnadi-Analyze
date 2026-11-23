"""Smoke tests: every layer variant and every model builds and runs a forward/backward pass on random audio.

    python -m tests.test_models            (or: pytest tests/)

No data and no network needed: pretrained weights are switched off; models that need an external download
(Perch torch port, EfficientAT) are only tested in their offline (`precomputed`) form or skipped.
"""

import numpy as np
import torch

from model import build_model, count_params
from layers.registry import build_frontend

torch.manual_seed(0)
WAVE = torch.randn(2, 128000) * 0.1


def test_frontends():
    for kw in [
        dict(),
        dict(norm='percentile'),
        dict(norm='meanstd'),
        dict(norm='pcen'),
        dict(norm='magma'),
        dict(norm='delta'),
        dict(bank='linear', n_bins=192, fmin=2000, fmax=9500, n_fft=2048),
        dict(n_fft=[512, 1024, 4096], norm='percentile'),
    ]:
        fe = build_frontend(**kw)
        s = fe(WAVE)
        assert s.shape == (2, fe.out_channels, fe.n_bins, 400) and torch.isfinite(s).all(), kw


def test_stft_paths_agree():
    fe = build_frontend()
    a = fe(WAVE)
    fe.stft.use_fft = False
    b = fe(WAVE)
    assert (a - b).abs().max() < 1e-3, float((a - b).abs().max())


def test_mel_matches_librosa():
    try:
        import librosa
    except ImportError:
        return
    from layers.mel_filterbank import mel_weights

    ref = librosa.filters.mel(
        sr=32000, n_fft=1024, n_mels=128, fmin=1500, fmax=11000, htk=True, norm='slaney'
    )
    assert np.abs(ref - mel_weights(32000, 1024, 128, 1500, 11000)).max() < 1e-6


def test_spec_augment_and_mixstyle():
    from layers.spec_augment import SpecAugment
    from layers.freq_mixstyle import FreqMixStyle

    s = build_frontend()(WAVE)
    for layer in (SpecAugment(), SpecAugment(fill='min'), FreqMixStyle(p=1.0)):
        layer.train()
        out = layer(s)
        assert out.shape == s.shape and torch.isfinite(out).all()
        layer.eval()
        assert torch.equal(layer(s), s)


WAVE_MODELS = {
    'strnadnet': {},
    'strnadnet_factorized': ('strnadnet', dict(head='factorized')),
    'strnadnet_cosine': ('strnadnet', dict(head='cosine', head_kw=dict(k=2))),
    'strnadnet_gru': ('strnadnet', dict(temporal='gru')),
    'strnadnet_attn': ('strnadnet', dict(temporal='attn')),
    'strnadnet_dw': ('strnadnet', dict(dw=True)),
    'strnadnet_pcen_linear': (
        'strnadnet',
        dict(
            frontend=dict(bank='linear', norm='pcen', n_bins=128, fmin=2000, fmax=9500, n_fft=2048)
        ),
    ),
    'strnadnet_tonal': {},
    'strnadnet_measure': {},
    'timm_temporal': dict(backbone='mobilenetv3_small_050', pretrained=False),
    'patch_transformer': {},
    'contour_tcn': {},
    'contour_ridge': (
        'contour_tcn',
        dict(
            features='ridge',
            frontend=dict(
                bank='linear', norm='percentile', n_bins=192, fmin=2000, fmax=9500, n_fft=2048
            ),
        ),
    ),
    'densenet121_attnpool': dict(pretrained=False),
    'densenet_small': dict(blocks=(2, 2, 2, 2)),
    'mobilenetv3_delta': dict(pretrained=False),
    'densenet_crnn_embed': dict(pretrained=False),
    'bc_resnet': {},
    'fdy_crnn': {},
    'ending_zoom': {},
    'element_tagger': {},
}


def test_classifiers():
    for key, spec in WAVE_MODELS.items():
        name, kw = spec if isinstance(spec, tuple) else (key, spec)
        m = build_model(name, n_classes=7, **kw)
        m.train()
        out = m(WAVE)
        assert out['logits'].shape == (2, 7), key
        out['logits'].logsumexp(1).sum().backward()
        if getattr(m, 'frontend', None) is not None and not getattr(m, 'wave_input', False):
            m.eval()
            spec_in = m.frontend(WAVE)
            assert torch.allclose(m(spec_in)['logits'], m(WAVE)['logits'], atol=1e-4), (
                key
            )  # waveform and spectrogram entry agree
        print(f'{key:24s} {count_params(m) / 1e6:6.2f} M params')


def test_factorized_head_is_normalised():
    m = build_model('strnadnet', head='factorized').eval()
    assert torch.allclose(m(WAVE)['logits'].exp().sum(1), torch.ones(2), atol=1e-5)


def test_other_models():
    for name in ('conv_autoencoder', 'mobile_autoencoder'):
        out = build_model(name)(WAVE)
        assert out['recon'].shape == out['target'].shape and out['emb'].shape == (2, 64), name
    det = build_model('song_detector')
    out = det(WAVE)
    assert out['frame_logits'].shape == (2, 200)
    assert isinstance(det.eval().detect(WAVE[0]), list)
    tag = build_model('element_tagger')(WAVE)
    assert tag['element_logits'].shape == (2, 7, 200)
    mil = build_model('recording_mil')
    out = mil(
        torch.randn(3, 9, 128), torch.tensor([[1] * 9, [1] * 4 + [0] * 5, [1] + [0] * 8]).bool()
    )
    assert out['logits'].shape == (3, 6) and torch.allclose(
        out['att'].sum(1), torch.ones(3), atol=1e-5
    )
    assert build_model('perch_mlp')(torch.randn(2, 1536))['logits'].shape == (2, 7)
    assert build_model('perch_temporal')(torch.randn(2, 16, 1536))['logits'].shape == (2, 7)
    from layers.prototype_head import PrototypeHead
    from layers.arcface_head import ArcFaceHead

    assert PrototypeHead(160, 7)(torch.randn(2, 160, 100))['logits'].shape == (2, 7)
    assert ArcFaceHead(128, 7)(torch.randn(2, 128), torch.tensor([1, 3]))['logits'].shape == (2, 7)


def test_decode_elements():
    from model.element_tagger import decode_elements

    tags = np.array([0] * 10 + [1] * 60 + [0] * 3 + [2] * 15 + [0] * 2 + [2] * 15 + [0] * 20)
    ridge = np.r_[np.zeros(73), np.full(15, 5200.0), np.zeros(2), np.full(15, 4700.0), np.zeros(20)]
    assert decode_elements(tags, ridge)[0] == 'BhBl' and decode_elements(tags, ridge[::-1].copy())[
        0
    ] in ('BhBl', 'BlBh')
    assert decode_elements(np.array([0] * 10 + [1] * 60 + [0] * 30))[0] == ''


if __name__ == '__main__':
    torch.set_num_threads(1)
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f()
            print('ok', k)
