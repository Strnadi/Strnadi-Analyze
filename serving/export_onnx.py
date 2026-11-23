"""Export a trained model (one checkpoint, or the fold ensemble of a run averaged into one graph) to ONNX and
check the exported graph against PyTorch. Replaces the legacy `convert.py` (Keras -> TFLite).

Deploy graph: waveform (B, 128000) @ 32 kHz -> probs (B, 7), emb (B, D)
- peak normalisation is inside the graph (the service sends raw decoded audio);
- every STFT runs as conv1d, so only standard ops are used.
Parity is checked twice: the export graph against the training-path model (FFT STFT), and ONNX Runtime against
the export graph. (The research exporter only checked the second; the audit asked for both.)

For phones: ONNX Runtime Mobile runs this file directly; a TFLite build can be produced from it with `onnx2tf`,
after which the same parity check must be repeated on the .tflite (not done here).

Usage: python -m serving.export_onnx <model.pt | run_dir> <out.onnx>
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from training.checkpoint import load_model, run_checkpoints, export_mode


class Deploy(nn.Module):
    def __init__(self, models):
        super().__init__(); self.models = nn.ModuleList(models)

    def forward(self, wave):
        wave = wave / wave.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) * 0.5
        probs, emb = 0, 0
        for m in self.models:
            out = m(wave); probs = probs + torch.softmax(out['logits'], 1); emb = emb + out['emb']
        return probs / len(self.models), emb / len(self.models)


def main():
    spec, out_path = sys.argv[1], sys.argv[2]
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    dep = Deploy([load_model(p) for p in run_checkpoints(spec)]).eval()
    x = torch.randn(2, 128000) * 0.1
    with torch.no_grad():
        ref_fft = dep(x)
        export_mode(dep, True); ref = dep(x)
    d_path = float((ref_fft[0] - ref[0]).abs().max())
    print('FFT path vs export (conv) path, max |dp|:', d_path)
    torch.onnx.export(dep, (x,), out_path, input_names=['waveform'], output_names=['probs', 'emb'],
                      dynamic_axes={'waveform': {0: 'batch'}, 'probs': {0: 'batch'}, 'emb': {0: 'batch'}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    x5 = torch.randn(5, 128000) * 0.1                      # a batch size other than the traced one: catches baked-in shapes
    with torch.no_grad():
        ref = dep(x5)
    got = sess.run(None, {'waveform': x5.numpy()})
    d = float(np.abs(got[0] - ref[0].numpy()).max())
    print(f'ONNX Runtime vs PyTorch, max |dp|: {d:.2e}; size {os.path.getsize(out_path) / 1e6:.1f} MB; '
          f'params {sum(p.numel() for p in dep.parameters()) / 1e6:.2f} M')
    assert d < 1e-3 and d_path < 1e-3, 'exported graph does not match the trained model'


if __name__ == '__main__':
    main()
