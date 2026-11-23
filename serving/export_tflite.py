"""Convert an exported ONNX model to TFLite for the phone app.

    python -m serving.export_onnx   <model.pt | run_dir> model.onnx      # PyTorch environment
    python -m serving.export_tflite model.onnx [out_dir]                 # environment with onnx2tf + TensorFlow
    python -m tests.check_tflite    <model.pt | run_dir> out_dir/*.tflite   # PyTorch + ai-edge-litert

Writes float32, float16 and dynamic-range int8 variants. Which to ship (measured by the research
session on R2 fold 0, 400 clips, research log section 17.9):

    float16   3.3 MB, same argmax as PyTorch on every clip: the deploy format
    float32   6.5 MB, same predictions
    int8      2.0 MB, 1.5 % of clips change class and it is slower on x86: needs full-integer
              quantisation with a calibration set before use

Three things matter and are fixed here:

- the input is static, one clip of 4 s (`waveform:1,128000`). With a dynamic batch dimension a
  traced reshape in the graph cannot be allocated by the TFLite runtime. The backend keeps using
  the ONNX file, which stays dynamic;
- the converter works on a copy of the ONNX file: onnx2tf overwrites its input with a graph fixed to
  one clip, which the backend (batches of windows) can no longer run;
- the TensorFlow converter backend (`-tb tf_converter`). The default backend of recent onnx2tf
  versions writes the flatbuffer directly and produced graphs that fail to allocate.

This module imports neither torch nor the rest of the repository, so it runs in a TensorFlow-only
environment.
"""

import glob
import os
import shutil
import subprocess
import sys

CLIP_SAMPLES = 128000  # 4 s at 32 kHz


def convert(onnx_path, out_dir):
    """Run onnx2tf; return the paths of the .tflite files it wrote.

    onnx2tf overwrites its input file with a simplified graph that has the static input shape.
    The backend needs the original, dynamic-batch file, so the converter gets a copy.
    """
    converter = shutil.which('onnx2tf') or os.path.join(os.path.dirname(sys.executable), 'onnx2tf')
    os.makedirs(out_dir, exist_ok=True)
    onnx_copy = os.path.join(out_dir, os.path.basename(onnx_path))
    shutil.copy(onnx_path, onnx_copy)
    try:
        return _run_converter(converter, onnx_copy, out_dir)
    finally:
        os.remove(onnx_copy)


def _run_converter(converter, onnx_path, out_dir):
    command = [
        converter,
        '-i',
        onnx_path,
        '-o',
        out_dir,
        '-odrqt',  # also write the dynamic-range quantised variant
        '-ois',
        f'waveform:1,{CLIP_SAMPLES}',
        '-tb',
        'tf_converter',
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        raise SystemExit(f'onnx2tf failed with exit code {result.returncode}')
    return sorted(glob.glob(os.path.join(out_dir, '*.tflite')))


def main():
    onnx_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(onnx_path)[0] + '_tflite'
    for path in convert(onnx_path, out_dir):
        print(f'{os.path.getsize(path) / 1e6:6.2f} MB  {path}')
    print(
        'now check the result: python -m tests.check_tflite <model.pt | run_dir> <file.tflite> ...'
    )


if __name__ == '__main__':
    main()
