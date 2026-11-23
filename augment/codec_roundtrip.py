"""Encode + decode with a random phone / upload codec (AAC, Opus, MP3 at app-like bitrates) through ffmpeg.
Phone recordings are the domain the models are evaluated on least. Falls back to the input on any error.
Origin: research session (R5)."""
import subprocess
import numpy as np
from .fill import fill_to

CODECS = [('aac', 'adts', '32k'), ('aac', 'adts', '48k'), ('aac', 'adts', '64k'), ('libopus', 'ogg', '16k'), ('libopus', 'ogg', '24k'),
          ('libopus', 'ogg', '32k'), ('libmp3lame', 'mp3', '32k'), ('libmp3lame', 'mp3', '64k')]
_FFMPEG = None


def _ffmpeg():
    global _FFMPEG
    if _FFMPEG is None:
        try:
            import imageio_ffmpeg
            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            _FFMPEG = 'ffmpeg'
    return _FFMPEG


def codec_roundtrip(y, sr=32000):
    codec, fmt, bitrate = CODECS[np.random.randint(len(CODECS))]
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()
    try:
        enc = subprocess.run([_ffmpeg(), '-v', 'quiet', '-f', 's16le', '-ar', str(sr), '-ac', '1', '-i', '-', '-c:a', codec, '-b:a', bitrate,
                              '-f', fmt, '-'], input=pcm, stdout=subprocess.PIPE, timeout=10).stdout
        dec = subprocess.run([_ffmpeg(), '-v', 'quiet', '-i', '-', '-f', 's16le', '-ar', str(sr), '-ac', '1', '-'], input=enc,
                             stdout=subprocess.PIPE, timeout=10).stdout
        z = np.frombuffer(dec, np.int16).astype(np.float32) / 32768
        return y if len(z) < len(y) // 2 else fill_to(z, len(y))
    except Exception:
        return y
