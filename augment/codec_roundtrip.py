"""Encode + decode with a random phone / upload codec (AAC, Opus, MP3 at app-like bitrates)
through ffmpeg.

Phone recordings are the domain the models are evaluated on least. Falls back to the input on any
error.

Origin: research session (R5).
"""

import subprocess
import numpy as np
from .fill import fill_to

# The codecs to choose from: (ffmpeg audio encoder, container format, bitrate).
CODECS = [
    ('aac', 'adts', '32k'),
    ('aac', 'adts', '48k'),
    ('aac', 'adts', '64k'),
    ('libopus', 'ogg', '16k'),
    ('libopus', 'ogg', '24k'),
    ('libopus', 'ogg', '32k'),
    ('libmp3lame', 'mp3', '32k'),
    ('libmp3lame', 'mp3', '64k'),
]

# Path of the ffmpeg executable; looked up on first use.
_FFMPEG = None


def _ffmpeg():
    """Return the ffmpeg executable to run, looking it up once.

    Returns:
        The binary shipped with the `imageio_ffmpeg` package when that package is installed,
        otherwise the plain command name 'ffmpeg' (found through PATH).
    """
    global _FFMPEG
    if _FFMPEG is None:
        try:
            import imageio_ffmpeg

            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            _FFMPEG = 'ffmpeg'
    return _FFMPEG


def codec_roundtrip(y, sr=32000):
    """Compress a clip with a randomly chosen lossy codec and decode it again.

    The codec is drawn from `CODECS` with the global `np.random` state. Audio is passed to and
    from ffmpeg through pipes, no files are written.

    Args:
        y: waveform, 1-D float32 array with values in -1 ... 1 (larger values are clipped).
        sr: sample rate of `y` in Hz.

    Returns:
        The decoded float32 waveform with the same number of samples as `y`. The input `y` itself
        is returned when anything fails (ffmpeg missing, time-out, ...) or when less than half of
        the clip came back.
    """
    codec_index = np.random.randint(len(CODECS))
    codec, container_format, bitrate = CODECS[codec_index]

    # ffmpeg reads the clip as raw 16-bit integer samples.
    clipped = np.clip(y, -1, 1)
    pcm_bytes = (clipped * 32767).astype(np.int16).tobytes()

    try:
        ffmpeg = _ffmpeg()
        quiet = ['-v', 'quiet']
        # Description of the raw audio: signed 16-bit little-endian, `sr` Hz, one channel.
        raw_pcm = ['-f', 's16le', '-ar', str(sr), '-ac', '1']
        read_stdin = ['-i', '-']
        write_stdout = ['-']

        # Step 1: raw samples in, compressed stream out.
        compression = ['-c:a', codec, '-b:a', bitrate, '-f', container_format]
        encode_command = [ffmpeg, *quiet, *raw_pcm, *read_stdin, *compression, *write_stdout]
        encoded = subprocess.run(
            encode_command,
            input=pcm_bytes,
            stdout=subprocess.PIPE,
            timeout=10,  # seconds
        ).stdout

        # Step 2: compressed stream in, raw samples out.
        decode_command = [ffmpeg, *quiet, *read_stdin, *raw_pcm, *write_stdout]
        decoded = subprocess.run(
            decode_command,
            input=encoded,
            stdout=subprocess.PIPE,
            timeout=10,  # seconds
        ).stdout

        # Back from 16-bit integers to floats in -1 ... 1.
        decoded_waveform = np.frombuffer(decoded, np.int16).astype(np.float32) / 32768

        # Much less audio than went in means the round trip failed: keep the original.
        if len(decoded_waveform) < len(y) // 2:
            return y
        # The decoded length can differ from the input: cut or pad back to the original length.
        return fill_to(decoded_waveform, len(y))
    except Exception:
        return y
