import ffmpeg
import numpy as np
import wave
import io


SAMPLE_RATE = 48000 # 48 kHz

def wav_length_from_bytes(data: bytes) -> float:
    with wave.open(io.BytesIO(data), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def ffmpeg_filter(in_bytes: bytes, filters: str, output_format="wav") -> bytes:
    """Apply arbitrary ffmpeg audio filters to in-memory WAV data."""
    ff = (
        ffmpeg
        .input('pipe:0', format='wav')
        .output('pipe:1', format=output_format, af=filters, acodec='pcm_s16le')
    )

    process = ff.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    out_bytes, err = process.communicate(input=in_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {err.decode()}")
    return out_bytes


def trim_audio_to_np_float(in_bytes: bytes, start_sec: float, end_sec: float, pad_len=None):
    """
    Trim in-memory WAV between start_sec and end_sec and return PCM with values in range -1, 1
    """
    filters = f"atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS"
    # Output format 's16le' = raw PCM16 little-endian, no header
    pcm_bytes = ffmpeg_filter(in_bytes, filters, output_format="s16le")
    # bytes to numpy
    samples = np.frombuffer(pcm_bytes, dtype='<i2')
    # 16 bit ints to floats
    float_samples = samples.astype(np.float32) / 32768.0

    if((pad_len is not None) and (pad_len * SAMPLE_RATE > len(float_samples))):
        float_samples.resize(pad_len * SAMPLE_RATE) # resize to right size, implicitly padded with zeros at end

    return float_samples


def normalize_audio(in_bytes: bytes) -> bytes:
    """Normalize, resample to 48kHz mono, apply high/low pass filters, outputs WAV (with headers)"""
    filters = (
        "loudnorm,"
        f"aresample={SAMPLE_RATE},"
        "pan=mono|c0=.5*c0+.5*c1,"
        "highpass=f=3500,"
        "lowpass=f=8500"
    )
    return ffmpeg_filter(in_bytes, filters, output_format="wav")


