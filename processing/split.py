#!/usr/bin/env python3
"""
Recursively convert all audio files to 48kHz 16-bit PCM WAV
and preserve the directory structure in the output folder.

Equivalent of the provided bash script:
- Finds files with extensions: .mp3, .flac, .ogg, .wav (case-insensitive)
- For each file, builds the output path under the output directory with a .wav extension
- Calls the conversion script (by default the same path used in the bash script)

Usage:
    python3 convert_all.py /path/to/input /path/to/output

Options:
    --script PATH    Path to the conversion script to run on each file
    --dry-run        Print actions without invoking conversion
"""

import shutil
from pathlib import Path
import argparse
import re
import subprocess
import sys
import os
import requests
from typing import *
import logging
import time
import numpy as np
import random
import ffmpeg
import tempfile
import datetime

import librosa

from tqdm.auto import tqdm
from scipy.io.wavfile import write
from scipy.ndimage import shift

EXTENSIONS = {".mp3", ".flac", ".ogg", ".wav"}

def load_and_normalize_audio(file_path, target_sr=48000):
    """
    Load audio file in various formats and normalize it
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if audio.ndim > 1:
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        audio = librosa.util.normalize(audio)
        return audio

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# def measure_rms(in_path: str, start: float, end: float) -> float:
#     """Returns RMS level (dB) of an audio file using ffmpeg-python."""
#     try:
#         stream = (
#             ffmpeg
#             .input(in_path)
#             .trim(start=start, end=end)
#             .audio
#             .filter('highpass', f=9000)
#             .filter('volumedetect')
#             .output('null', f='null')
#         )
#         _, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
#         match = re.search(r"mean_volume:\s*(-?\d+\.\d+)", err)
#         if match:
#             return float(match.group(1))
#     except Exception:
#         pass
#     return -25.0

def measure_rms(in_path: str, start: float, end: float) -> float:
    """Returns RMS level (dB) of a wav file."""
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", str(datetime.timedelta(seconds=start)), "-to", str(datetime.timedelta(seconds=end)), "-i", in_path,
        "-af", f"highpass=f={3000},lowpass=f={9000},volumedetect",
        "-vn", "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    match = re.search(r"mean_volume:\s*(-?\d+\.\d+)", proc.stderr)
    if match:
        return float(match.group(1))

    print(f"Warning: could not measure RMS for {in_path}")
    return -25.0  # fa

def ffmpeg_augment_recording(in_path: str, out_path: str, duration: float, total_duration: float, prefix_length: float, padding_length: float, fade_in: float = 0.3, fade_out: float = 0.06) -> None:
    rms = measure_rms(in_path, prefix_length, prefix_length + duration)
    volume_factor = (10 ** (rms / 50.0)) * 0.15
    volume_factor = max(0.005, min(volume_factor, 0.05))
    noise_fade_in = fade_in * 1.5
    noise_fade_out = fade_out * 1.5

    audio = (
        ffmpeg
        .input(in_path)
        .audio
        .filter('aresample', 48000)
        # .filter('pan', 'mono|c0=0.5*c0')
        # .filter('afade', t='in', st=max(0.0, prefix_length - fade_in), d=fade_in)
        # .filter('afade', t='out', st=max(0.0, prefix_length + duration - fade_out), d=fade_out)
    )

    noise_prefix = (
        ffmpeg
        .input(f'aevalsrc=random(0):d={duration}:s=48000:channel_layout=mono', f='lavfi')
        .filter('atrim', start=0, end=prefix_length)
        .filter('volume', volume_factor)
        # .filter('afade', t='in', st=0, d=noise_fade_in)
        # .filter('afade', t='out', st=max(0.0, prefix_length - noise_fade_out), d=noise_fade_out)
        # .filter('adelay', delays=f'{fade_in}|{fade_in}')
    )

    padding_delay_ms = int((prefix_length + duration) * 1000)
    noise_padding = (
        ffmpeg
        .input(f'aevalsrc=random(0):d={padding_length}:s=48000:channel_layout=mono', f='lavfi')
        .filter('atrim', start=0, end=padding_length)
        .filter('volume', volume_factor)
        # .filter('afade', t='in', st=0, d=noise_fade_in)
        # .filter('afade', t='out', st=max(0.0, padding_length - noise_fade_out), d=noise_fade_out)
        .filter('adelay', delays=f'{padding_delay_ms}|{padding_delay_ms}')
    )

    mixed = ffmpeg.filter([audio, noise_prefix, noise_padding], 'amix', inputs=3, duration='first', dropout_transition=0)
    out = ffmpeg.output(mixed, out_path, ac=1, ar=48000, acodec='pcm_s16le').overwrite_output()
    ffmpeg.run(out, quiet=True)


def get_yellowhammer_intervals(wav_bytes :bytes) -> List[Tuple[float, float]]:
    url = os.environ.get("BIRDNET_URL", "http://localhost:32808/process")
    response = requests.post(url, files={"file": ("audio.wav", wav_bytes, "audio/wav")})
    response.raise_for_status()

    json = response.json()
    return json['segments']


def find_audio_files(root: Path):
    """Yield Path objects for audio files under root (recursively)."""

    file_list = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in EXTENSIONS:
                file_list.append(Path(dirpath) / fn)

    return file_list

def find_window(interval, wav_len, padding: float = 1.0):
    """Return a window of length (interval span + 2*padding) within the audio.

    If the desired window is longer than the audio, the full clip is returned.
    Handles inverted intervals and keeps the requested window length by
    shifting it inside the audio bounds instead of shrinking the padding.
    """
    if wav_len <= 0:
        raise ValueError("wav_len must be positive")

    try:
        int_start, int_end = float(interval[0]), float(interval[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("interval must contain two numeric values") from exc

    padding = max(0.0, float(padding))

    if int_start > int_end:
        int_start, int_end = int_end, int_start

    base_length = max(0.0, int_end - int_start)
    desired_length = base_length + 2 * padding

    if desired_length >= wav_len:
        return 0.0, float(wav_len)

    preferred_start = int_start - padding
    bounded_start = min(max(preferred_start, 0.0), wav_len - desired_length)
    window_start = bounded_start
    window_end = window_start + desired_length

    return window_start, window_end

def main():
    p = argparse.ArgumentParser(description="Recursively convert audio files and preserve directory structure")
    p.add_argument("in_dir", nargs="?", default='.', help="input folder (default=current directory)")
    p.add_argument("out_dir", nargs="?", default='converted', help="output folder (default=./converted)")
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not in_dir.exists():
        print(f"Input directory does not exist: {in_dir}", file=sys.stderr)
        sys.exit(2)

    # Walk and process
    for infile in tqdm(find_audio_files(in_dir), desc="Processing files"):
        try:
            relpath = infile.relative_to(in_dir)
        except Exception:
            # Fallback if infile is not under in_dir for some reason
            relpath = infile.name

        outfile = out_dir / relpath.with_suffix('.wav')
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # print(f"Converting: {infile} -> {outfile}")

        audio = load_and_normalize_audio(infile, target_sr=48000)
        wav_len = len(audio) / 48000.0

        # If longer than 5 seconds, split
        if wav_len > 5.0:
            try:
                with open(infile, 'rb') as f:
                    intervals = get_yellowhammer_intervals(f.read())

            except Exception as e:
                print(f"Error during segmentation of {infile}: {e}", file=sys.stderr)
                continue

            i = 0
            for start, end in intervals:
                window_start, window_end = find_window((start, end), wav_len, padding=1.0)

                start_5sec = window_start
                end_5sec = window_end

                trimmed = audio[int(start_5sec * 48000):int(end_5sec * 48000)].copy()

                if abs(end_5sec - start_5sec) < 5.0:
                    total_padding = 5.0 - (end_5sec - start_5sec)
                    pad_start = random.uniform(0, total_padding)
                    pad_start_samples = int(pad_start * 48000)

                    trimmed.resize(trimmed.shape[0] + int(total_padding * 48000))
                    trimmed = shift(trimmed, pad_start_samples, cval=0)

                trimmed_int16 = np.int16(trimmed * 32767)
                write(f"{outfile}.{i}.wav", 48000, trimmed_int16)

                if abs(end_5sec - start_5sec) < 5.0:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
                        ffmpeg_augment_recording(f"{outfile}.{i}.wav", fp.name, end_5sec - start_5sec, 5.0, pad_start, total_padding - pad_start)
                        fp.flush()

                        shutil.move(fp.name, f"{outfile}.{i}.wav")

                i += 1

        elif wav_len < 5.0:
            total_padding = 5.0 - wav_len
            pad_start = random.uniform(0, total_padding)
            pad_start_samples = int(pad_start * 48000)

            trimmed = audio[int(0 * 48000):int(wav_len * 48000)].copy()
            trimmed.resize(trimmed.shape[0] + int(total_padding * 48000))

            trimmed = shift(trimmed, pad_start_samples, cval=0)
            trimmed_int16 = np.int16(trimmed * 32767)
            write(f"{outfile}.wav", 48000, trimmed_int16)

            with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
                ffmpeg_augment_recording(f"{outfile}.wav", fp.name, wav_len, 5, pad_start, total_padding-pad_start)
                fp.flush()

                shutil.move(fp.name, f"{outfile}.wav")

        else:
            trimmed = audio[int(0 * 48000):int(wav_len * 48000)].copy()
            trimmed_int16 = np.int16(trimmed * 32767)
            write(f"{outfile}.wav", 48000, trimmed_int16)


if __name__ == '__main__':
    main()
