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
import datetime
import io
from ai_edge_litert.interpreter import Interpreter

import librosa

from tqdm.auto import tqdm
from scipy.io.wavfile import write
from scipy.ndimage import shift

EXTENSIONS = {".mp3", ".flac", ".ogg", ".wav"}
SAMPLE_RATE = 48000


def to_wav_bytes(audio: np.ndarray, sample_rate: int = 48000) -> bytes:
    """Serialize mono int16 audio to WAV bytes."""
    buf = io.BytesIO()
    write(buf, sample_rate, audio)
    return buf.getvalue()

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

def measure_rms(wav_bytes: bytes, start: float, end: float) -> float:
    """Returns RMS level (dB) for the provided WAV bytes."""
    cmd = [
        "ffmpeg", "-hide_banner", "-ss", str(datetime.timedelta(seconds=round(start, 5))), "-to", str(datetime.timedelta(seconds=round(end, 5))), "-i", "pipe:0",
        "-af", f"highpass=f={3000},lowpass=f={9000},volumedetect",
        "-vn", "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, input=wav_bytes, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    match = re.search(r"mean_volume:\s*(-?\d+\.\d+)", proc.stderr.decode("utf-8", errors="ignore"))
    if match:
        return float(match.group(1))

    print("Warning: could not measure RMS for provided audio bytes")
    return -25.0  # fallback level


def time_stretch_to_duration(wav_bytes: bytes, current_duration: float, target_duration: float = 5.0) -> bytes:
    """
    Time-stretch audio to target duration without changing pitch.
    Prefers ffmpeg rubberband (higher quality, wider tempo range); falls back to atempo if unavailable.
    """
    if current_duration <= 0 or target_duration <= 0:
        return wav_bytes

    tempo = current_duration / target_duration  # <1 slows down, >1 speeds up

    # Build base stream
    stream = (
        ffmpeg
        .input('pipe:0')
        .audio
        .filter('aresample', 48000)
    )

    try:
        # Rubberband handles large tempo changes and keeps formants more naturally
        rb_stream = stream.filter('rubberband', tempo=f'{tempo:.6f}', formant='preserved')
        out_stream = ffmpeg.output(rb_stream, 'pipe:1', ac=1, ar=48000, acodec='pcm_s16le', format='wav').overwrite_output()
        out_bytes, stderr = ffmpeg.run(out_stream, input=wav_bytes, capture_stdout=True, capture_stderr=True)
        return out_bytes
    except ffmpeg.Error as e:
        # Fallback to atempo chain if rubberband is not available
        err_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        if 'No such filter: "rubberband"' not in err_msg:
            print(f"Rubberband error: {err_msg}")
        
        remaining_tempo = tempo
        fallback = stream
        while remaining_tempo < 0.5:
            fallback = fallback.filter('atempo', 0.5)
            remaining_tempo /= 0.5
        while remaining_tempo > 2.0:
            fallback = fallback.filter('atempo', 2.0)
            remaining_tempo /= 2.0
        fallback = fallback.filter('atempo', f'{remaining_tempo:.6f}')

        out_stream = ffmpeg.output(fallback, 'pipe:1', ac=1, ar=48000, acodec='pcm_s16le', format='wav').overwrite_output()
        try:
            out_bytes, stderr = ffmpeg.run(out_stream, input=wav_bytes, capture_stdout=True, capture_stderr=True)
            return out_bytes
        except ffmpeg.Error as e2:
            print(f"Time stretch error (fallback): {e2.stderr.decode('utf-8', errors='ignore') if e2.stderr else str(e2)}")
            return wav_bytes

def pad_recording(wav_bytes: bytes, duration: float, total_duration: float, prefix_length: float, padding_length: float, fade_in: float = 0.5, fade_out: float = 0.5) -> bytes:
    """
    Blend audio with noise padding using crossfades for seamless spectrogram transitions.
    
    The input wav_bytes has silence at the start (prefix_length) and end (padding_length).
    This function fills that silence with noise that crossfades smoothly with the audio.
    """
    try:
        rms = measure_rms(wav_bytes, prefix_length, prefix_length + duration)
        volume_factor = (10 ** (rms / 50.0)) * 0.15
        volume_factor = max(0.005, min(volume_factor, 0.05))
        
        # Crossfade duration - where noise and audio overlap
        crossfade = prefix_length
        crossfade_out = padding_length

        # Main audio with crossfade envelopes at boundaries
        # Audio starts at prefix_length, so fade in there; fade out before padding starts
        audio = (
            ffmpeg
            .input('pipe:0')
            .audio
            .filter('aresample', 48000)
            .filter('afade', t='in', st=f"{max(0.0, prefix_length - crossfade):.5f}", d=f"{(crossfade * 2):.5f}", curve='qsin')
            .filter('afade', t='out', st=f"{max(0.0, prefix_length + duration - crossfade_out):.5f}", d=f"{(crossfade_out * 2):.5f}", curve='qsin')
        )

        # Noise prefix: fills [0, prefix_length], fades out as audio fades in
        noise_prefix = (
            ffmpeg
            .input(f'anoisesrc=d={prefix_length + crossfade:.5f}:c=pink:s=48000', f='lavfi')
            .filter('volume', f"{volume_factor:.8f}")
            .filter('afade', t='out', st=f"{max(0.0, prefix_length - crossfade):.5f}", d=f"{(crossfade * 2):.5f}", curve='qsin')
        )

        # Noise padding: fills [prefix_length + duration, end], fades in as audio fades out
        padding_delay_ms = int((prefix_length + duration - crossfade_out) * 1000)
        noise_padding = (
            ffmpeg
            .input(f'anoisesrc=d={padding_length + crossfade_out:.5f}:c=pink:s=48000', f='lavfi')
            .filter('volume', f"{volume_factor:.8f}")
            .filter('afade', t='in', st=0, d=f"{(crossfade_out * 2):.5f}", curve='qsin')
            .filter('adelay', delays=f'{padding_delay_ms}|{padding_delay_ms}')
        )

        mixed = ffmpeg.filter([audio, noise_prefix, noise_padding], 'amix', inputs=3, duration='first', dropout_transition=0)
        out_stream = ffmpeg.output(mixed, 'pipe:1', ac=1, ar=48000, acodec='pcm_s16le', format='wav').overwrite_output()
        out_bytes, stderr = ffmpeg.run(out_stream, input=wav_bytes, capture_stdout=True, capture_stderr=False)
        return out_bytes
    except Exception as e:
        print(f"Error during ffmpeg augmentation: {e}")
        if 'stderr' in locals():
            print("FFmpeg stderr:", stderr.decode("utf-8", errors="ignore"))
        raise e

def ffmpeg_augment_recording(wav_bytes: bytes, duration: float, total_duration: float, prefix_length: float, padding_length: float, fade_in: float = 0.5, fade_out: float = 0.5) -> bytes:
    """
    Blend audio with noise padding using crossfades for seamless spectrogram transitions.
    
    The input wav_bytes has silence at the start (prefix_length) and end (padding_length).
    This function fills that silence with noise that crossfades smoothly with the audio.
    """
    return time_stretch_to_duration(wav_bytes, duration, total_duration)
    # return pad_recording(wav_bytes, duration, total_duration, prefix_length, padding_length, fade_in, fade_out)

def chunk_audio(audio, clip_length=3.0, step=0.5, target_sr=SAMPLE_RATE):
    for i in range(0, len(audio), int(step * target_sr)):
        chunk = audio[i:i + int(clip_length * target_sr)]

        if len(chunk) < int(clip_length * target_sr):
            padding = int(clip_length * target_sr) - len(chunk)
            chunk = np.pad(chunk, (0, padding), 'constant')

        start_s = i / target_sr
        end_s = start_s + clip_length
        yield chunk, start_s, end_s


def process_audio(audio, batch_size=8, thread_count=8):
    interpreter = Interpreter(model_path="audio-model.tflite", num_threads=thread_count)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], [batch_size, 144000])
    interpreter.allocate_tensors()

    prediction = [] # [(start, end, label, confidence)]

    # Collect chunks into batches
    batch_chunks = []
    batch_times = []

    with open("labels/en_us.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    for chunk, start, end in chunk_audio(audio):
        batch_chunks.append(chunk)
        batch_times.append((start, end))

        if len(batch_chunks) == batch_size:
            # Process full batch
            interpreter.set_tensor(input_details[0]['index'], np.array(batch_chunks))
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            for i, (start, end) in enumerate(batch_times):
                predicted = zip(labels, output_data[i])

                yellowhammer = list(filter(lambda x: x[0] == 'Emberiza citrinella_Yellowhammer', predicted))[0]

                if yellowhammer[1] > 0.4:
                    prediction.append((start, end, yellowhammer[1]))

            batch_chunks = []
            batch_times = []

    # Process remaining chunks (partial batch)
    if batch_chunks:
        # Pad batch to full size with zeros
        while len(batch_chunks) < batch_size:
            batch_chunks.append(np.zeros(144000, dtype=np.float32))
            batch_times.append(None)

        interpreter.set_tensor(input_details[0]['index'], np.array(batch_chunks))
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        for i, times in enumerate(batch_times):
            if times is None:
                break
            start, end = times
            predicted = zip(labels, output_data[i])
            yellowhammer = list(filter(lambda x: x[0] == 'Emberiza citrinella_Yellowhammer', predicted))[0]

            if yellowhammer[1] > 0.4:
                prediction.append((start, end, yellowhammer[1]))

    return prediction

def merge_overlaps_simple(detections: list[tuple[float, float, float]], FALL_THRESHOLD=0.8):
    """
    Merge overlapping detections (start, end, dialect, confidence).
    Keeps the most confident one if they overlap above threshold.
    """
    detections.sort(key=lambda x: x[0])
    merged = []

    for det in detections:
        if not merged:
            merged.append(det)
            continue

        last = merged[-1]

        if(last[1] + 1 > det[0]):
            # Merge them — keep the one with higher confidence
            if det[2] > last[2]:
                merged[-1] = det
            else:
                if abs(det[2] - last[2]) >= FALL_THRESHOLD:
                    merged.append(det)
        else:
            merged.append(det)

    return merged

def get_yellowhammer_intervals(audio: np.ndarray) -> list[tuple[float, float, float]]:
    raw_segments = process_audio(audio, thread_count=8, batch_size=8)
    merged_segments = merge_overlaps_simple(raw_segments)
    return merged_segments


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
    for infile in tqdm(sorted(find_audio_files(in_dir)), desc="Processing files", file=sys.stdout):
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
                    intervals = get_yellowhammer_intervals(audio)

            except Exception as e:
                print(f"Error during segmentation of {infile}: {e}", file=sys.stderr)
                continue

            i = 0
            for start, end, confidence in intervals:
                window_start, window_end = find_window((start, end), wav_len, padding=1.0)

                start_5sec = window_start
                end_5sec = window_end

                trimmed = audio[int(start_5sec * 48000):int(end_5sec * 48000)].copy()
                clip_duration = end_5sec - start_5sec

                trimmed_int16 = np.int16(trimmed * 32767)
                wav_bytes = to_wav_bytes(trimmed_int16, sample_rate=48000)

                # Time-stretch to exactly 5 seconds if needed
                if abs(clip_duration - 5.0) > 0.01:
                    wav_bytes = time_stretch_to_duration(wav_bytes, clip_duration, 5.0)

                with open(f"{outfile}.{i}.wav", "wb") as fp:
                    fp.write(wav_bytes)

                i += 1

        elif wav_len < 5.0:
            trimmed = audio.copy()
            trimmed_int16 = np.int16(trimmed * 32767)
            wav_bytes = to_wav_bytes(trimmed_int16, sample_rate=48000)

            # Time-stretch to exactly 5 seconds
            wav_bytes = time_stretch_to_duration(wav_bytes, wav_len, 5.0)

            with open(f"{outfile}.wav", "wb") as fp:
                fp.write(wav_bytes)

        else:
            trimmed = audio[int(0 * 48000):int(wav_len * 48000)].copy()
            trimmed_int16 = np.int16(trimmed * 32767)
            write(f"{outfile}.wav", 48000, trimmed_int16)


if __name__ == '__main__':
    main()
