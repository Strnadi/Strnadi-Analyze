#!/usr/bin/env python3
import os
import glob
import time
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration (mirror densenet_simclr.py)
WORKSPACE = "/workspace"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
VALIDATION_SPLIT = 0.3

def collect_files(dataset_dir):
    audio_files = []
    labels = []
    subdirs = [f for f in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, f))]
    class_names = sorted(subdirs)
    class_indices = {name: i for i, name in enumerate(class_names)}

    for subdir in subdirs:
        class_dir = os.path.join(dataset_dir, subdir)
        class_idx = class_indices[subdir]
        for ext in AUDIO_EXTENSIONS:
            pattern = os.path.join(class_dir, f"*{ext}")
            for path in glob.glob(pattern):
                audio_files.append(path)
                labels.append(class_idx)

    return audio_files, np.array(labels, dtype=np.int32), class_names

def collect_dataset(dataset_dir, validation_split=VALIDATION_SPLIT, train = False, val = False, test = False, seed=None):
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    audio_files, labels, class_names = collect_files(dataset_dir)

    if len(audio_files) == 0:
        raise SystemExit(f"No audio files found under {dataset_dir}")

    split_time = seed if seed is not None else int(time.time())

    # Train vs (val+test)
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files,
        labels,
        test_size=validation_split,
        stratify=labels,
        random_state=split_time,
    )

    # Val vs test
    val_files, test_files, val_labels, test_labels = train_test_split(
        val_files,
        val_labels,
        test_size=0.33,
        stratify=val_labels,
        random_state=split_time + 1,
    )

    return_files, return_labels = [], []

    if train:
        return_files.extend(train_files)
        return_labels.extend(train_labels)

    if val:
        return_files.extend(val_files)
        return_labels.extend(val_labels)

    if test:
        return_files.extend(test_files)
        return_labels.extend(test_labels)

    return return_files, return_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    files, labels = collect_dataset(dataset_dir=args.dataset_dir, validation_split=VALIDATION_SPLIT, train=args.train, val=args.val, test=args.test, seed=args.seed)
    for file, label in zip(files, labels):
        print(file)
    # print(labels)
