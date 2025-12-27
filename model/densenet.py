import os
import sys
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import math
import glob
import keras
import librosa
import sklearn
import imblearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import cache
from pathlib import Path

from ..layers.global_gem2d import GlobalGeMPool2D
from ..layers.mel_to_magma import mel_to_magma


LOCAL_WORKSPACE = '/content'
WORKSPACE = '/content/drive/MyDrive/strnadi-data'

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
REMOTE_DATASET = os.path.join(WORKSPACE, 'new-data.zip')
DATASET = os.path.join(LOCAL_WORKSPACE, 'dataset-v3.zip')
DATASET_DIR = os.path.join(LOCAL_WORKSPACE, 'dataset-v3')
SAMPLE_RATE, SAMPLE_SECONDS = 48000, 4
BATCH_SIZE = 32


## @cache
def load_and_normalize_audio(file_path, target_sr=SAMPLE_RATE):
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


def audio_generator(files, labels, class_names, shuffle):
    """Generator that yields audio chunks and labels on demand"""
    num_classes = len(class_names)
    indices = list(range(len(files)))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices:
        file_path: str = files[idx]
        label = labels[idx]  # This is the integer label

        # One-hot encode the label for loss calculation
        one_hot = np.zeros(num_classes)
        one_hot[label] = 1
        audio = load_and_normalize_audio(file_path)

        if audio is not None:
            # Yield ((audio_input, integer_label_input), one_hot_label_for_loss)
            # yield ((audio, label), one_hot)
            yield audio, one_hot


def compute_class_weights(labels, class_names):
    """
    Compute class weights inversely proportional to class frequencies
    """
    # Count samples per class
    class_counts = np.bincount(labels, minlength=len(class_names))

    # Calculate weights inversely proportional to counts
    total_samples = len(labels)
    class_weights = {}

    for i, count in enumerate(class_counts):
        if count > 0:
            # Formula: total_samples / (num_classes * samples_in_class)
            class_weights[i] = total_samples / (len(class_names) * count)

    return class_weights


def load_data(directory, validation_split=0.3, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create a TensorFlow dataset from audio files in directory
    """
    audio_files = []
    class_names = []
    labels = []

    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    class_names = sorted(subdirs)
    class_indices = {name: i for i, name in enumerate(class_names)}

    # For each class directory, find all audio files
    for subdir in subdirs:
        class_dir = os.path.join(directory, subdir)
        class_idx = class_indices[subdir]

        for ext in AUDIO_EXTENSIONS:
            pattern = os.path.join(class_dir, f"*{ext}")
            audio_paths = glob.glob(pattern)

            for path in audio_paths:
                audio_files.append(path)
                labels.append(class_idx)

    # Calculate class weights before splitting
    class_weights = compute_class_weights(labels, class_names)
    split_time = int(time.time())

    # Split into train and validation sets
    train_files, val_files, train_labels, val_labels = sklearn.model_selection.train_test_split(
        audio_files, labels, test_size=validation_split, stratify=labels, random_state=split_time
    )

    val_files, test_files, val_labels, test_labels = sklearn.model_selection.train_test_split(
        val_files, val_labels, test_size=0.33, stratify=val_labels, random_state=split_time+1
    )

    print(f"Dataset seed: {split_time}, {split_time+1}")
    print(f"Found {len(audio_files)} audio files in {len(class_names)} classes")
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files, testing on {len(test_files)} files")

    # Define output signature for the generator:
    # ( (audio_input_spec, integer_label_input_spec), one_hot_label_target_spec )
    output_signature = (
        # (
            tf.TensorSpec(shape=(SAMPLE_RATE * SAMPLE_SECONDS,), dtype=tf.float32),
            # tf.TensorSpec(shape=(), dtype=tf.int32)
        # )
        # ,
        tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)
    )

    # Create TensorFlow datasets using generators
    train_dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(train_files, train_labels, class_names, shuffle=shuffle),
        output_signature=output_signature
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(val_files, val_labels, class_names, shuffle=False),
        output_signature=output_signature
    )

    # Apply batching and prefetching and caching
    train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE).cache()
    val_dataset   = val_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE).cache()

    train_steps    = math.floor(len(train_files) / batch_size)
    val_steps      = math.floor(len(val_files)   / batch_size)
    return train_dataset, val_dataset, test_files, test_labels, class_names, class_weights, train_steps, val_steps


ds_train, ds_validate, test_files, test_labels, class_names, class_weights, train_steps, val_steps = load_data(DATASET_DIR)
num_classes = len(class_names)

# inv_max = (1.0/max(list(class_weights.values())))
# class_weights = {k: v * inv_max for k,v in class_weights.items()}
print("Class weights:", class_weights)


def make_model():
    inp = keras.Input(shape=(SAMPLE_SECONDS * SAMPLE_RATE,))
    spect = keras.layers.MelSpectrogram(
        sampling_rate=SAMPLE_RATE,
        min_freq=3000,
        max_freq=9000,
        num_mel_bins=224,
        fft_length=1024,
        power_to_db=True
    )(inp)
    image = keras.layers.Lambda(mel_to_magma)(spect)

    cnn = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling=None
    )

    x = cnn(image)
    x = GlobalGeMPool2D()(x)

    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.3)(x)

    outp = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inp, outputs=outp)


model = make_model()

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=keras.losses.CategoricalCrossentropy(),#from_logits=True),
    metrics=[
        keras.metrics.F1Score(average="weighted", name="f1_score"),
        keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2"),
        keras.metrics.AUC(curve="PR", name="auc")
    ]
)

model.summary()

EPOCHS = 100

# Make checkpoint dir
current_time = int(time.time())
checkpoint_dir = os.path.join(WORKSPACE, 'checkpoints', str(current_time))
tensorboard_dir = os.path.join(WORKSPACE, 'tensorboard', str(current_time))
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

backup_dir = os.path.join(WORKSPACE, 'training-backups', str(current_time))

history = model.fit(
    ds_train,
    validation_data=ds_validate,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    # steps_per_epoch=train_steps,
    # validation_steps=val_steps,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '{epoch}-{val_f1_score:.4f}-{val_loss:.4f}.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_freq="epoch"
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.BackupAndRestore(backup_dir, double_checkpoint=True),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir
        )
    ]
)
