import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import keras
import scipy, scipy.io, scipy.signal
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import glob
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import partial


BATCH_SIZE = 8
SAMPLE_RATE = 16000
EPOCHS = 50
SAMPLE_SECONDS = 4
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
directory = "/home/hroudis/strnad-data"

class AudioSequence(keras.utils.Sequence):
    """
    Keras Sequence for audio data that splits files into fixed-length chunks
    and yields batches of (audio_chunk, one_hot_label).
    """
    def __init__(
        self,
        file_paths,
        labels,
        class_names,
        batch_size=8,
        sample_rate=16000,
        sample_seconds=4,
        shuffle=True
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.class_names = class_names
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_seconds = sample_seconds
        self.shuffle = shuffle
        self.num_classes = len(class_names)

        # Build a list of (file_path, label, chunk_idx) tuples
        self._prepare_indices()
        self.on_epoch_end()

    def _prepare_indices(self):
        self.indices = []
        target_frames = self.sample_rate * self.sample_seconds
        for fp, lbl in zip(self.file_paths, self.labels):
            try:
                info = sf.info(fp)
                total_frames = info.frames
            except:
                # fallback to loading audio if sf.info fails
                audio, _ = librosa.load(fp, sr=self.sample_rate, mono=True)
                total_frames = len(audio)
            num_chunks = int(np.ceil(total_frames / target_frames))
            for i in range(num_chunks):
                self.indices.append((fp, lbl, i))

    def __len__(self):
        # number of batches per epoch
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        # fetch batch of indices
        batch_inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_size = len(batch_inds)
        chunk_len = self.sample_rate * self.sample_seconds

        X = np.zeros((batch_size, chunk_len), dtype=np.float32)
        y = np.zeros((batch_size, self.num_classes), dtype=np.float32)

        for i, (fp, lbl, chunk_idx) in enumerate(batch_inds):
            # load & normalize
            try:
                audio, sr = librosa.load(fp, sr=self.sample_rate, mono=True)
                audio = librosa.util.normalize(audio)
            except:
                audio = np.zeros(chunk_len, dtype=np.float32)

            start = chunk_idx * chunk_len
            end = start + chunk_len

            if start >= len(audio):
                chunk = np.zeros(chunk_len, dtype=np.float32)
            else:
                seg = audio[start:end]
                if len(seg) < chunk_len:
                    padded = np.zeros(chunk_len, dtype=np.float32)
                    padded[: len(seg)] = seg
                    chunk = padded
                else:
                    chunk = seg

            X[i] = chunk
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[lbl] = 1.0
            y[i] = one_hot

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def compute_class_weights(labels, class_names):
    """
    Compute class weights inversely proportional to class frequencies
    """
    class_counts = np.bincount(labels, minlength=len(class_names))
    total = len(labels)
    weights = {}
    for i, c in enumerate(class_counts):
        if c > 0:
            weights[i] = total / (len(class_names) * c)
    return weights


def load_data(directory, validation_split=0.3, batch_size=8, shuffle=True):
    """
    Scans `directory` for class subfolders, builds
    train/validation AudioSequence instances.
    """
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]

    # enumerate class folders
    class_names = sorted(
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    )
    class_indices = {name: i for i, name in enumerate(class_names)}

    # collect file paths & labels
    files, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(directory, cls)
        for ext in AUDIO_EXTENSIONS:
            for fp in glob.glob(os.path.join(cls_dir, f"*{ext}")):
                files.append(fp)
                labels.append(class_indices[cls])

    # compute class weights
    class_weights = compute_class_weights(labels, class_names)

    # split
    train_f, val_f, train_l, val_l = train_test_split(
        files, labels, test_size=validation_split, stratify=labels, random_state=42
    )

    # build sequences
    seq_train = AudioSequence(
        train_f, train_l, class_names, batch_size=batch_size, shuffle=shuffle
    )
    seq_val = AudioSequence(
        val_f, val_l, class_names, batch_size=batch_size, shuffle=False
    )

    return seq_train, seq_val, class_names, class_weights


ds_train, ds_validate, class_names, class_weights = load_data(directory)
num_classes = len(class_names)


def build_model(sample_seconds=SAMPLE_SECONDS):
    audio_in = keras.Input(shape=(SAMPLE_RATE * sample_seconds,), name="audio_in")

    spectrogram_layer = keras.layers.MelSpectrogram(
        min_freq=4000,
        max_freq=9000,
        num_mel_bins=512,
        sequence_length=1024,
        sequence_stride=128,
        fft_length=4 * SAMPLE_RATE,
        sampling_rate=SAMPLE_RATE,
        power_to_db=True,
        window="hann",
        name="mel_spectrogram"
    )
    spectrogram = spectrogram_layer(audio_in)

    # spectrogram = StaticNoiseReducer(percentile=0.1, floor_margin_db=5.0, name="spectral_noise_gate")(spectrogram)

    # Add a channel dimension
    reshaped_spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)

    # Convert single-channel spectrogram to 3 channels for ResNet
    three_channel_spectrogram = keras.layers.Concatenate(axis=-1)(
        [reshaped_spectrogram, reshaped_spectrogram, reshaped_spectrogram]
    )

    # 3) ResNet50 for feature extraction
    # resnet_base = keras.applications.ResNet50V2(
    #     include_top=False,
    #     weights=None,
    #     pooling="avg"
    # )
    # feat_resnet = resnet_base(three_channel_spectrogram)

    effinet_base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    feat_efficient = effinet_base(three_channel_spectrogram)

    # # Encoder
    # encoder = keras.layers.Dense(128, activation="relu", name="encoder_1")(feat_resnet)
    # latent = keras.layers.Dense(64, activation="relu", name="latent")(encoder)

    # # Decoder
    # decoder = keras.layers.Dense(128, activation="relu", name="decoder_1")(latent)
    # reconstruction = keras.layers.Dense(feat_resnet.shape[-1], activation="linear", name="reconstruction")(decoder)

    # dialect_classify_hidden = keras.layers.Dense(64, activation="relu", name="bird_classify_hidden")(feat_resnet)
    dialect_classify = keras.layers.Dense(num_classes, activation="softmax", name="bird_classify")(feat_efficient)

    return keras.Model(
        inputs=audio_in,
        outputs=dialect_classify,
        # outputs={
        #     "dialect": dialect_classify,
        #     # "vector": latent,
        # }
    )


model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
    # loss={
    #     "dialect": keras.losses.CategoricalCrossentropy(from_logits=True),
    #     # "dialect_classifier": "categorical_crossentropy",
    #     # "reconstruction": "mse"
    # },
    # loss_weights={
    #     "dialect": 1.0,
    #     # "dialect_classifier": 1.0,
    #     # "reconstruction": 0.1
    # },
    # metrics={
    #     "bird_detection": ["accuracy", keras.metrics.AUC(name="auc")],
    #     # "dialect_classifier": ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    # }
)

model.summary()

keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=True
)


# Update the training code to work with our custom dataset
model.fit(
    ds_train,
    # validation_data=ds_validate,
    class_weight=class_weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        # keras.callbacks.TensorBoard(log_dir="./logs")
    ]
)

# Save the model
model.save("bird_dialect_model.keras")

# Also save some metadata about the classes for future inference
with open("model_metadata.json", "w") as f:
    json.dump(
        {
            "class_names": class_names,
            "sample_rate": SAMPLE_RATE,
            "sample_seconds": SAMPLE_SECONDS
        },
        f
    )
