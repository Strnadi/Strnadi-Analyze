import os
import sys
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Perch v2.0: load via zoo and embed as a TF-graph layer so weights save with the model.
from perch_hoplite.zoo import hub as perch_hub

# Ensure project root is on sys.path so we can import sibling packages like 'layers'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
import seaborn

from layers.global_gem2d import GlobalGeMPool2D
# from layers.attentive_stats_pool import AttentiveStatsPool
from layers.mel_to_magma import mel_to_magma
from layers.densenet import DenseNet
from layers.spec_augument import SpecAugment


def compute_delta_features(spect, win=2):
    """
    Compute delta and delta-delta features from mel spectrogram along the time axis.
    spect: (B, T, F) — batch, time, freq
    Returns: (B, T, F, 2) — delta and delta-delta stacked as channels
    """
    # Central difference: delta[t] = (x[t+win] - x[t-win]) / (2*win)
    pad = win
    padded = tf.pad(spect, [[0, 0], [pad, pad], [0, 0]], mode="edge")
    delta = (padded[:, 2 * pad :, :] - padded[:, :-2 * pad, :]) / (2.0 * pad)

    # Delta-delta: apply same to delta
    delta_padded = tf.pad(delta, [[0, 0], [pad, pad], [0, 0]], mode="edge")
    delta_delta = (delta_padded[:, 2 * pad :, :] - delta_padded[:, :-2 * pad, :]) / (
        2.0 * pad
    )

    # Stack as channels and transpose to (B, F, T, 2) for CNN (freq=height, time=width)
    stacked = tf.stack([delta, delta_delta], axis=-1)
    # Transpose: (B, T, F, 2) -> (B, F, T, 2)
    return tf.transpose(stacked, [0, 2, 1, 3])


def compute_deltas(spec):
    """
    Computes Delta and Delta-Delta using fixed Convolutions (TFLite friendly).
    Replaces manual array slicing (StridedSlice) with tf.nn.conv2d.
    
    Input:  (Batch, Time, Freq, 1)
    Output: delta (B, T, F, 1), delta2 (B, T, F, 1)
    """
    
    # 1. Define Fixed Kernels for Finite Difference
    # Shape needed for Conv2D: (Kernel_Height, Kernel_Width, In_Channels, Out_Channels)
    # We want to slide over Time (Height), so Kernel is (3, 1, 1, 1)
    
    # First Derivative Kernel: [-0.5, 0, 0.5]
    # (Corresponds to (x[t+1] - x[t-1]) * 0.5)
    k_delta_vals = [-0.5, 0.0, 0.5]
    kernel_delta = tf.constant(k_delta_vals, dtype=spec.dtype)
    kernel_delta = tf.reshape(kernel_delta, [3, 1, 1, 1])
    
    # Second Derivative Kernel: [1, -2, 1]
    # (Corresponds to x[t+1] - 2x[t] + x[t-1])
    k_delta2_vals = [1.0, -2.0, 1.0]
    kernel_delta2 = tf.constant(k_delta2_vals, dtype=spec.dtype)
    kernel_delta2 = tf.reshape(kernel_delta2, [3, 1, 1, 1])

    # 2. Handle Padding (To match your 'SYMMETRIC' logic)
    # If we use padding='SAME' in conv2d, it uses Zeros. 
    # To keep your Symmetric padding, we pad explicitly first.
    # Pad 1 step on Time axis (axis 1)
    spec_pad = tf.pad(spec, [[0,0], [1,1], [0,0], [0,0]], mode='SYMMETRIC')

    # 3. Perform Convolution
    # stride=1, padding='VALID' (because we manually padded)
    delta = tf.nn.conv2d(spec_pad, kernel_delta, strides=[1, 1, 1, 1], padding='VALID')
    delta2 = tf.nn.conv2d(spec_pad, kernel_delta2, strides=[1, 1, 1, 1], padding='VALID')
    
    return delta, delta2

def add_physics_channels(spec):
    # Ensure input is 4D: (Batch, Time, Freq, 1)
    if len(spec.shape) == 3:
        x = tf.expand_dims(spec, axis=-1)
    else:
        x = spec

    # Compute features using the Conv method
    delta, delta2 = compute_deltas(x)
    
    # Concatenate: (Batch, Time, Freq, 3)
    return tf.concat([x, delta, delta2], axis=-1)


def nt_xent_loss(z_i, z_j, temperature=0.1):
    """
    Computes the NT-Xent loss for a batch of embeddings.
    
    Args:
        z_i: Embeddings for the first augmented view (Shape: [Batch, Dim])
        z_j: Embeddings for the second augmented view (Shape: [Batch, Dim])
        temperature: Controls the sharpness of the softmax distribution.
    """
    # 1. Combine all views into a single batch. 
    # If your batch size is N, z now has 2N elements.
    z = tf.concat([z_i, z_j], axis=0) 
    
    # 2. Compute pairwise cosine similarity.
    # Because z_i and z_j are already L2 normalized by your model, 
    # the dot product is exactly the cosine similarity.
    sim_matrix = tf.matmul(z, z, transpose_b=True)
    
    # 3. Scale by the temperature parameter
    sim_matrix = sim_matrix / temperature
    
    # 4. Build labels for the positive pairs.
    # For embedding i in z_i, its positive pair is at index i + batch_size in z.
    batch_size = tf.shape(z_i)[0]
    
    # Create pseudo-labels: [batch_size, ..., 2*batch_size-1, 0, ..., batch_size-1]
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    
    # 5. Mask out self-similarity (the main diagonal)
    # We do not want the model to compare an image to its exact self, 
    # only to its augmented pair and the negatives.
    LARGE_NUM = 1e9
    masks = tf.one_hot(tf.range(2 * batch_size), 2 * batch_size)
    logits = sim_matrix - (masks * LARGE_NUM)
    
    # 6. Calculate the standard Cross Entropy Loss
    # We treat the contrastive task as a classification problem where the 
    # "correct class" is the index of the augmented pair.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    
    return tf.reduce_mean(loss)

class AttentiveStatsPool1D(keras.layers.Layer):
    def build(self, shape):
        # shape[-1] is now the number of GRU features
        self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")

    def call(self, x):
        # Input 'x' from BiGRU: (Batch, Time_Steps, Features)
        # NO RESHAPE NEEDED! It is already a 1D sequence of features.
        
        # Calculate attention scores (alpha) for each time step
        dot_product = tf.matmul(x, self.w) 
        alpha = tf.nn.softmax(tf.squeeze(dot_product, axis=-1)) 

        # Expand dims for broadcasting
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  

        # Calculate Mu (weighted mean across time steps)
        mu = tf.reduce_sum(x * alpha_expanded, axis=1)   

        # Calculate Sigma (weighted standard deviation across time steps)
        mu_expanded = tf.expand_dims(mu, axis=1)         
        squared_diff = tf.square(x - mu_expanded)
        weighted_squared_diff = tf.reduce_sum(alpha_expanded * squared_diff, axis=1)
        sigma = tf.sqrt(weighted_squared_diff + 1e-9)    

        # Concatenate mean and standard deviation
        return tf.concat([mu, sigma], axis=-1)

WORKSPACE = '/workspace'

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
DATASET_DIR = os.path.join(WORKSPACE, 'dataset')
SAMPLE_RATE, SAMPLE_SECONDS = 48000, 4
BATCH_SIZE = 32

# Perch v2.0 expects 32 kHz, 5 s windows → 160000 samples per clip
PERCH_SAMPLE_RATE = 32000
PERCH_WINDOW_S = 5.0
PERCH_AUDIO_LENGTH = int(PERCH_SAMPLE_RATE * PERCH_WINDOW_S)  # 160000
PERCH_EMBEDDING_DIM = 1536


def _frame_audio_tf(audio: tf.Tensor, window_size_s: float, hop_size_s: float, sample_rate: int):
    """Frame audio along the last axis. audio: (B, T) -> (B, num_frames, frame_length)."""
    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    length = tf.shape(audio)[-1]
    pad_amount = tf.maximum(0, frame_length - length)
    audio = tf.pad(audio, [[0, 0], [0, pad_amount]])
    return tf.signal.frame(audio, frame_length, hop_length, pad_end=False)


def _normalize_audio_tf(framed_audio: tf.Tensor, target_peak: float):
    """Normalize framed audio to target_peak (mirrors zoo_interface.normalize_audio)."""
    if target_peak is None:
        return framed_audio
    x = framed_audio - tf.reduce_mean(framed_audio, axis=-1, keepdims=True)
    peak = tf.reduce_max(tf.abs(x), axis=-1, keepdims=True)
    x = tf.where(peak > 0, x / peak * target_peak, x)
    return x


class PerchEmbeddingLayer(keras.layers.Layer):
    """
    Keras layer that embeds Perch v2.0 inside the model using only TF ops.
    Perch weights are part of the graph and are saved with the model.

    Save with Perch embedded: model.save('path', save_format='tf').
    Load: keras.models.load_model('path', custom_objects={'PerchEmbeddingLayer': PerchEmbeddingLayer})
    """

    def __init__(
        self,
        model_path: str | None = None,
        tfhub_slug: str | None = None,
        tfhub_version: int | None = None,
        target_peak: float = 0.25,
        window_size_s: float = 5.0,
        hop_size_s: float = 5.0,
        sample_rate: int = 32000,
        time_pooling: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None and (tfhub_slug is None or tfhub_version is None):
            raise ValueError("Provide either model_path or both tfhub_slug and tfhub_version.")
        self.model_path = model_path
        self.tfhub_slug = tfhub_slug
        self.tfhub_version = tfhub_version
        self.target_peak = target_peak
        self.window_size_s = window_size_s
        self.hop_size_s = hop_size_s
        self.sample_rate = sample_rate
        self.time_pooling = time_pooling
        self.embedding_dim = PERCH_EMBEDDING_DIM
        self.trainable = False
        self._perch_model = None
        self._infer_fn = None

    def build(self, input_shape):
        if self._perch_model is None:
            if self.model_path:
                path = self.model_path
            else:
                path = perch_hub.resolve(self.tfhub_slug, self.tfhub_version)
            self._perch_model = tf.saved_model.load(path)
            self._infer_fn = self._perch_model.signatures["serving_default"]
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time)
        framed = _frame_audio_tf(
            inputs, self.window_size_s, self.hop_size_s, self.sample_rate
        )
        # framed: (B, num_frames, frame_length) -> flatten to (B*num_frames, frame_length)
        batch_size = tf.shape(framed)[0]
        num_frames = tf.shape(framed)[1]
        frame_length = tf.shape(framed)[2]
        rebatched = tf.reshape(framed, [-1, frame_length])
        normalized = _normalize_audio_tf(rebatched, self.target_peak)
        # Perch v2 signature: inputs -> dict with 'embedding' (N, 1536) or (N, 1, 1536)
        outputs = self._infer_fn(inputs=normalized)
        emb = outputs["embedding"]
        # Flatten to (N, D) then reshape to (B, num_frames, 1, D)
        emb = tf.reshape(emb, [-1, self.embedding_dim])
        emb = tf.reshape(emb, [batch_size, num_frames, 1, self.embedding_dim])
        # Time/channel pool to (B, embedding_dim)
        if self.time_pooling == "mean":
            emb = tf.reduce_mean(emb, axis=[1, 2])
        elif self.time_pooling == "max":
            emb = tf.reduce_max(emb, axis=[1, 2])
        elif self.time_pooling == "first":
            emb = emb[:, 0, 0, :]
        else:
            emb = tf.reduce_mean(emb, axis=[1, 2])
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_path": self.model_path,
            "tfhub_slug": self.tfhub_slug,
            "tfhub_version": self.tfhub_version,
            "target_peak": self.target_peak,
            "window_size_s": self.window_size_s,
            "hop_size_s": self.hop_size_s,
            "sample_rate": self.sample_rate,
            "time_pooling": self.time_pooling,
        })
        return config


## @cache

def load_and_normalize_audio(file_path, target_sr=32000, target_duration=5):
    """
    Load audio file in various formats and normalize it.
    Returns a 1D numpy array or None on error.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if audio.ndim > 1:
            # if multi-channel, average channels unless second channel is all zeros
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        if len(audio) < target_duration * target_sr:
            audio = np.pad(audio, (0, int(target_duration * target_sr - len(audio))), mode='constant')
        elif len(audio) > target_duration * target_sr:
            audio = audio[:int(target_duration * target_sr)]

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
        audio = load_and_normalize_audio(
            file_path, target_sr=PERCH_SAMPLE_RATE, target_duration=PERCH_WINDOW_S
        )

        if audio is not None:
            # Yield raw audio so Perch is embedded inside the Keras model (PerchEmbeddingLayer).
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

    output_signature = (
        tf.TensorSpec(shape=(PERCH_AUDIO_LENGTH,), dtype=tf.float32),
        tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32),
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
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache().shuffle(buffer_size=len(train_files))
    val_dataset   = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).cache().shuffle(buffer_size=len(val_files))

    train_steps    = math.floor(len(train_files) / batch_size)
    val_steps      = math.floor(len(val_files)   / batch_size)
    return train_dataset, val_dataset, test_files, test_labels, class_names, class_weights, train_steps, val_steps


ds_train, ds_validate, test_files, test_labels, class_names, class_weights, train_steps, val_steps = load_data(DATASET_DIR)
num_classes = len(class_names)

# inv_max = (1.0/max(list(class_weights.values())))
# class_weights = {k: v * inv_max for k,v in class_weights.items()}
print("Class weights:", class_weights)

def make_model():
    # Perch v2.0 embedded in the TF graph; save with model.save('path', save_format='tf').
    inp = keras.Input(shape=(PERCH_AUDIO_LENGTH,), dtype=tf.float32)
    x = PerchEmbeddingLayer(
        tfhub_slug=perch_hub.PERCH_V2_SLUG,
        tfhub_version=2,
        time_pooling="mean",
    )(inp)
    x = keras.layers.Dense(512, activation="gelu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inp, outputs=x)

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
    # class_weight=class_weights,
    # steps_per_epoch=train_steps,
    # validation_steps=val_steps,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '{epoch}-{val_loss:.4f}.keras'),
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
        # keras.callbacks.BackupAndRestore(backup_dir, double_checkpoint=True),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
)

true_labels, pred_labels = [], []

for audio_batch, one_hot in ds_validate:
    pred = model.predict(audio_batch, verbose=0)
    # argmax over classes (axis=1), not over flattened batch
    one_hot_np = one_hot.numpy() if hasattr(one_hot, 'numpy') else np.array(one_hot)
    pred_indices = np.argmax(pred, axis=1)
    true_indices = np.argmax(one_hot_np, axis=1)
    for i in range(len(pred_indices)):
        pred_label = class_names[pred_indices[i]]
        true_label = class_names[true_indices[i]]
        true_labels.append(true_label)
        pred_labels.append(pred_label)

cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=class_names)
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

plt.figure(figsize=(8, 6))
seaborn.heatmap(
    cm_norm,
    annot=cm,
    fmt="d",
    cmap="Blues",
    vmin=0, vmax=1,
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label": "Proportion of true class"}
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()