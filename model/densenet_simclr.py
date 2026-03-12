import os
import sys
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

WORKSPACE = '.'

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
DATASET_DIR = os.path.join(WORKSPACE, '_dataset')
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
    inp = keras.Input(shape=(SAMPLE_SECONDS * SAMPLE_RATE,))
    
    # 1. Spectrogram Generation
    spect = keras.layers.MelSpectrogram(
        sampling_rate=SAMPLE_RATE,
        min_freq=3000,
        max_freq=9000,
        num_mel_bins=224,
        fft_length=1024,
        power_to_db=True
    )(inp)

    # 2. Augmentation (Crucial for Contrastive Learning)
    spect = SpecAugment(
        freq_mask_param=27,      
        time_mask_param=80,      
        num_freq_masks=2,        
        num_time_masks=2,        
        max_time_mask_ratio=0.2, 
        mask_value=0.0,
        name="spec_augment",
    )(spect)

    # Convert to 3-channel for ImageNet weights
    spect = keras.layers.Lambda(mel_to_magma)(spect)

    # 3. CNN Feature Extractor (Spatial/Frequency Shapes)
    cnn = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling=None 
    )
    x = cnn(spect)

    # --- CRNN ADAPTATION STARTS HERE ---
    
    # CNN output shape is (Batch, Time_Frames, Freq_Bins, Channels).
    # DenseNet downsamples spatial dimensions by a factor of ~32. 
    # If input is 224 mel bins, Freq_Bins is roughly 7. Channels is 1024.
    b, t, f, c = x.shape
    
    # 4. Collapse Frequency into Features to create a Time Sequence
    # We reshape to (Batch, Time_Frames, Features) so the RNN can read it left-to-right
    x = keras.layers.Reshape((-1, f * c))(x) 

    # 5. Temporal Sequence Modeling
    # A Bidirectional GRU tracks how the frequency shapes evolve over time
    x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(x)

    # 6. Pooling
    # Your AttentiveStatsPool will now pool across the time dimension of the RNN output
    # (Ensure your AttentiveStatsPool implementation accepts 3D inputs: batch, time, features)
    x = AttentiveStatsPool1D()(x)

    # 7. Projection Head (For Contrastive Learning)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation='silu')(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Final Embedding Layer (e.g., 128 dimensions)
    embeddings = keras.layers.Dense(128, activation=None, name='raw_embedding')(x)
    
    # 8. L2 Normalization (Crucial for Cosine Similarity/Clustering)
    outp = keras.layers.UnitNormalization(axis=-1)(embeddings)

    return keras.Model(inputs=inp, outputs=outp)


class ContrastiveLearner(keras.Model):
    def __init__(self, encoder, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            # Forward pass both views through your embedding model
            # training=True ensures SpecAugment and Dropout are active
            z_i = self.encoder(x, training=True) 
            z_j = self.encoder(x, training=True)
            
            # Calculate the contrastive loss
            loss = nt_xent_loss(z_i, z_j, self.temperature)

        # Calculate gradients and update weights
        gradients = tape.gradient(loss, self.encoder.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_weights))

        # Update and return metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # Unpack the data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass both views through your embedding model
        # training=True to ensure augmentation for testing views
        z_i = self.encoder(x, training=True) 
        z_j = self.encoder(x, training=True)
        
        # Calculate the contrastive loss
        loss = nt_xent_loss(z_i, z_j, self.temperature)

        # Update and return metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


base_embedding_model = make_model()

contrastive_model = ContrastiveLearner(
    encoder=base_embedding_model, 
    temperature=0.1
)

contrastive_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    # loss=keras.losses.CategoricalCrossentropy(),#from_logits=True),
    # metrics=[
    #     keras.metrics.F1Score(average="weighted", name="f1_score"),
    #     keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2"),
    #     keras.metrics.AUC(curve="PR", name="auc")
    # ]
)

contrastive_model.summary()

EPOCHS = 100

# Make checkpoint dir
current_time = int(time.time())
checkpoint_dir = os.path.join(WORKSPACE, 'checkpoints', str(current_time))
tensorboard_dir = os.path.join(WORKSPACE, 'tensorboard', str(current_time))
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

backup_dir = os.path.join(WORKSPACE, 'training-backups', str(current_time))

history = contrastive_model.fit(
    ds_train,
    # validation_data=ds_validate,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    # class_weight=class_weights,
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
