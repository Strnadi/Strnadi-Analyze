import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import librosa

NUM_COLORS = 256
LUT = tf.constant(
    plt.get_cmap("magma", NUM_COLORS)(np.arange(NUM_COLORS))[:, :3].astype("float32"),
    dtype=tf.float32
)

WORKSPACE = '/workspace'

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
DATASET_DIR = os.path.join(WORKSPACE, 'dataset')
SAMPLE_RATE, SAMPLE_SECONDS = 48000, 4
BATCH_SIZE = 32


def mel_to_magma(t):
    # t: (B,T,F) or (B,T,F,1)  →  (B,T,F,3)
    if t.shape.rank == 4 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1) # (B,T,F)

    t_min = tf.reduce_min(t, axis=[1, 2], keepdims=True)
    t_max = tf.reduce_max(t, axis=[1, 2], keepdims=True)
    t_norm = (t - t_min) / (t_max - t_min + 1e-6)  # [0,1]

    idx = tf.cast(tf.round(t_norm * (NUM_COLORS - 1)), tf.int32)
    return tf.gather(LUT, idx)  # (B,T,F,3)

class GlobalGeMPool2D(keras.layers.Layer):
    def __init__(self, p_init=3.0, **kwargs):
        super().__init__(**kwargs);
        self.p = tf.Variable(p_init, dtype=tf.float32)

    def call(self, t):
        t = tf.maximum(t, 1e-6)
        return tf.pow(tf.reduce_mean(tf.pow(t, self.p), axis=[1, 2]), 1./self.p)

# class AttentiveStatsPool(keras.layers.Layer):
#     def build(self, shape):
#         self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")
#
#     def call(self, t):
#         # t: (B, T, F, C)  ➜ flatten freq
#         x = tf.reshape(t, (tf.shape(t)[0], -1, t.shape[-1]))   # (B, T, C)
#         alpha = tf.nn.softmax(tf.squeeze(tf.matmul(x, self.w), -1))# (B, T)
#         mu = tf.reduce_sum(x * alpha[..., None], axis=1)
#         sigma = tf.sqrt(tf.reduce_sum(alpha[..., None] * tf.square(x-mu[:,None,:]), axis=1)+1e-9)
#         return tf.concat([mu, sigma], axis=-1)                 # (B, 2C)

class AttentiveStatsPool(keras.layers.Layer):
    def build(self, shape):
        self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")

    def call(self, t):
        # t: (B, T, F, C) ➜ flatten freq
        # We use tf.shape(t)[0] for the batch size to handle dynamic batch sizes safely
        shape = tf.shape(t)
        x = tf.reshape(t, (shape[0], -1, t.shape[-1]))   # (B, T, C)

        # Calculate alpha
        # Result of matmul is (B, T, 1), squeeze makes it (B, T)
        dot_product = tf.matmul(x, self.w)
        alpha = tf.nn.softmax(tf.squeeze(dot_product, axis=-1)) # (B, T)

        # FIX: Use tf.expand_dims instead of alpha[..., None]
        # This replaces the complex StridedSlice with a native TFLite ExpandDims op
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # (B, T, 1)

        # Calculate Mu
        mu = tf.reduce_sum(x * alpha_expanded, axis=1)   # (B, C)

        # FIX: Use tf.expand_dims instead of mu[:, None, :]
        mu_expanded = tf.expand_dims(mu, axis=1)         # (B, 1, C)

        # Calculate Sigma
        # (x - mu_expanded) broadcasts correctly now
        squared_diff = tf.square(x - mu_expanded)
        weighted_squared_diff = tf.reduce_sum(alpha_expanded * squared_diff, axis=1)
        sigma = tf.sqrt(weighted_squared_diff + 1e-9)

        return tf.concat([mu, sigma], axis=-1)           # (B, 2C)

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


    dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(audio_files, labels, class_names, shuffle=shuffle),
        output_signature=(
            tf.TensorSpec(shape=(SAMPLE_RATE * SAMPLE_SECONDS,), dtype=tf.float32),
            tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)
        )
    ).batch(batch_size)

    return dataset

def representative_dataset():
    dataset = load_data(DATASET_DIR, batch_size=1, shuffle=False)

    for audio_batch, _ in dataset.unbatch().batch(1):
        yield [tf.cast(audio_batch, tf.float32)]

model = keras.saving.load_model("good-model-2.keras", custom_objects={"mel_to_magma": mel_to_magma, "GlobalGeMPool2D": GlobalGeMPool2D, "AttentiveStatsPool": AttentiveStatsPool})
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.float16]
converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable LiteRT ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

tflite_model = converter.convert()

with tf.io.gfile.GFile('good-model-2_opt.tflite', 'wb') as f:
  f.write(tflite_model)
