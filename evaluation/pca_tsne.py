import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import hdbscan
from scipy.spatial import ConvexHull
import tensorflow as tf
import keras
import librosa


SAMPLE_RATE = 48000
BATCH_SIZE = 8
# Set to 3 to enable 3D PCA visualization
PCA_N_COMPONENTS = 3

def load_audio(file_path, target_sr=48000):
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

        audio = librosa.util.normalize(audio)
        return audio

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def collect_dataset(data_root="data", valid_extensions=(".wav", ".mp3", ".flac", ".ogg", ".m4a")):
    """
    Collect audio file paths and labels from a folder structure:
    data_root/<class_name>/<audio_files>
    """
    audio_files = []
    true_labels = []

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    class_names = sorted(
        class_name
        for class_name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, class_name))
    )

    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        for file_name in sorted(os.listdir(class_dir)):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(valid_extensions):
                audio_files.append(file_path)
                true_labels.append(class_name)

    return audio_files, true_labels

NUM_COLORS = 256
LUT = tf.constant(
    plt.get_cmap("magma", NUM_COLORS)(np.arange(NUM_COLORS))[:, :3].astype("float32"),
    dtype=tf.float32
)

@keras.saving.register_keras_serializable()
def mel_to_magma(t):
    # t: (B,T,F) or (B,T,F,1)  →  (B,T,F,3)
    if t.shape.rank == 4 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1) # (B,T,F)

    t_min = tf.reduce_min(t, axis=[1, 2], keepdims=True)
    t_max = tf.reduce_max(t, axis=[1, 2], keepdims=True)
    t_norm = (t - t_min) / (t_max - t_min + 1e-6)  # [0,1]

    idx = tf.cast(tf.round(t_norm * (NUM_COLORS - 1)), tf.int32)
    return tf.gather(LUT, idx)  # (B,T,F,3)

@keras.saving.register_keras_serializable()
class GlobalGeMPool2D(keras.layers.Layer):
    def __init__(self, p_init=3.0, **kwargs):
        super().__init__(**kwargs);
        self.p = tf.Variable(p_init, dtype=tf.float32)

    def call(self, t):
        t = tf.maximum(t, 1e-6)
        return tf.pow(tf.reduce_mean(tf.pow(t, self.p), axis=[1, 2]), 1./self.p)

@keras.saving.register_keras_serializable()
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

model = keras.saving.load_model("good-model-2.keras", custom_objects={"mel_to_magma": mel_to_magma, "GlobalGeMPool2D": GlobalGeMPool2D, "AttentiveStatsPool": AttentiveStatsPool})
model.summary()


# 1. Create the Feature Extractor
# We tap into the original model and output the features from "dense_1"
feature_extractor = keras.Model(
    inputs=model.input, 
    outputs=model.get_layer("dense_1").output,
    name="embedding_extractor"
)
feature_extractor.summary()

# 2. Prepare your dataset
audio_files, true_labels = collect_dataset(data_root="data")
print(f"Found {len(audio_files)} audio files across {len(set(true_labels))} classes.")

features_list = []
valid_labels = []

print("Extracting features...")
batch_audio = []
batch_labels = []
feature_progress = keras.utils.Progbar(len(audio_files), unit_name="file")


def flush_batch():
    if not batch_audio:
        return

    max_len = max(len(audio) for audio in batch_audio)
    batch_array = np.stack(
        [
            np.pad(audio, (0, max_len - len(audio)), mode="constant")
            for audio in batch_audio
        ],
        axis=0
    ).astype(np.float32)

    embeddings = feature_extractor.predict(batch_array, batch_size=len(batch_audio), verbose=0)
    features_list.extend(embeddings)
    valid_labels.extend(batch_labels)
    batch_audio.clear()
    batch_labels.clear()


for idx, (file_path, label) in enumerate(zip(audio_files, true_labels), start=1):
    try:
        # Load and pad/trim audio just like your single-file script
        audio = load_audio(file_path)
        if audio is None:
            print(f"Skipping {file_path} due to audio loading failure")
            continue

        batch_audio.append(np.asarray(audio, dtype=np.float32))
        batch_labels.append(label)
        if len(batch_audio) == BATCH_SIZE:
            flush_batch()
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
    finally:
        feature_progress.update(idx)

flush_batch()

# Convert to numpy arrays for scikit-learn
X = np.array(features_list)  # Shape: (N_samples, dense_1_units)
y = np.array(valid_labels)   # Shape: (N_samples,)

# --- DIMENSIONALITY REDUCTION ---

print("Computing PCA and t-SNE...")
reduction_progress = keras.utils.Progbar(2, unit_name="step")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X)
reduction_progress.update(1)

# Note: t-SNE's 'perplexity' usually ranges from 5 to 50. 
# It must be less than your number of samples! Adjust it based on your dataset size.
n_samples = len(X)
perplexity = 30 # min(30, n_samples - 1) 

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
X_tsne = tsne.fit_transform(X)
reduction_progress.update(2)

# --- CLUSTERING (in 2D t-SNE space so we can show clusters/centroids there) ---
n_classes = len(np.unique(y))
print("Running k-means and HDBSCAN in t-SNE space...")
kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_tsne)
centroids_kmeans = kmeans.cluster_centers_  # (k, 2) in t-SNE coordinates

# HDBSCAN in 2D t-SNE space; label -1 = noise
clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_samples // 20), min_samples=1)
labels_hdbscan = clusterer.fit_predict(X_tsne)
n_hdbscan_clusters = len(set(labels_hdbscan) - {-1})

# --- PLOTTING ---

fig = plt.figure(figsize=(16, 14))

# Plot PCA (2D or 3D depending on PCA_N_COMPONENTS)
if PCA_N_COMPONENTS >= 3:
    ax_pca = fig.add_subplot(2, 2, 1, projection="3d")

    unique_labels = np.unique(y)
    palette = sns.color_palette("tab10", len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        idx = y == label
        ax_pca.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            X_pca[idx, 2],
            label=label,
            color=label_to_color[label],
            s=10,
            alpha=0.8,
        )

    ax_pca.set_title(
        f"PCA (3D) of DenseNet Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}"
    )
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    ax_pca.set_zlabel("Principal Component 3")
    ax_pca.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    ax_pca = fig.add_subplot(2, 2, 1)
    sns.scatterplot(
        ax=ax_pca,
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y,
        palette="tab10",
        marker="o",
        s=10,
        alpha=0.8,
    )
    ax_pca.set_title(
        f"PCA of DenseNet Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}"
    )
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    ax_pca.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot t-SNE (always 2D) — true labels
ax_tsne = fig.add_subplot(2, 2, 2)
sns.scatterplot(
    ax=ax_tsne,
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y,
    palette="tab10",
    marker="o",
    s=10,
    alpha=0.8,
)
ax_tsne.set_title("t-SNE of DenseNet Embeddings (true labels)")
ax_tsne.set_xlabel("t-SNE Dimension 1")
ax_tsne.set_ylabel("t-SNE Dimension 2")
ax_tsne.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

# K-Means clusters in t-SNE space: points + centroids
ax_kmeans = fig.add_subplot(2, 2, 3)
palette_k = sns.color_palette("tab10", n_classes)
sns.scatterplot(
    ax=ax_kmeans,
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=labels_kmeans,
    palette=palette_k,
    marker="o",
    s=10,
    alpha=0.8,
    legend="full",
)
# Overlay cluster centroids (the "clusters themselves")
for i in range(n_classes):
    ax_kmeans.scatter(
        centroids_kmeans[i, 0],
        centroids_kmeans[i, 1],
        s=200,
        c=[palette_k[i]],
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
ax_kmeans.set_title(f"K-Means in t-SNE space (k={n_classes}); X = centroids")
ax_kmeans.set_xlabel("t-SNE Dimension 1")
ax_kmeans.set_ylabel("t-SNE Dimension 2")
ax_kmeans.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

# HDBSCAN clusters in t-SNE space: points + convex hulls per cluster
ax_hdbscan = fig.add_subplot(2, 2, 4)
unique_h = np.unique(labels_hdbscan)
has_noise = -1 in unique_h
palette_h = (["#444444"] + list(sns.color_palette("tab10", len(unique_h) - 1))) if has_noise else sns.color_palette("tab10", len(unique_h))
hue_order = sorted(unique_h, key=lambda x: (x == -1, x))
# Draw convex hulls for each cluster (skip noise -1)
for i, cid in enumerate(hue_order):
    if cid == -1:
        continue
    mask = labels_hdbscan == cid
    if np.sum(mask) < 3:
        continue
    pts = X_tsne[mask]
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    color = palette_h[i]
    ax_hdbscan.fill(
        verts[:, 0], verts[:, 1],
        facecolor=color, edgecolor=color, alpha=0.25, linewidth=1.5,
    )
sns.scatterplot(
    ax=ax_hdbscan,
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=labels_hdbscan,
    hue_order=hue_order,
    palette=palette_h,
    marker="o",
    s=10,
    alpha=0.8,
    legend="full",
)
ax_hdbscan.set_title(f"HDBSCAN in t-SNE space ({n_hdbscan_clusters} clusters); hulls = cluster regions")
ax_hdbscan.set_xlabel("t-SNE Dimension 1")
ax_hdbscan.set_ylabel("t-SNE Dimension 2")
ax_hdbscan.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
output_path = "latent_space_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully saved PCA and t-SNE visualizations to: {output_path}")
