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
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "layers"))

from file_names import collect_dataset
from layers.spec_augument import SpecAugment

SAMPLE_RATE = 48000
BATCH_SIZE = 8
# Set to 3 to enable 3D PCA visualization
PCA_N_COMPONENTS = 2
WORKSPACE = "/workspace"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
VALIDATION_SPLIT = 0.3
SEED = 1773398195

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.encoder,
            "temperature": self.temperature,
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Intercept the config dictionary and explicitly deserialize the encoder
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.saving.deserialize_keras_object(
                config["encoder"], custom_objects=custom_objects
            )
        return cls(**config)

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

model = keras.saving.load_model(os.path.join(WORKSPACE, "checkpoints", "1773398197", "8-0.2523.keras"), custom_objects={"ContrastiveLearner": ContrastiveLearner, "mel_to_magma": mel_to_magma, "GlobalGeMPool2D": GlobalGeMPool2D, "AttentiveStatsPool": AttentiveStatsPool, "SpecAugment": SpecAugment, "AttentiveStatsPool1D": AttentiveStatsPool1D})
model.summary()


# # 1. Create the Feature Extractor from the encoder inside the contrastive learner
# encoder = model.encoder
# encoder.trainable = False

# feature_extractor = keras.Model(
#     inputs=encoder.input,
#     outputs=encoder.output,
#     name="embedding_extractor",
# )
# feature_extractor.summary()

# 2. Prepare your dataset
audio_files, true_labels = collect_dataset(dataset_dir=DATASET_DIR, validation_split=VALIDATION_SPLIT, train=False, val=False, test=True, seed=SEED)
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

    embeddings = model.encoder.predict(batch_array, batch_size=len(batch_audio), verbose=0)
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

clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(X)
print(f"HDBSCAN clusters: {clusters}")
print(f"Number of clusters: {len(set(clusters))}")
print(f"Number of noise points: {sum(clusters == -1)}")
print(f"Number of points in each cluster: {np.bincount(clusters + 1)}")
print(f"Number of points in each cluster: {np.bincount(clusters + 1)}")

kmeans_clusters = KMeans(n_clusters=7, random_state=42).fit_predict(X)
print(f"KMeans clusters: {kmeans_clusters}")
print(f"Number of clusters: {len(set(kmeans_clusters))}")
print(f"Number of noise points: {sum(kmeans_clusters == -1)}")
print(f"Number of points in each cluster: {np.bincount(kmeans_clusters + 1)}")
print(f"Number of points in each cluster: {np.bincount(kmeans_clusters + 1)}")

X_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

plt.figure(figsize=(10, 8))
unique_labels = np.unique(clusters)
has_noise = -1 in unique_labels
palette = (
    ["#444444"]
    + list(sns.color_palette("tab10", len(unique_labels) - 1))
    if has_noise
    else sns.color_palette("tab10", len(unique_labels))
)
hue_order = sorted(unique_labels, key=lambda x: (x == -1, x))

sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=clusters,
    hue_order=hue_order,
    palette=palette,
    s=15,
    alpha=0.8,
    legend="full",
)
plt.title(f"HDBSCAN clusters of SimCLR embeddings (test set)\nClusters: {len(set(clusters))} (noise=-1)")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

out_path = os.path.join(WORKSPACE, "simclr_hdbscan_clusters_test.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved HDBSCAN cluster visualization to: {out_path}")

# tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform()

# # --- DIMENSIONALITY REDUCTION ---

# print("Computing PCA and t-SNE...")
# reduction_progress = keras.utils.Progbar(2, unit_name="step")
# pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
# X_pca = pca.fit_transform(X)
# reduction_progress.update(1)

# # Note: t-SNE's 'perplexity' usually ranges from 5 to 50. 
# # It must be less than your number of samples! Adjust it based on your dataset size.
# n_samples = len(X)
# perplexity = 30 # min(30, n_samples - 1) 

# tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
# X_tsne = tsne.fit_transform(X)
# reduction_progress.update(2)

# # --- CLUSTERING (in 2D t-SNE space so we can show clusters/centroids there) ---
# n_classes = len(np.unique(y))
# print("Running k-means and HDBSCAN in t-SNE space...")
# kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
# labels_kmeans = kmeans.fit_predict(X_tsne)
# centroids_kmeans = kmeans.cluster_centers_  # (k, 2) in t-SNE coordinates

# # HDBSCAN in 2D t-SNE space; label -1 = noise
# clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_samples // 20), min_samples=1)
# labels_hdbscan = clusterer.fit_predict(X_tsne)
# n_hdbscan_clusters = len(set(labels_hdbscan) - {-1})

# # --- PLOTTING ---

# fig = plt.figure(figsize=(16, 14))

# # Plot PCA (2D or 3D depending on PCA_N_COMPONENTS)
# if PCA_N_COMPONENTS >= 3:
#     ax_pca = fig.add_subplot(2, 2, 1, projection="3d")

#     unique_labels = np.unique(y)
#     palette = sns.color_palette("tab10", len(unique_labels))
#     label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

#     for label in unique_labels:
#         idx = y == label
#         ax_pca.scatter(
#             X_pca[idx, 0],
#             X_pca[idx, 1],
#             X_pca[idx, 2],
#             label=label,
#             color=label_to_color[label],
#             s=10,
#             alpha=0.8,
#         )

#     ax_pca.set_title(
#         f"PCA (3D) of DenseNet Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}"
#     )
#     ax_pca.set_xlabel("Principal Component 1")
#     ax_pca.set_ylabel("Principal Component 2")
#     ax_pca.set_zlabel("Principal Component 3")
#     ax_pca.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
# else:
#     ax_pca = fig.add_subplot(2, 2, 1)
#     sns.scatterplot(
#         ax=ax_pca,
#         x=X_pca[:, 0],
#         y=X_pca[:, 1],
#         hue=y,
#         palette="tab10",
#         marker="o",
#         s=10,
#         alpha=0.8,
#     )
#     ax_pca.set_title(
#         f"PCA of DenseNet Embeddings\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}"
#     )
#     ax_pca.set_xlabel("Principal Component 1")
#     ax_pca.set_ylabel("Principal Component 2")
#     ax_pca.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

# # Plot t-SNE (always 2D) — true labels
# ax_tsne = fig.add_subplot(2, 2, 2)
# sns.scatterplot(
#     ax=ax_tsne,
#     x=X_tsne[:, 0],
#     y=X_tsne[:, 1],
#     hue=y,
#     palette="tab10",
#     marker="o",
#     s=10,
#     alpha=0.8,
# )
# ax_tsne.set_title("t-SNE of DenseNet Embeddings (true labels)")
# ax_tsne.set_xlabel("t-SNE Dimension 1")
# ax_tsne.set_ylabel("t-SNE Dimension 2")
# ax_tsne.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

# # K-Means clusters in t-SNE space: points + centroids
# ax_kmeans = fig.add_subplot(2, 2, 3)
# palette_k = sns.color_palette("tab10", n_classes)
# sns.scatterplot(
#     ax=ax_kmeans,
#     x=X_tsne[:, 0],
#     y=X_tsne[:, 1],
#     hue=labels_kmeans,
#     palette=palette_k,
#     marker="o",
#     s=10,
#     alpha=0.8,
#     legend="full",
# )
# # Overlay cluster centroids (the "clusters themselves")
# for i in range(n_classes):
#     ax_kmeans.scatter(
#         centroids_kmeans[i, 0],
#         centroids_kmeans[i, 1],
#         s=200,
#         c=[palette_k[i]],
#         marker="X",
#         edgecolors="black",
#         linewidths=1.5,
#         zorder=5,
#     )
# ax_kmeans.set_title(f"K-Means in t-SNE space (k={n_classes}); X = centroids")
# ax_kmeans.set_xlabel("t-SNE Dimension 1")
# ax_kmeans.set_ylabel("t-SNE Dimension 2")
# ax_kmeans.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

# # HDBSCAN clusters in t-SNE space: points + convex hulls per cluster
# ax_hdbscan = fig.add_subplot(2, 2, 4)
# unique_h = np.unique(labels_hdbscan)
# has_noise = -1 in unique_h
# palette_h = (["#444444"] + list(sns.color_palette("tab10", len(unique_h) - 1))) if has_noise else sns.color_palette("tab10", len(unique_h))
# hue_order = sorted(unique_h, key=lambda x: (x == -1, x))
# # Draw convex hulls for each cluster (skip noise -1)
# for i, cid in enumerate(hue_order):
#     if cid == -1:
#         continue
#     mask = labels_hdbscan == cid
#     if np.sum(mask) < 3:
#         continue
#     pts = X_tsne[mask]
#     hull = ConvexHull(pts)
#     verts = pts[hull.vertices]
#     color = palette_h[i]
#     ax_hdbscan.fill(
#         verts[:, 0], verts[:, 1],
#         facecolor=color, edgecolor=color, alpha=0.25, linewidth=1.5,
#     )
# sns.scatterplot(
#     ax=ax_hdbscan,
#     x=X_tsne[:, 0],
#     y=X_tsne[:, 1],
#     hue=labels_hdbscan,
#     hue_order=hue_order,
#     palette=palette_h,
#     marker="o",
#     s=10,
#     alpha=0.8,
#     legend="full",
# )
# ax_hdbscan.set_title(f"HDBSCAN in t-SNE space ({n_hdbscan_clusters} clusters); hulls = cluster regions")
# ax_hdbscan.set_xlabel("t-SNE Dimension 1")
# ax_hdbscan.set_ylabel("t-SNE Dimension 2")
# ax_hdbscan.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

# plt.tight_layout()
# output_path = "latent_space_visualization.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Successfully saved PCA and t-SNE visualizations to: {output_path}")
