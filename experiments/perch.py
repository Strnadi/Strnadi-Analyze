from perch_hoplite.zoo import model_configs
import numpy as np
import os
import librosa
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Strnadi-Analyze"))
from evaluation.file_names import collect_dataset

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import hdbscan

from tqdm.auto import tqdm

WORKSPACE = "/workspace"
DATASET_DIR = os.path.join(WORKSPACE, "dataset")
VALIDATION_SPLIT = 0.3
SEED = 1773398195

def load_audio(file_path, target_sr=32000, target_duration=5):
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

# Input: 5 seconds of silence as mono 32 kHz waveform samples.
waveform = np.zeros(5 * 32000, dtype=np.float32)

# Automatically downloads the model from Kaggle.
model = model_configs.load_model_by_name('perch_v2')

outputs = model.embed(waveform)

audio_files, true_labels = collect_dataset(dataset_dir=DATASET_DIR, validation_split=VALIDATION_SPLIT, train=True, val=True, test=True, seed=SEED)

outputs, outputs_labels = [], []

for file, label in tqdm(zip(audio_files, true_labels)):
    waveform = load_audio(file)
    output = model.embed(waveform)
    outputs.append(output.embeddings[0][0])
    outputs_labels.append(label)

outputs = np.array(outputs)
outputs_labels = np.array(outputs_labels)

clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(outputs)

X_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(outputs)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=outputs_labels,          # color by cluster assignment
    palette="tab10",
    s=15,
    alpha=0.8,
)
plt.title("t-SNE of Perch Embeddings (HDBSCAN clusters)")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE, "perch_tsne_clusters.png"), dpi=300, bbox_inches="tight")
plt.close()

# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x=X_2d[:, 0],
#     y=X_2d[:, 1],
#     hue=outputs_labels,
#     palette="tab10",
#     s=15,
#     alpha=0.8,
# )
# plt.title("TSNE of Perch Embeddings (true labels)")
# plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.savefig(os.path.join(WORKSPACE, "perch_tsne.png"), dpi=300, bbox_inches="tight")
# plt.close()
