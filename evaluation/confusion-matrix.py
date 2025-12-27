import seaborn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

def plot_confusion_matrix(model, val_data, class_names, get_label):
    true_labels, pred_labels = [], []

    val_files = val_data if isinstance(val_data, list) else list(Path(val_data).glob("**/*.wav"))

    for wav_path in val_files:
        audio = load_and_normalize_audio(wav_path)
        if audio is None:
            continue

        audio_tensor = tf.convert_to_tensor(
            np.asarray(audio).reshape(1, -1),
            dtype=tf.float32
        )
        pred = model.predict(audio_tensor, batch_size=1, verbose=0)
        pred_label = class_names[int(np.argmax(pred))]
        true_label = class_names[get_label(wav_path)]

        if true_label not in class_names:                   # skip unknown labels
            print(f"Unknown label '{true_label}' in '{wav_path}' - skipping.")
            continue

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
    plt.show()
