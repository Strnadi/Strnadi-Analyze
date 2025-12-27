import matplotlib.pyplot as plt
import math

def plot_history(history, metrics):
    """
    Plots training and validation accuracy/loss curves from a Keras History object.
    """
    plt.figure(figsize=(12, 5))

    rows = math.ceil(len(metrics) / 2)
    for index, metric in enumerate(metrics):
        m = history.history.get(metric)
        val_m = history.history.get(f'val_{metric}')

        epochs = range(1, len(m) + 1)
        plt.subplot(rows, 2, index + 1)
        plt.title(f'Training and Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.plot(epochs, m, 'bo-', label=f'Training {metric}')

        if val_m:
            plt.plot(epochs, val_m, 'ro-', label=f'Validation {metric}')

    plt.tight_layout()
    plt.show()
