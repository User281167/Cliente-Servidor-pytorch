from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = list(range(conf_matrix.shape[0]))
    ax.set(
        xticks=classes,
        yticks=classes,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="MNIST Confusion Matrix",
    )

    threshold = conf_matrix.max().item() / 2 if conf_matrix.numel() else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = int(conf_matrix[i, j].item())

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    plt.show()


def plot_grid(
    history: List[Union[List[float], tuple]],
    labels: List[str],
    n_cols: int | None = None,
):
    if not history:
        return

    n_metrics = len(labels)
    n_epochs = len(history)
    columns = list(zip(*history))

    if len(columns) != n_metrics:
        raise ValueError(
            f"El número de columnas en history ({len(columns)}) no coincide con labels ({n_metrics})"
        )

    # auto rows
    n_cols = n_cols or n_metrics
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = np.array(axs).flatten()

    for idx, (col_values, label) in enumerate(zip(columns, labels)):
        ax = axs[idx]
        ax.plot(range(1, n_epochs + 1), col_values)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True)

    # Ocultar subplots sobrantes si el grid tiene más celdas que métricas
    for idx in range(n_metrics, len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
