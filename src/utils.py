"""Utilidades generales de reproducibilidad y visualización de cuadrículas `x | ŷ | y`."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42) -> None:
    """Fija semillas para reproducibilidad en random, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_tensor(t: torch.Tensor) -> np.ndarray:
    """Convierte un tensor `[-1, 1]` a arreglo NumPy `[0, 1]` para graficado."""
    arr = t.detach().cpu().float().numpy()
    return np.clip((arr + 1.0) * 0.5, 0.0, 1.0)


def save_triplet_grid(
    x: torch.Tensor,
    y_hat: torch.Tensor,
    y_real: torch.Tensor,
    output_path: Path | str,
    chars: list[str] | None = None,
    max_samples: int = 8,
    title: str | None = None,
) -> Path:
    """Guarda una cuadrícula comparativa `Entrada x | Generado ŷ | Real y`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_samples, x.shape[0])
    x_np = denormalize_tensor(x[:n])
    yhat_np = denormalize_tensor(y_hat[:n])
    y_np = denormalize_tensor(y_real[:n])

    fig, axes = plt.subplots(n, 3, figsize=(6.0, 2.0 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Entrada (x)", "Generado (ŷ)", "Objetivo (y)"]
    for row in range(n):
        imgs = [x_np[row, 0], yhat_np[row, 0], y_np[row, 0]]
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0 and chars is not None and row < len(chars):
                ax.set_ylabel(f"'{chars[row]}'", fontsize=10, rotation=0, labelpad=14)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path
