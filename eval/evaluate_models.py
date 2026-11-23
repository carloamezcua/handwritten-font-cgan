"""Evaluación cuantitativa de modelos Pix2Pix con jueces congelados (Rúbrica D1 y D3).

Evalúa uno o más checkpoints de `UNetGenerator` (más la referencia objetivo `y` como
techo de legibilidad) sobre los splits `val` y `test_unseen` usando los dos jueces
congelados (`ResNet-18` y `EfficientNet-B0`).

Métricas calculadas por modelo, split y juez:
- Accuracy y Macro-F1 (sobre las clases presentes en el split).
- Confianza promedio (probabilidad softmax máxima) y tasa de aprobación (>= umbral OOD).
- Distancia L1 promedio a nivel de píxel entre `ŷ` y `y`.
- FID de dominio (Fréchet Distance sobre embeddings de la penúltima capa del juez).

Artefactos generados:
- `results/eval/metrics_<tag>.json`: resumen estructurado de todas las métricas.
- `results/eval/predictions_<tag>.csv`: predicciones por muestra (usado por auditoría).
- `results/eval/summary_<tag>.md`: tabla Markdown lista para el README/informe.
- `docs/figuras/eval_cm_<tag>_<model>_<split>.png`: matrices de confusión.
- `docs/figuras/eval_errores_<tag>_<model>.png`: galería de errores representativos (D3).
- `docs/figuras/eval_comparativa_<tag>_<split>.png`: cuadrícula comparativa `x | ŷ | y`.

Uso (preentrenado actual):
    python -m eval.evaluate_models

Uso (comparativa final Exp A vs Exp B sobre datos reales):
    python -m eval.evaluate_models \\
        --data-dir data/real_pairs \\
        --models scratch=checkpoints/pix2pix_scratch.pth finetune=checkpoints/pix2pix_finetuned.pth \\
        --tag final
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
import numpy as np
import torch
from scipy import linalg
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    CHARSET,
    DIGITS,
    IDX_TO_CHAR,
    LOWERCASE,
    PUNCTUATION,
    SEED,
    UPPERCASE,
)
from src.data.dataset import GlyphPairDataset
from src.models.judges import build_judge, freeze_model
from src.models.pix2pix import UNetGenerator
from src.utils import denormalize_tensor, seed_everything


# Conjuntos de confusión visual típicos en escritura manuscrita
CONFUSABLE_GROUPS: list[set[str]] = [
    {"I", "l", "1", "i", "!", "/"},
    {"O", "0", "o", "Q", "D"},
    {"c", "C", "("},
    {"s", "S", "5"},
    {"z", "Z", "2"},
    {"v", "V", "u", "U"},
    {"w", "W"},
    {"x", "X"},
    {"p", "P"},
    {"k", "K"},
    {"g", "q", "9"},
    {"b", "d", "6"},
    {".", ",", ":", ";", "-"},
    {"+", "=", "-"},
]


def char_subgroup(ch: str) -> str:
    """Clasifica un carácter del CHARSET en uno de los 4 subgrupos de auditoría."""
    if ch in UPPERCASE:
        return "Mayúsculas (A-Z)"
    if ch in LOWERCASE:
        return "Minúsculas (a-z)"
    if ch in DIGITS:
        return "Dígitos (0-9)"
    if ch in PUNCTUATION:
        return "Signos"
    return "Otro"


def classify_error_type(true_ch: str, pred_ch: str, split: str, conf: float) -> str:
    """Agrupa un error de clasificación en una categoría interpretable para D3."""
    if split == "test_unseen":
        return "Artefacto Zero-Shot (unseen)"
    if true_ch.lower() == pred_ch.lower() and true_ch != pred_ch:
        return "Confusión mayús./minús."
    for grp in CONFUSABLE_GROUPS:
        if true_ch in grp and pred_ch in grp:
            return "Trazos morfológicamente similares"
    if true_ch in PUNCTUATION or pred_ch in PUNCTUATION:
        return "Signo delgado / puntuación"
    if conf < 0.50:
        return "Trazo ambiguo (baja confianza)"
    return "Confusión estructural de forma"


def compute_frechet_distance(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calcula la distancia de Fréchet (FID de dominio) entre dos conjuntos de embeddings.

    Aplica regularización diagonal mínima cuando hay pocas muestras para garantizar
    estabilidad numérica de `scipy.linalg.sqrtm`.
    """
    real_feats = np.asarray(real_feats, dtype=np.float64)
    fake_feats = np.asarray(fake_feats, dtype=np.float64)

    mu1 = np.mean(real_feats, axis=0)
    mu2 = np.mean(fake_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]])

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1e-2):
            offset = np.eye(sigma1.shape[0]) * (eps * 10.0)
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    return max(0.0, fid)


def load_generator(ckpt_path: Path, device: torch.device) -> UNetGenerator:
    """Carga un `UNetGenerator` congelado desde un checkpoint `.pth`."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint del generador: '{ckpt_path}'.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen = UNetGenerator().to(device)
    state = ckpt["generator_state_dict"] if "generator_state_dict" in ckpt else ckpt
    gen.load_state_dict(state)
    return freeze_model(gen)


def load_judge(judge_name: str, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Carga un clasificador juez congelado desde un checkpoint `.pth`."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint del juez: '{ckpt_path}'.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_judge(judge_name, num_classes=len(CHARSET), pretrained=False).to(device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return freeze_model(model)


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    """Convierte argumentos `etiqueta=ruta.pth` en lista de tuplas `(label, Path)`."""
    parsed: list[tuple[str, Path]] = []
    for item in specs:
        if "=" not in item:
            p = Path(item)
            parsed.append((p.stem, p))
        else:
            label, path_str = item.split("=", 1)
            parsed.append((label.strip(), Path(path_str.strip())))
    return parsed


@torch.no_grad()
def extract_split_outputs(
    loader: DataLoader,
    generators: dict[str, UNetGenerator],
    judges: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict:
    """Genera imágenes `ŷ` y extrae logits, probabilidades y embeddings de ambos jueces."""
    all_x: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_chars: list[str] = []

    gen_images: dict[str, list[torch.Tensor]] = {m: [] for m in generators}
    l1_per_sample: dict[str, list[float]] = {m: [] for m in generators}

    # Embeddings y predicciones: [source][judge_name]
    sources = ["reference_y"] + list(generators.keys())
    feats: dict[str, dict[str, list[np.ndarray]]] = {
        s: {j: [] for j in judges} for s in sources
    }
    preds: dict[str, dict[str, list[int]]] = {
        s: {j: [] for j in judges} for s in sources
    }
    confs: dict[str, dict[str, list[float]]] = {
        s: {j: [] for j in judges} for s in sources
    }

    for x, y, class_ids, chars in loader:
        x_dev = x.to(device)
        y_dev = y.to(device)

        all_x.append(x.cpu())
        all_y.append(y.cpu())
        all_labels.extend(int(c) for c in class_ids)
        all_chars.extend(list(chars))

        batch_imgs: dict[str, torch.Tensor] = {"reference_y": y_dev}
        for m_name, gen in generators.items():
            y_hat = gen(x_dev)
            batch_imgs[m_name] = y_hat
            gen_images[m_name].append(y_hat.cpu())
            l1_vals = torch.mean(torch.abs(y_hat - y_dev), dim=(1, 2, 3)).cpu().tolist()
            l1_per_sample[m_name].extend(float(v) for v in l1_vals)

        for s_name, imgs_dev in batch_imgs.items():
            for j_name, judge in judges.items():
                f = judge.extract_features(imgs_dev)
                if j_name == "resnet18":
                    logits = judge.backbone.fc(f)
                else:
                    logits = judge.backbone.classifier(f)
                probs = torch.softmax(logits, dim=1)
                conf_vals, pred_ids = probs.max(dim=1)

                feats[s_name][j_name].append(f.cpu().numpy())
                preds[s_name][j_name].extend(int(p) for p in pred_ids.cpu().tolist())
                confs[s_name][j_name].extend(float(c) for c in conf_vals.cpu().tolist())

    result = {
        "x": torch.cat(all_x, dim=0),
        "y": torch.cat(all_y, dim=0),
        "y_hat": {m: torch.cat(gen_images[m], dim=0) for m in generators},
        "labels": np.array(all_labels, dtype=np.int64),
        "chars": all_chars,
        "l1": {m: np.array(l1_per_sample[m], dtype=np.float64) for m in generators},
        "feats": {
            s: {j: np.concatenate(feats[s][j], axis=0) for j in judges}
            for s in sources
        },
        "preds": {
            s: {j: np.array(preds[s][j], dtype=np.int64) for j in judges}
            for s in sources
        },
        "confs": {
            s: {j: np.array(confs[s][j], dtype=np.float64) for j in judges}
            for s in sources
        },
    }
    return result


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    max_classes_full: int = 20,
) -> Path:
    """Guarda matriz de confusión legible (completa en test_unseen; top confusiones si >20 clases)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    present_true = sorted(set(int(v) for v in y_true))

    if len(present_true) <= max_classes_full:
        # Incluir etiquetas verdaderas + cualquier predicción frecuente fuera de ellas
        extra_preds = sorted(
            set(int(v) for v in y_pred) - set(present_true),
            key=lambda c: int(np.sum(y_pred == c)),
            reverse=True,
        )[:8]
        col_labels = present_true + extra_preds
        cm = np.zeros((len(present_true), len(col_labels)), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if int(p) in col_labels:
                cm[present_true.index(int(t)), col_labels.index(int(p))] += 1

        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm / row_sums

        fig_w = max(6.5, 0.55 * len(col_labels) + 1.8)
        fig_h = max(5.5, 0.50 * len(present_true) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels([IDX_TO_CHAR[c] for c in col_labels], fontsize=9)
        ax.set_yticks(range(len(present_true)))
        ax.set_yticklabels([IDX_TO_CHAR[c] for c in present_true], fontsize=9)
        ax.set_xlabel("Predicción del juez", fontsize=10)
        ax.set_ylabel("Carácter real", fontsize=10)
        ax.set_title(title, fontsize=11)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0:
                    color = "white" if cm_norm[i, j] > 0.55 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Proporción por fila")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Para splits con muchas clases (p. ej. 64 en val): panel doble
    # Izquierda: matriz completa 64x64; Derecha: top-15 pares de confusión más frecuentes
    cm = confusion_matrix(y_true, y_pred, labels=present_true)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if int(t) != int(p):
            pair_counts[(IDX_TO_CHAR[int(t)], IDX_TO_CHAR[int(p)])] += 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    tick_step = max(1, len(present_true) // 16)
    tick_idx = list(range(0, len(present_true), tick_step))
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([IDX_TO_CHAR[present_true[i]] for i in tick_idx], fontsize=8)
    ax1.set_yticks(tick_idx)
    ax1.set_yticklabels([IDX_TO_CHAR[present_true[i]] for i in tick_idx], fontsize=8)
    ax1.set_xlabel("Predicción del juez", fontsize=10)
    ax1.set_ylabel("Carácter real", fontsize=10)
    ax1.set_title(f"Matriz normalizada ({len(present_true)} clases)", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)

    if top_pairs:
        labels_txt = [f"'{t}' -> '{p}'" for (t, p), _ in reversed(top_pairs)]
        counts = [cnt for _, cnt in reversed(top_pairs)]
        ax2.barh(range(len(labels_txt)), counts, color="#C0392B", alpha=0.85)
        ax2.set_yticks(range(len(labels_txt)))
        ax2.set_yticklabels(labels_txt, fontsize=9)
        ax2.set_xlabel("Número de errores", fontsize=10)
        ax2.set_title("Top confusiones (Real -> Predicho)", fontsize=10)
        ax2.grid(axis="x", alpha=0.3)
        for idx, c in enumerate(counts):
            ax2.text(c + 0.15, idx, str(c), va="center", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Sin errores de clasificación", ha="center", va="center", fontsize=11)
        ax2.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_gallery(
    error_records: list[dict],
    title: str,
    output_path: Path,
    max_errors: int = 8,
) -> tuple[Path, list[dict]]:
    """Selecciona hasta `max_errors` errores diversos por categoría y guarda su galería visual."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not error_records:
        return output_path, []

    # Selección estratificada por categoría de error (evitando repetir el mismo carácter real)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in error_records:
        by_cat[rec["error_type"]].append(rec)

    for cat in by_cat:
        # Priorizar errores más ilustrativos (confianza media/alta o L1 alto)
        by_cat[cat].sort(key=lambda r: (r["conf"], r["l1"]), reverse=True)

    selected: list[dict] = []
    used_chars: set[tuple[str, str]] = set()
    cats = sorted(by_cat.keys())

    # Ronda 1: uno por categoría y luego rellenar hasta max_errors con caracteres distintos
    idx_in_cat = {c: 0 for c in cats}
    while len(selected) < max_errors:
        added = False
        for c in cats:
            lst = by_cat[c]
            while idx_in_cat[c] < len(lst):
                cand = lst[idx_in_cat[c]]
                idx_in_cat[c] += 1
                key = (cand["split"], cand["true_char"])
                if key not in used_chars:
                    selected.append(cand)
                    used_chars.add(key)
                    added = True
                    break
            if len(selected) >= max_errors:
                break
        if not added:
            break

    # Respaldo solo si hubo menos de max_errors caracteres distintos con error
    if len(selected) < min(max_errors, len(error_records)):
        selected_ids = {id(r) for r in selected}
        for rec in error_records:
            if id(rec) not in selected_ids:
                selected.append(rec)
                selected_ids.add(id(rec))
            if len(selected) >= max_errors:
                break

    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(8.2, 2.05 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Entrada (x)", "Generado (ŷ)", "Objetivo (y)"]
    for row, rec in enumerate(selected):
        imgs = [rec["x_np"], rec["yhat_np"], rec["y_np"]]
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(
            f"[{rec['split']}]\nReal: '{rec['true_char']}'",
            fontsize=8.5,
            rotation=0,
            labelpad=42,
            va="center",
        )
        axes[row, 1].set_xlabel(
            f"Pred: '{rec['pred_char']}' (p={rec['conf']:.2f}) | {rec['error_type']}",
            fontsize=8,
            color="#B03A2E",
        )

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_selected = [
        {
            "split": r["split"],
            "sample_idx": r["sample_idx"],
            "true_char": r["true_char"],
            "pred_char": r["pred_char"],
            "conf": round(r["conf"], 4),
            "l1": round(r["l1"], 4),
            "subgroup": r["subgroup"],
            "error_type": r["error_type"],
        }
        for r in selected
    ]
    return output_path, summary_selected


def plot_model_comparison_grid(
    split_data: dict,
    model_names: list[str],
    split_name: str,
    output_path: Path,
    max_samples: int = 8,
) -> Path:
    """Guarda cuadrícula comparativa `x | ŷ_model1 | ... | y` eligiendo caracteres variados."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chars = split_data["chars"]

    # Elegir índices con caracteres distintos distribuidos a lo largo del split
    seen: set[str] = set()
    chosen: list[int] = []
    step = max(1, len(chars) // max(max_samples * 3, 1))
    for i in range(0, len(chars), step):
        if chars[i] not in seen:
            seen.add(chars[i])
            chosen.append(i)
        if len(chosen) >= max_samples:
            break
    if len(chosen) < min(max_samples, len(chars)):
        for i in range(len(chars)):
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= max_samples:
                break

    cols = 2 + len(model_names)
    col_titles = ["Entrada (x)"] + [f"Generado ({m})" for m in model_names] + ["Objetivo (y)"]

    fig, axes = plt.subplots(len(chosen), cols, figsize=(2.1 * cols, 2.0 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    x_np = denormalize_tensor(split_data["x"])
    y_np = denormalize_tensor(split_data["y"])
    yhat_np = {m: denormalize_tensor(split_data["y_hat"][m]) for m in model_names}

    for r_idx, s_idx in enumerate(chosen):
        row_imgs = [x_np[s_idx, 0]] + [yhat_np[m][s_idx, 0] for m in model_names] + [y_np[s_idx, 0]]
        for c_idx, img in enumerate(row_imgs):
            ax = axes[r_idx, c_idx]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.5, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"'{chars[s_idx]}'", fontsize=10, rotation=0, labelpad=14)
            elif 1 <= c_idx <= len(model_names):
                m_name = model_names[c_idx - 1]
                pred_c = IDX_TO_CHAR[int(split_data["preds"][m_name]["resnet18"][s_idx])]
                conf_v = float(split_data["confs"][m_name]["resnet18"][s_idx])
                color = "#1E8449" if pred_c == chars[s_idx] else "#B03A2E"
                ax.set_xlabel(f"R18:'{pred_c}' ({conf_v:.2f})", fontsize=7.5, color=color)

    fig.suptitle(f"Comparativa visual — split '{split_name}'", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_markdown_summary(report: dict) -> str:
    """Construye una tabla Markdown legible con todas las métricas cuantitativas."""
    lines: list[str] = [
        f"# Resumen de Evaluación Cuantitativa (`{report['tag']}`)",
        "",
        f"- **Dataset evaluado:** `{report['data_dir']}`",
        f"- **Umbral de confianza OOD:** `{report['ood_threshold']:.2f}`",
        f"- **Dispositivo:** `{report['device']}`",
        "",
    ]

    for split_name, split_info in report["splits"].items():
        lines.append(
            f"## Split `{split_name}` "
            f"({split_info['num_samples']} muestras, {split_info['num_classes_present']} clases)"
        )
        lines.append("")
        lines.append(
            "| Modelo | Juez | Accuracy | Macro-F1 | Conf. media | Pass ≥0.70 | L1 media | FID dominio |"
        )
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for src_name, src_metrics in split_info["sources"].items():
            l1_str = f"{src_metrics['mean_l1']:.4f}" if src_metrics["mean_l1"] is not None else "—"
            for j_name, jm in src_metrics["judges"].items():
                fid_str = f"{jm['domain_fid']:.2f}" if jm["domain_fid"] is not None else "0.00 (ref)"
                lines.append(
                    f"| `{src_name}` | `{j_name}` | {jm['accuracy']*100:.2f}% | "
                    f"{jm['macro_f1']*100:.2f}% | {jm['mean_conf']:.4f} | "
                    f"{jm['pass_rate_at_threshold']*100:.2f}% | {l1_str} | {fid_str} |"
                )
        lines.append("")

    if report.get("error_analysis"):
        lines.append("## Resumen de Análisis de Errores (Juez ResNet-18)")
        lines.append("")
        for m_name, err_info in report["error_analysis"].items():
            lines.append(
                f"### Modelo `{m_name}` (total errores: {err_info['total_errors']})"
            )
            lines.append("")
            lines.append("| Categoría de error | Conteo |")
            lines.append("| :--- | :---: |")
            for cat, cnt in sorted(
                err_info["counts_by_category"].items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"| {cat} | {cnt} |")
            lines.append("")
            if err_info["representative_errors"]:
                lines.append("**Errores representativos seleccionados:**")
                lines.append("")
                lines.append("| Split | Real | Predicho | Confianza | L1 | Categoría |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
                for r in err_info["representative_errors"]:
                    lines.append(
                        f"| `{r['split']}` | `{r['true_char']}` | `{r['pred_char']}` | "
                        f"{r['conf']:.3f} | {r['l1']:.4f} | {r['error_type']} |"
                    )
                lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa (Accuracy, Macro-F1, CM, FID, errores) de Pix2Pix."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic_pairs"),
        help="Directorio del dataset con metadata.csv (synthetic_pairs o real_pairs).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test_unseen"],
        help="Splits a evaluar.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pretrained=checkpoints/pix2pix_pretrained.pth"],
        help="Lista de modelos en formato etiqueta=ruta.pth.",
    )
    parser.add_argument(
        "--judges-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directorio con judge_resnet18.pth y judge_efficientnet_b0.pth.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/eval"),
        help="Carpeta donde guardar métricas JSON, CSV y tabla Markdown.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figuras"),
        help="Carpeta donde guardar matrices de confusión y galerías.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="pretrain",
        help="Sufijo identificador del experimento (ej. pretrain, final).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ood-threshold", type=float, default=0.70)
    parser.add_argument("--max-errors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_specs = parse_model_specs(args.models)
    model_names = [m for m, _ in model_specs]

    print(f"[INFO] Dispositivo: {device} | Dataset: {args.data_dir} | Tag: {args.tag}")
    generators = {m: load_generator(p, device) for m, p in model_specs}

    judges: dict[str, torch.nn.Module] = {
        "resnet18": load_judge("resnet18", args.judges_dir / "judge_resnet18.pth", device),
        "efficientnet_b0": load_judge(
            "efficientnet_b0", args.judges_dir / "judge_efficientnet_b0.pth", device
        ),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "tag": args.tag,
        "data_dir": str(args.data_dir),
        "device": str(device),
        "ood_threshold": args.ood_threshold,
        "models": {m: str(p) for m, p in model_specs},
        "splits": {},
        "error_analysis": {},
        "figures": [],
    }

    csv_rows: list[dict] = []
    errors_by_model: dict[str, list[dict]] = {m: [] for m in model_names}

    for split_name in args.splits:
        ds = GlyphPairDataset(root_dir=args.data_dir, split=split_name, augment=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"\n=== Evaluando split '{split_name}' ({len(ds)} muestras) ===")

        out = extract_split_outputs(loader, generators, judges, device)
        y_true = out["labels"]
        present_labels = sorted(set(int(v) for v in y_true))

        split_report: dict = {
            "num_samples": len(ds),
            "num_classes_present": len(present_labels),
            "classes_present": [IDX_TO_CHAR[c] for c in present_labels],
            "sources": {},
        }

        sources = ["reference_y"] + model_names
        for s_name in sources:
            mean_l1 = float(out["l1"][s_name].mean()) if s_name in out["l1"] else None
            src_dict: dict = {
                "mean_l1": round(mean_l1, 5) if mean_l1 is not None else None,
                "judges": {},
            }

            for j_name in judges:
                y_pred = out["preds"][s_name][j_name]
                confs = out["confs"][s_name][j_name]
                acc = float(accuracy_score(y_true, y_pred))
                macro_f1 = float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=present_labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                weighted_f1 = float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=present_labels,
                        average="weighted",
                        zero_division=0,
                    )
                )
                pass_rate = float(np.mean(confs >= args.ood_threshold))

                if s_name == "reference_y":
                    domain_fid = None
                else:
                    domain_fid = compute_frechet_distance(
                        out["feats"]["reference_y"][j_name],
                        out["feats"][s_name][j_name],
                    )

                src_dict["judges"][j_name] = {
                    "accuracy": round(acc, 5),
                    "macro_f1": round(macro_f1, 5),
                    "weighted_f1": round(weighted_f1, 5),
                    "mean_conf": round(float(confs.mean()), 5),
                    "std_conf": round(float(confs.std()), 5),
                    "pass_rate_at_threshold": round(pass_rate, 5),
                    "domain_fid": round(domain_fid, 4) if domain_fid is not None else None,
                }

                fid_txt = f"{domain_fid:.2f}" if domain_fid is not None else "0.00 (ref)"
                print(
                    f"  [{s_name:12s} | {j_name:15s}] "
                    f"Acc={acc*100:6.2f}% | Macro-F1={macro_f1*100:6.2f}% | "
                    f"Conf={confs.mean():.3f} | Pass>={args.ood_threshold:.2f}={pass_rate*100:5.1f}% | "
                    f"FID={fid_txt}"
                )

                # Guardar filas para CSV de auditoría por subgrupos
                for i in range(len(y_true)):
                    true_ch = out["chars"][i]
                    pred_ch = IDX_TO_CHAR[int(y_pred[i])]
                    csv_rows.append(
                        {
                            "split": split_name,
                            "sample_idx": i,
                            "source": s_name,
                            "judge": j_name,
                            "true_id": int(y_true[i]),
                            "true_char": true_ch,
                            "subgroup": char_subgroup(true_ch),
                            "pred_id": int(y_pred[i]),
                            "pred_char": pred_ch,
                            "correct": int(y_true[i] == y_pred[i]),
                            "confidence": round(float(confs[i]), 5),
                            "pass_ood": int(confs[i] >= args.ood_threshold),
                            "l1": round(float(out["l1"][s_name][i]), 5)
                            if s_name in out["l1"]
                            else "",
                        }
                    )

            split_report["sources"][s_name] = src_dict

        # Matrices de confusión y recolección de errores (con juez principal ResNet-18)
        x_np = denormalize_tensor(out["x"])
        y_np = denormalize_tensor(out["y"])
        for m_name in model_names:
            cm_path = args.figures_dir / f"eval_cm_{args.tag}_{m_name}_{split_name}.png"
            plot_confusion_matrix_figure(
                y_true=y_true,
                y_pred=out["preds"][m_name]["resnet18"],
                title=f"Matriz de Confusión — {m_name} ({split_name}, Juez ResNet-18)",
                output_path=cm_path,
            )
            report["figures"].append(str(cm_path))

            yhat_np = denormalize_tensor(out["y_hat"][m_name])
            preds_r18 = out["preds"][m_name]["resnet18"]
            confs_r18 = out["confs"][m_name]["resnet18"]
            for i in range(len(y_true)):
                if int(preds_r18[i]) != int(y_true[i]):
                    true_ch = out["chars"][i]
                    pred_ch = IDX_TO_CHAR[int(preds_r18[i])]
                    conf_val = float(confs_r18[i])
                    errors_by_model[m_name].append(
                        {
                            "split": split_name,
                            "sample_idx": i,
                            "true_char": true_ch,
                            "pred_char": pred_ch,
                            "conf": conf_val,
                            "l1": float(out["l1"][m_name][i]),
                            "subgroup": char_subgroup(true_ch),
                            "error_type": classify_error_type(
                                true_ch, pred_ch, split_name, conf_val
                            ),
                            "x_np": x_np[i, 0],
                            "yhat_np": yhat_np[i, 0],
                            "y_np": y_np[i, 0],
                        }
                    )

        comp_path = args.figures_dir / f"eval_comparativa_{args.tag}_{split_name}.png"
        plot_model_comparison_grid(out, model_names, split_name, comp_path, max_samples=8)
        report["figures"].append(str(comp_path))

        report["splits"][split_name] = split_report

    # Galerías de errores representativos (Rúbrica D3)
    for m_name, err_list in errors_by_model.items():
        cat_counts: dict[str, int] = defaultdict(int)
        for rec in err_list:
            cat_counts[rec["error_type"]] += 1

        err_fig_path = args.figures_dir / f"eval_errores_{args.tag}_{m_name}.png"
        _, selected_summary = plot_error_gallery(
            err_list,
            title=f"Análisis de Errores Representativos — {m_name} ({args.tag})",
            output_path=err_fig_path,
            max_errors=args.max_errors,
        )
        if selected_summary:
            report["figures"].append(str(err_fig_path))
        report["error_analysis"][m_name] = {
            "total_errors": len(err_list),
            "counts_by_category": dict(cat_counts),
            "representative_errors": selected_summary,
            "figure_path": str(err_fig_path) if selected_summary else None,
        }

    # Guardar salidas estructuradas
    json_path = args.results_dir / f"metrics_{args.tag}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.results_dir / f"predictions_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path = args.results_dir / f"summary_{args.tag}.md"
    md_path.write_text(build_markdown_summary(report), encoding="utf-8")

    print(f"\n[OK] Métricas JSON:   {json_path}")
    print(f"[OK] Predicciones:    {csv_path} ({len(csv_rows)} filas)")
    print(f"[OK] Resumen MD:      {md_path}")
    print(f"[OK] Figuras ({len(report['figures'])}):     {', '.join(report['figures'])}")


if __name__ == "__main__":
    main()
