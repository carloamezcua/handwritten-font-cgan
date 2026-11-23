"""Análisis de interpretabilidad con Grad-CAM sobre los jueces congelados (Avance A).

Calcula mapas de activación Grad-CAM sobre `ResNet-18` (`backbone.layer4[-1]`) y
`EfficientNet-B0` (`backbone.features[-1]`) evaluando tanto glifos de referencia `y`
como glifos generados `ŷ = G(x)` por el generador Pix2Pix.

Genera 4 figuras en `docs/figuras/` y un resumen cuantitativo en `results/eval/`:
1. `gradcam_<tag>_val_correctos.png`: 6 glifos bien clasificados de los 4 subgrupos.
2. `gradcam_<tag>_unseen_zeroshot.png`: 6 glifos Zero-Shot de `test_unseen`.
3. `gradcam_<tag>_errores.png`: 6 errores o casos de baja confianza explicando el fallo.
4. `gradcam_<tag>_resnet_vs_effnet.png`: comparativa cruzada ResNet-18 vs EfficientNet-B0.

Uso (modelo preentrenado actual):
    python -m eval.gradcam_analysis

Uso (modelo final fine-tuned sobre datos reales):
    python -m eval.gradcam_analysis \\
        --data-dir data/real_pairs \\
        --generator-ckpt checkpoints/pix2pix_finetuned.pth \\
        --tag final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.evaluate_models import char_subgroup, load_generator, load_judge
from src.config import CHARSET, DIGITS, IDX_TO_CHAR, LOWERCASE, PUNCTUATION, SEED, UPPERCASE
from src.data.dataset import GlyphPairDataset
from src.utils import denormalize_tensor, seed_everything


class GradCAM:
    """Implementación ligera y exacta de Grad-CAM con hooks en PyTorch."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Calcula el mapa Grad-CAM normalizado en [0, 1] para una imagen `[1, 1, H, W]`.

        Devuelve `(cam_128x128, pred_class_idx, pred_confidence)`.
        """
        assert x.ndim == 4 and x.shape[0] == 1, "Se esperaba tensor [1, 1, H, W]."
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Habilitamos gradientes en la entrada para que el grafo fluya por el modelo congelado
        x_req = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x_req)
            probs = torch.softmax(logits, dim=1)
            pred_conf, pred_idx_t = probs.max(dim=1)
            pred_idx = int(pred_idx_t.item())
            conf_val = float(pred_conf.item())

            cls_idx = pred_idx if target_class is None else int(target_class)
            score = logits[0, cls_idx]
            score.backward()

        assert self._activations is not None and self._gradients is not None
        # Pesos globales por canal (Global Average Pooling de los gradientes)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self._activations, dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam[0, 0].cpu().numpy()
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np.astype(np.float32), pred_idx, conf_val


def get_target_layer(model: torch.nn.Module, judge_name: str) -> torch.nn.Module:
    """Devuelve la última capa convolucional espacial antes del pooling global."""
    if judge_name == "resnet18":
        return model.backbone.layer4[-1]
    if judge_name == "efficientnet_b0":
        return model.backbone.features[-1]
    raise ValueError(f"Juez no soportado para Grad-CAM: {judge_name}")


def overlay_cam_on_glyph(
    glyph_01: np.ndarray,
    cam_01: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpone mapa de calor Grad-CAM (colormap TURBO/JET) preservando nítido el trazo negro."""
    gray_u8 = np.clip(glyph_01 * 255.0, 0, 255).astype(np.uint8)
    base_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32) / 255.0

    cam_u8 = np.clip(cam_01 * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Mezcla controlada: atenúa calor en zonas muy frías para mantener fondo limpio
    weight = alpha * (0.25 + 0.75 * cam_01[..., None])
    blended = (1.0 - weight) * base_rgb + weight * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def cam_cosine_similarity(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    """Similitud coseno entre dos mapas de atención Grad-CAM [0, 1]."""
    va = cam_a.reshape(-1).astype(np.float64)
    vb = cam_b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denom)


def collect_split_samples(
    data_dir: Path,
    split: str,
    gen: torch.nn.Module,
    judge: torch.nn.Module,
    device: torch.device,
) -> list[dict]:
    """Ejecuta generador y juez sobre un split completo para seleccionar muestras representativas."""
    ds = GlyphPairDataset(root_dir=data_dir, split=split, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    records: list[dict] = []
    idx_global = 0
    with torch.no_grad():
        for x, y, class_ids, chars in loader:
            x_dev = x.to(device)
            y_dev = y.to(device)
            y_hat = gen(x_dev)

            probs_y = torch.softmax(judge(y_dev), dim=1)
            conf_y, pred_y = probs_y.max(dim=1)

            probs_yh = torch.softmax(judge(y_hat), dim=1)
            conf_yh, pred_yh = probs_yh.max(dim=1)

            for i in range(x.shape[0]):
                records.append(
                    {
                        "split": split,
                        "idx": idx_global,
                        "x": x[i : i + 1].cpu(),
                        "y": y[i : i + 1].cpu(),
                        "y_hat": y_hat[i : i + 1].cpu(),
                        "true_id": int(class_ids[i]),
                        "true_char": chars[i],
                        "subgroup": char_subgroup(chars[i]),
                        "pred_y_id": int(pred_y[i].item()),
                        "pred_y_char": IDX_TO_CHAR[int(pred_y[i].item())],
                        "conf_y": float(conf_y[i].item()),
                        "pred_yh_id": int(pred_yh[i].item()),
                        "pred_yh_char": IDX_TO_CHAR[int(pred_yh[i].item())],
                        "conf_yh": float(conf_yh[i].item()),
                        "correct_yh": int(pred_yh[i].item()) == int(class_ids[i]),
                        "l1": float(torch.mean(torch.abs(y_hat[i] - y_dev[i])).item()),
                    }
                )
                idx_global += 1
    return records


def select_diverse_correct(records: list[dict], preferred_chars: list[str], k: int = 6) -> list[dict]:
    """Elige `k` muestras bien clasificadas cubriendo distintos caracteres y subgrupos."""
    correct = [r for r in records if r["correct_yh"] and r["pred_y_id"] == r["true_id"]]
    correct.sort(key=lambda r: r["conf_yh"], reverse=True)

    selected: list[dict] = []
    used_chars: set[str] = set()

    for ch in preferred_chars:
        for r in correct:
            if r["true_char"] == ch and ch not in used_chars:
                selected.append(r)
                used_chars.add(ch)
                break
        if len(selected) >= k:
            return selected

    for r in correct:
        if r["true_char"] not in used_chars:
            selected.append(r)
            used_chars.add(r["true_char"])
        if len(selected) >= k:
            break
    return selected


def select_diverse_errors(records: list[dict], k: int = 6) -> list[dict]:
    """Elige `k` errores representativos (o casos de menor confianza si hubiera <k errores)."""
    errors = [r for r in records if not r["correct_yh"]]
    # Priorizar errores donde el glifo real sí era legible para el juez (fallo atribuible a G)
    errors.sort(key=lambda r: (r["pred_y_id"] == r["true_id"], r["conf_yh"]), reverse=True)

    selected: list[dict] = []
    used_chars: set[str] = set()
    for r in errors:
        if r["true_char"] not in used_chars:
            selected.append(r)
            used_chars.add(r["true_char"])
        if len(selected) >= k:
            return selected

    for r in errors:
        if r not in selected:
            selected.append(r)
        if len(selected) >= k:
            return selected

    # Respaldo si hay menos de k errores: muestras con menor confianza
    low_conf = sorted(records, key=lambda r: r["conf_yh"])
    for r in low_conf:
        if r not in selected:
            selected.append(r)
        if len(selected) >= k:
            break
    return selected


def render_gradcam_panel(
    samples: list[dict],
    gradcam: GradCAM,
    device: torch.device,
    title: str,
    output_path: Path,
    explain_predicted: bool = True,
) -> list[dict]:
    """Renderiza panel de 5 columnas: x | y | Grad-CAM(y) | ŷ | Grad-CAM(ŷ)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(11.5, 2.25 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Entrada (x)",
        "Referencia (y)",
        "Grad-CAM (y)",
        "Generado (ŷ)",
        "Grad-CAM (ŷ)",
    ]

    meta_out: list[dict] = []
    for row, rec in enumerate(samples):
        x_np = denormalize_tensor(rec["x"])[0, 0]
        y_np = denormalize_tensor(rec["y"])[0, 0]
        yh_np = denormalize_tensor(rec["y_hat"])[0, 0]

        cam_y, pred_y, conf_y = gradcam(rec["y"].to(device), target_class=rec["true_id"])
        target_for_yh = None if explain_predicted else rec["true_id"]
        cam_yh, pred_yh, conf_yh = gradcam(rec["y_hat"].to(device), target_class=target_for_yh)

        sim = cam_cosine_similarity(cam_y, cam_yh)
        ov_y = overlay_cam_on_glyph(y_np, cam_y)
        ov_yh = overlay_cam_on_glyph(yh_np, cam_yh)

        imgs = [x_np, y_np, ov_y, yh_np, ov_yh]
        cmaps = ["gray", "gray", None, "gray", None]

        for col in range(5):
            ax = axes[row, col]
            if cmaps[col] is not None:
                ax.imshow(imgs[col], cmap=cmaps[col], vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9.5, fontweight="bold")

        axes[row, 0].set_ylabel(
            f"[{rec['split']}]\n'{rec['true_char']}'",
            fontsize=9,
            rotation=0,
            labelpad=36,
            va="center",
        )
        axes[row, 2].set_xlabel(
            f"Pred y: '{IDX_TO_CHAR[pred_y]}' ({conf_y:.2f})",
            fontsize=8,
            color="#1E8449" if pred_y == rec["true_id"] else "#B03A2E",
        )
        ok_yh = pred_yh == rec["true_id"]
        axes[row, 4].set_xlabel(
            f"Pred ŷ: '{IDX_TO_CHAR[pred_yh]}' ({conf_yh:.2f}) | cos={sim:.2f}",
            fontsize=8,
            color="#1E8449" if ok_yh else "#B03A2E",
        )

        meta_out.append(
            {
                "split": rec["split"],
                "true_char": rec["true_char"],
                "subgroup": rec["subgroup"],
                "pred_y": IDX_TO_CHAR[pred_y],
                "conf_y": round(conf_y, 4),
                "pred_yh": IDX_TO_CHAR[pred_yh],
                "conf_yh": round(conf_yh, 4),
                "correct_yh": ok_yh,
                "cam_cosine_sim": round(sim, 4),
                "l1": round(rec["l1"], 4),
            }
        )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return meta_out


def render_cross_judge_panel(
    samples: list[dict],
    cam_r18: GradCAM,
    cam_eff: GradCAM,
    device: torch.device,
    title: str,
    output_path: Path,
) -> list[dict]:
    """Compara atención Grad-CAM de ResNet-18 vs EfficientNet-B0 sobre los mismos glifos `ŷ`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(9.6, 2.25 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Entrada (x)",
        "Generado (ŷ)",
        "Grad-CAM ResNet-18",
        "Grad-CAM EfficientNet-B0",
    ]

    meta_out: list[dict] = []
    for row, rec in enumerate(samples):
        x_np = denormalize_tensor(rec["x"])[0, 0]
        yh_np = denormalize_tensor(rec["y_hat"])[0, 0]
        yh_dev = rec["y_hat"].to(device)

        c_r18, p_r18, cf_r18 = cam_r18(yh_dev)
        c_eff, p_eff, cf_eff = cam_eff(yh_dev)
        sim_judges = cam_cosine_similarity(c_r18, c_eff)

        imgs = [
            x_np,
            yh_np,
            overlay_cam_on_glyph(yh_np, c_r18),
            overlay_cam_on_glyph(yh_np, c_eff),
        ]
        cmaps = ["gray", "gray", None, None]

        for col in range(4):
            ax = axes[row, col]
            if cmaps[col] is not None:
                ax.imshow(imgs[col], cmap=cmaps[col], vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9.5, fontweight="bold")

        axes[row, 0].set_ylabel(
            f"[{rec['split']}]\n'{rec['true_char']}'",
            fontsize=9,
            rotation=0,
            labelpad=36,
            va="center",
        )
        axes[row, 2].set_xlabel(
            f"'{IDX_TO_CHAR[p_r18]}' ({cf_r18:.2f})",
            fontsize=8,
            color="#1E8449" if p_r18 == rec["true_id"] else "#B03A2E",
        )
        axes[row, 3].set_xlabel(
            f"'{IDX_TO_CHAR[p_eff]}' ({cf_eff:.2f}) | cos={sim_judges:.2f}",
            fontsize=8,
            color="#1E8449" if p_eff == rec["true_id"] else "#B03A2E",
        )

        meta_out.append(
            {
                "split": rec["split"],
                "true_char": rec["true_char"],
                "resnet18_pred": IDX_TO_CHAR[p_r18],
                "resnet18_conf": round(cf_r18, 4),
                "effnet_pred": IDX_TO_CHAR[p_eff],
                "effnet_conf": round(cf_eff, 4),
                "cross_judge_cam_cos": round(sim_judges, 4),
            }
        )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return meta_out


def compute_bulk_cam_alignment(
    records: list[dict],
    gradcam: GradCAM,
    device: torch.device,
    max_samples: int = 80,
) -> dict[str, float]:
    """Calcula similitud coseno promedio de Grad-CAM(y) vs Grad-CAM(ŷ) sobre una muestra."""
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(records))
    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)

    sims_all: list[float] = []
    sims_correct: list[float] = []
    sims_wrong: list[float] = []

    for idx in indices:
        rec = records[int(idx)]
        cam_y, _, _ = gradcam(rec["y"].to(device), target_class=rec["true_id"])
        cam_yh, pred_yh, _ = gradcam(rec["y_hat"].to(device))
        sim = cam_cosine_similarity(cam_y, cam_yh)
        sims_all.append(sim)
        if pred_yh == rec["true_id"]:
            sims_correct.append(sim)
        else:
            sims_wrong.append(sim)

    return {
        "n_evaluated": len(sims_all),
        "mean_cos_all": round(float(np.mean(sims_all)), 4) if sims_all else 0.0,
        "std_cos_all": round(float(np.std(sims_all)), 4) if sims_all else 0.0,
        "mean_cos_correct": round(float(np.mean(sims_correct)), 4) if sims_correct else 0.0,
        "mean_cos_wrong": round(float(np.mean(sims_wrong)), 4) if sims_wrong else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera mapas Grad-CAM sobre jueces congelados evaluando glifos reales y generados."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_pairs"))
    parser.add_argument(
        "--generator-ckpt",
        type=Path,
        default=Path("checkpoints/pix2pix_pretrained.pth"),
    )
    parser.add_argument("--judges-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figuras"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--tag", type=str, default="pretrain")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[INFO] Grad-CAM | device={device} | data={args.data_dir} | "
        f"gen={args.generator_ckpt} | tag={args.tag}"
    )

    gen = load_generator(args.generator_ckpt, device)
    r18 = load_judge("resnet18", args.judges_dir / "judge_resnet18.pth", device)
    eff = load_judge("efficientnet_b0", args.judges_dir / "judge_efficientnet_b0.pth", device)

    cam_r18 = GradCAM(r18, get_target_layer(r18, "resnet18"))
    cam_eff = GradCAM(eff, get_target_layer(eff, "efficientnet_b0"))

    val_records = collect_split_samples(args.data_dir, "val", gen, r18, device)
    unseen_records = collect_split_samples(args.data_dir, "test_unseen", gen, r18, device)

    # 1. Correctos diversos en val (Mayús, Minús, Dígitos, Signos)
    sel_val = select_diverse_correct(val_records, ["A", "R", "g", "e", "5", "!"], k=6)
    fig_val = args.figures_dir / f"gradcam_{args.tag}_val_correctos.png"
    meta_val = render_gradcam_panel(
        sel_val,
        cam_r18,
        device,
        title=f"Grad-CAM (ResNet-18) — Aciertos en Validación ({args.tag})",
        output_path=fig_val,
    )

    # 2. Zero-Shot en test_unseen (caracteres nunca vistos por G)
    sel_unseen = select_diverse_correct(unseen_records, ["G", "Q", "k", "4", "?", "&"], k=6)
    fig_unseen = args.figures_dir / f"gradcam_{args.tag}_unseen_zeroshot.png"
    meta_unseen = render_gradcam_panel(
        sel_unseen,
        cam_r18,
        device,
        title=f"Grad-CAM (ResNet-18) — Generalización Zero-Shot en test_unseen ({args.tag})",
        output_path=fig_unseen,
    )

    # 3. Errores representativos (mezclando val y test_unseen)
    sel_err = select_diverse_errors(val_records + unseen_records, k=6)
    fig_err = args.figures_dir / f"gradcam_{args.tag}_errores.png"
    meta_err = render_gradcam_panel(
        sel_err,
        cam_r18,
        device,
        title=f"Grad-CAM (ResNet-18) — Diagnóstico de Errores y Confusiones ({args.tag})",
        output_path=fig_err,
        explain_predicted=True,
    )

    # 4. Comparativa cruzada ResNet-18 vs EfficientNet-B0
    cross_pool = (sel_val[:3] + sel_unseen[:2])[:5]
    fig_cross = args.figures_dir / f"gradcam_{args.tag}_resnet_vs_effnet.png"
    meta_cross = render_cross_judge_panel(
        cross_pool,
        cam_r18,
        cam_eff,
        device,
        title=f"Consistencia entre Jueces: ResNet-18 vs EfficientNet-B0 ({args.tag})",
        output_path=fig_cross,
    )

    # Estadísticas globales de alineación espacial de atención
    align_val = compute_bulk_cam_alignment(val_records, cam_r18, device, max_samples=80)
    align_unseen = compute_bulk_cam_alignment(unseen_records, cam_r18, device, max_samples=80)

    cam_r18.remove_hooks()
    cam_eff.remove_hooks()

    summary = {
        "tag": args.tag,
        "data_dir": str(args.data_dir),
        "generator_ckpt": str(args.generator_ckpt),
        "alignment_val": align_val,
        "alignment_test_unseen": align_unseen,
        "panels": {
            "val_correctos": {"figure": str(fig_val), "samples": meta_val},
            "unseen_zeroshot": {"figure": str(fig_unseen), "samples": meta_unseen},
            "errores": {"figure": str(fig_err), "samples": meta_err},
            "resnet_vs_effnet": {"figure": str(fig_cross), "samples": meta_cross},
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"gradcam_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Alineación Grad-CAM (val):         {align_val}")
    print(f"[OK] Alineación Grad-CAM (test_unseen): {align_unseen}")
    print(f"[OK] Resumen JSON: {out_json}")
    print(f"[OK] Figuras: {fig_val}, {fig_unseen}, {fig_err}, {fig_cross}")


if __name__ == "__main__":
    main()
