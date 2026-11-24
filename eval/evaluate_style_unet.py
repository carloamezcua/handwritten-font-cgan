"""Evaluación cuantitativa y visual del experimento StyleUNetGenerator con los jueces congelados.

Usa EXACTAMENTE los mismos clasificadores congelados (`ResNet-18` y `EfficientNet-B0`) y
la misma función de FID de dominio que `eval/evaluate_models.py` para comparar directamente
`StyleUNet` (Few-Shot sin fine-tuning y/o con fine-tuning) contra `Pix2Pix`.

Uso:
    python -m eval.evaluate_style_unet \\
        --data-dir data/real_pairs \\
        --models style_fewshot=checkpoints/style_unet_pretrained.pth \\
                 style_finetune=checkpoints/style_unet_finetuned.pth \\
        --tag style_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.evaluate_models import compute_frechet_distance
from src.config import CHARSET, SEED
from src.data.style_dataset import StyleGlyphPairDataset
from src.models.judges import build_judge, freeze_model
from src.models.style_unet import StyleUNetGenerator
from src.utils import denormalize_tensor, seed_everything


def load_style_generator(ckpt_path: Path, device: torch.device) -> tuple[StyleUNetGenerator, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: '{ckpt_path}'.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    style_dim = int(ckpt.get("style_dim", 128))
    num_refs = int(ckpt.get("num_style_refs", 4))
    gen = StyleUNetGenerator(style_dim=style_dim).to(device)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen, num_refs


def load_judges(judges_dir: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    judges: dict[str, torch.nn.Module] = {}
    for name in ("resnet18", "efficientnet_b0"):
        p = judges_dir / f"judge_{name}.pth"
        if not p.exists():
            raise FileNotFoundError(f"No se encontró el juez '{p}'.")
        model = build_judge(name, num_classes=len(CHARSET), pretrained=False).to(device)
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        freeze_model(model)
        judges[name] = model
    return judges


def _extract_judge_outputs(
    judge: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds, confs, feats = [], [], []
    head = judge.backbone.fc if hasattr(judge.backbone, "fc") else judge.backbone.classifier
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size].to(device)
            emb = judge.extract_features(batch)
            logits = head(emb)
            probs = torch.softmax(logits, dim=1)
            c, p = probs.max(dim=1)
            preds.append(p.cpu().numpy())
            confs.append(c.cpu().numpy())
            feats.append(emb.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(feats, axis=0)


def _save_style_comparison_grid(
    x_list: torch.Tensor,
    refs_list: torch.Tensor,
    y_real_list: torch.Tensor,
    gen_outputs: dict[str, torch.Tensor],
    chars: list[str],
    out_path: Path,
    title: str,
    max_samples: int = 8,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_samples, len(chars))
    model_names = list(gen_outputs.keys())
    ncols = 3 + len(model_names)  # Ref[0] | Entrada x | modelos... | Objetivo y

    fig, axes = plt.subplots(n, ncols, figsize=(2.1 * ncols, 2.0 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ["Ref estilo", "Entrada (x)"] + model_names + ["Real (y)"]
    for r in range(n):
        row_imgs = [
            denormalize_tensor(refs_list[r, 0])[0],
            denormalize_tensor(x_list[r])[0],
            *[denormalize_tensor(gen_outputs[m][r])[0] for m in model_names],
            denormalize_tensor(y_real_list[r])[0],
        ]
        for c, im in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[c], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"'{chars[r]}'", fontsize=10, rotation=0, labelpad=14)

    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa StyleUNetGenerator con los jueces congelados.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_pairs"))
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Lista de `nombre=ruta.pth` (ej. style_fewshot=checkpoints/style_unet_pretrained.pth).",
    )
    parser.add_argument("--judges-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figuras"))
    parser.add_argument("--tag", type=str, default="style_real")
    parser.add_argument("--ood-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_specs: list[tuple[str, Path]] = []
    for item in args.models:
        if "=" not in item:
            raise ValueError(f"Formato inválido '{item}'. Usa `nombre=ruta.pth`.")
        name, path_str = item.split("=", 1)
        model_specs.append((name.strip(), Path(path_str.strip())))

    judges = load_judges(args.judges_dir, device)
    loaded_models = {name: load_style_generator(path, device) for name, path in model_specs}

    metrics_out: dict = {"data_dir": str(args.data_dir), "tag": args.tag, "splits": {}}
    md_lines = [
        f"# Evaluación StyleUNet (`{args.tag}` — `{args.data_dir}`)\n",
        "| Split | Modelo | Juez | Accuracy | Macro-F1 | Confianza | Pass >= 0.70 | L1 media | FID dominio |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for split in ("val", "test_unseen"):
        num_refs = next(iter(loaded_models.values()))[1]
        ds = StyleGlyphPairDataset(
            args.data_dir,
            split=split,
            num_style_refs=num_refs,
            augment=False,
            stroke_augment=False,
            seed=args.seed,
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        xs, ys, refs_all, labels, chars = [], [], [], [], []
        for x, y, refs, cid, ch, _ in loader:
            xs.append(x)
            ys.append(y)
            refs_all.append(refs)
            labels.append(cid.numpy())
            chars.extend(list(ch))

        x_cat = torch.cat(xs, dim=0)
        y_cat = torch.cat(ys, dim=0)
        refs_cat = torch.cat(refs_all, dim=0)
        y_true = np.concatenate(labels)
        present_labels = sorted(set(y_true.tolist()))

        metrics_out["splits"][split] = {}
        ref_feats_by_judge: dict[str, np.ndarray] = {}

        # Referencia real `y`
        metrics_out["splits"][split]["reference_y"] = {}
        for jname, judge in judges.items():
            preds, confs, feats = _extract_judge_outputs(judge, y_cat, device)
            ref_feats_by_judge[jname] = feats
            acc = float(accuracy_score(y_true, preds))
            f1 = float(f1_score(y_true, preds, labels=present_labels, average="macro", zero_division=0))
            pass_rate = float(np.mean(confs >= args.ood_threshold))
            m_dict = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "mean_conf": round(float(confs.mean()), 4),
                "pass_rate": round(pass_rate, 4),
                "l1": 0.0,
                "fid": 0.0,
            }
            metrics_out["splits"][split]["reference_y"][jname] = m_dict
            md_lines.append(
                f"| `{split}` | `reference_y` | `{jname}` | {acc*100:.2f}% | {f1*100:.2f}% | "
                f"{confs.mean():.4f} | {pass_rate*100:.2f}% | — | 0.00 |"
            )

        # Modelos StyleUNet
        gen_outputs: dict[str, torch.Tensor] = {}
        for mname, (gen, _) in loaded_models.items():
            yhat_chunks = []
            with torch.no_grad():
                for i in range(0, len(x_cat), 64):
                    yh = gen(x_cat[i : i + 64].to(device), refs_cat[i : i + 64].to(device))
                    yhat_chunks.append(yh.cpu())
            yhat_cat = torch.cat(yhat_chunks, dim=0)
            gen_outputs[mname] = yhat_cat
            l1_val = float(torch.abs(yhat_cat - y_cat).mean().item())

            metrics_out["splits"][split][mname] = {}
            for jname, judge in judges.items():
                preds, confs, feats = _extract_judge_outputs(judge, yhat_cat, device)
                acc = float(accuracy_score(y_true, preds))
                f1 = float(
                    f1_score(y_true, preds, labels=present_labels, average="macro", zero_division=0)
                )
                pass_rate = float(np.mean(confs >= args.ood_threshold))
                fid = compute_frechet_distance(ref_feats_by_judge[jname], feats)
                m_dict = {
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                    "mean_conf": round(float(confs.mean()), 4),
                    "pass_rate": round(pass_rate, 4),
                    "l1": round(l1_val, 4),
                    "fid": round(fid, 2),
                }
                metrics_out["splits"][split][mname][jname] = m_dict
                print(
                    f"  [{split:11s} | {mname:15s} | {jname:15s}] "
                    f"Acc={acc*100:6.2f}% | Macro-F1={f1*100:6.2f}% | L1={l1_val:.4f} | FID={fid:.2f}"
                )
                md_lines.append(
                    f"| `{split}` | `{mname}` | `{jname}` | {acc*100:.2f}% | {f1*100:.2f}% | "
                    f"{confs.mean():.4f} | {pass_rate*100:.2f}% | {l1_val:.4f} | {fid:.2f} |"
                )

        # Guardar cuadrícula visual comparativa
        _save_style_comparison_grid(
            x_cat,
            refs_cat,
            y_cat,
            gen_outputs,
            chars,
            args.figures_dir / f"style_eval_{args.tag}_{split}.png",
            title=f"StyleUNet ({args.tag}) — {split} (Refs estilo | x | Generado | Real)",
            max_samples=8,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / f"metrics_{args.tag}.json"
    md_path = args.results_dir / f"summary_{args.tag}.md"
    json_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\n[OK] Métricas JSON: {json_path}")
    print(f"[OK] Resumen MD:    {md_path}")


if __name__ == "__main__":
    main()
