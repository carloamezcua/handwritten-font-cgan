"""Entrenamiento y congelado de clasificadores jueces (ResNet-18 y EfficientNet-B0).

Entrena ambos modelos sobre el dataset sintético usando la partición `judge_split`
(las 75 clases de `CHARSET` separadas estrictamente por fuente tipográfica), con:
- Optimizador AdamW + CosineAnnealingLR.
- Early Stopping monitoreando Macro-F1 en validación.
- Guardado del mejor checkpoint en `checkpoints/judge_<modelo>.pth`.
- Curvas de entrenamiento/validación en `docs/figuras/train_judge_<modelo>.png`.

Uso en Colab GPU:
    python -m src.train.train_judges --model all --epochs 30 --batch-size 64 --pretrained

Prueba rápida en CPU local (sin descargar pesos ni sobrescribir figuras):
    python -m src.train.train_judges --model all --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.config import CHARSET, IMG_SIZE, SEED
from src.data.dataset import GlyphJudgeDataset
from src.models.judges import build_judge
from src.utils import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_judge(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa un juez y devuelve pérdida promedio, Accuracy y Macro-F1."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            bsz = images.size(0)
            total_loss += float(loss.item()) * bsz
            total_samples += bsz

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
    }


def save_judge_curves(
    history: dict[str, list[float]],
    model_name: str,
    output_path: Path,
) -> None:
    """Guarda curvas de pérdida, Accuracy y Macro-F1 de entrenamiento/validación."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val loss", marker="s", markersize=3)
    ax1.set_title(f"{model_name} — Pérdida (Cross-Entropy)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    ax2.plot(
        epochs,
        history["val_macro_f1"],
        label="Val Macro-F1",
        marker="^",
        markersize=4,
        linewidth=2,
    )
    ax2.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="Meta 0.90")
    ax2.set_title(f"{model_name} — Accuracy y Macro-F1")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_single_judge(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pretrained: bool,
    checkpoint_path: Path,
    figure_path: Path,
    history_path: Path,
) -> dict:
    """Entrena un clasificador juez con early stopping sobre Macro-F1 de validación."""
    print(f"\n=== Entrenando juez: {model_name} (pretrained={pretrained}, device={device}) ===")
    model = build_judge(model_name, num_classes=len(CHARSET), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_f1 = -1.0
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = images.size(0)
            running_loss += float(loss.item()) * bsz
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += bsz

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_metrics = evaluate_judge(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["lr"].append(current_lr)

        print(
            f"  Época {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_name": model_name,
                    "num_classes": len(CHARSET),
                    "img_size": IMG_SIZE,
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "best_val_acc": best_acc,
                    "state_dict": model.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stopping en época {epoch} "
                    f"(sin mejora en Macro-F1 durante {patience} épocas)."
                )
                break

    elapsed = time.time() - start_time
    save_judge_curves(history, model_name, figure_path)

    summary = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_acc": best_acc,
        "elapsed_sec": round(elapsed, 2),
        "checkpoint_path": str(checkpoint_path),
        "figure_path": str(figure_path),
        "history": history,
    }
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"  [OK] Mejor {model_name}: época {best_epoch} | "
        f"val_macro_f1={best_f1:.4f}, val_acc={best_acc:.4f} ({elapsed:.1f}s)"
    )
    print(f"  [OK] Checkpoint: {checkpoint_path}")
    print(f"  [OK] Curvas:     {figure_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena clasificadores jueces (ResNet-18 y EfficientNet-B0)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["resnet18", "efficientnet_b0", "all"],
        help="Arquitectura del juez a entrenar.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic_pairs"),
        help="Carpeta del dataset sintético con metadata.csv.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directorio para guardar los pesos .pth.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figuras"),
        help="Directorio para guardar curvas de entrenamiento.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/judges"),
        help="Directorio para historiales JSON.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Inicializa con pesos preentrenados de ImageNet (recomendado en Colab GPU).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prueba rápida en CPU (1 época, 32 train / 16 val, guarda en results/smoke_test/judges).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")

    train_ds = GlyphJudgeDataset(
        root_dir=args.data_dir,
        split="train",
        split_column="judge_split",
        augment=True,
    )
    val_ds = GlyphJudgeDataset(
        root_dir=args.data_dir,
        split="val",
        split_column="judge_split",
        augment=False,
    )

    if args.dry_run:
        train_ds = Subset(train_ds, list(range(min(32, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(16, len(val_ds)))))
        epochs = 1
        batch_size = 8
        pretrained = False
        ckpt_dir = Path("results/smoke_test/judges")
        fig_dir = Path("results/smoke_test/judges")
        res_dir = Path("results/smoke_test/judges")
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        pretrained = args.pretrained
        ckpt_dir = args.checkpoints_dir
        fig_dir = args.figures_dir
        res_dir = args.results_dir

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    models_to_train = (
        ["resnet18", "efficientnet_b0"] if args.model == "all" else [args.model]
    )

    for model_name in models_to_train:
        seed_everything(args.seed)
        train_single_judge(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            pretrained=pretrained,
            checkpoint_path=ckpt_dir / f"judge_{model_name}.pth",
            figure_path=fig_dir / f"train_judge_{model_name}.png",
            history_path=res_dir / f"judge_{model_name}_history.json",
        )


if __name__ == "__main__":
    main()
