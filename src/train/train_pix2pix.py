"""Entrenamiento de Pix2Pix (U-Net + PatchGAN) para pretrain sintético, scratch y fine-tuning.

Modos soportados (`--mode`):
- `pretrain`: Preentrenamiento sobre `data/synthetic_pairs/` (Fase 3).
- `scratch`:  Experimento A desde pesos aleatorios sobre `data/real_pairs/` (Fase 5).
- `finetune`: Experimento B cargando `--pretrained-ckpt` sobre `data/real_pairs/` (Fase 5).

Regla anti-fuga (Rúbrica B3):
- `test_unseen/` nunca se usa para entrenar ni para elegir el mejor checkpoint.
- El mejor modelo se selecciona exclusivamente con `val_l1` sobre `val/`.
- Al finalizar el entrenamiento, se genera una cuadrícula visual `x | ŷ | y` sobre
  `val` y `test_unseen` usando el mejor checkpoint.

Uso en Colab GPU (Fase 3):
    python -m src.train.train_pix2pix --mode pretrain --epochs 60 --batch-size 32

Prueba rápida en CPU local:
    python -m src.train.train_pix2pix --mode pretrain --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import IMG_SIZE, SEED
from src.data.dataset import GlyphPairDataset
from src.models.pix2pix import PatchGANDiscriminator, Pix2PixLoss, UNetGenerator
from src.utils import save_triplet_grid, seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_DEFAULTS = {
    "pretrain": {
        "data_dir": Path("data/synthetic_pairs"),
        "ckpt_name": "pix2pix_pretrained.pth",
        "epochs": 60,
        "batch_size": 32,
        "lr": 2e-4,
    },
    "scratch": {
        "data_dir": Path("data/real_pairs"),
        "ckpt_name": "pix2pix_scratch.pth",
        "epochs": 150,
        "batch_size": 16,
        "lr": 2e-4,
    },
    "finetune": {
        "data_dir": Path("data/real_pairs"),
        "ckpt_name": "pix2pix_finetuned.pth",
        "epochs": 100,
        "batch_size": 16,
        "lr": 1e-4,
    },
}


def evaluate_val_l1(
    gen: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Calcula L1 promedio sobre validación sin actualizar gradientes."""
    gen.eval()
    l1 = nn.L1Loss(reduction="sum")
    total_l1 = 0.0
    total_elements = 0

    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            fake_y = gen(x)
            total_l1 += float(l1(fake_y, y).item())
            total_elements += y.numel()

    return total_l1 / max(total_elements, 1)


def save_pix2pix_curves(
    history: dict[str, list[float]],
    mode: str,
    output_path: Path,
) -> None:
    """Guarda curvas de D_loss, G_adv y Train/Val L1."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["d_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epochs, history["d_loss"], label="D loss", linewidth=1.6)
    ax1.plot(epochs, history["g_adv"], label="G adv loss", linewidth=1.6)
    ax1.set_title(f"Pix2Pix ({mode}) — Pérdidas Adversariales")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_l1"], label="Train L1", linewidth=1.6)
    ax2.plot(epochs, history["val_l1"], label="Val L1", linewidth=2.0)
    ax2.set_title(f"Pix2Pix ({mode}) — Reconstrucción L1")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("L1 Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_split_grid(
    gen: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    max_samples: int = 8,
    title: str | None = None,
) -> None:
    """Genera y guarda una cuadrícula `x | ŷ | y` para el primer lote de un DataLoader."""
    gen.eval()
    x, y, _, chars = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        fake_y = gen(x)
    save_triplet_grid(
        x=x,
        y_hat=fake_y,
        y_real=y,
        output_path=output_path,
        chars=list(chars),
        max_samples=max_samples,
        title=title,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena Pix2Pix (pretrain sintético, from scratch o fine-tuning)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "scratch", "finetune"],
        help="Modo de entrenamiento.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directorio del dataset (por defecto según --mode).",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=Path("checkpoints/pix2pix_pretrained.pth"),
        help="Checkpoint base para --mode finetune.",
    )
    parser.add_argument(
        "--reset-discriminator",
        action="store_true",
        help="En finetune, reinicia el discriminador en lugar de cargar sus pesos previos.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Carpeta donde guardar checkpoints .pth.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figuras"),
        help="Carpeta donde guardar curvas y cuadrículas finales.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/pix2pix"),
        help="Carpeta para historiales JSON y muestras intermedias.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--real-label", type=float, default=0.9)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prueba rápida en CPU (1 época, 16 train / 8 val / 8 unseen).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = MODE_DEFAULTS[args.mode]

    data_dir = args.data_dir or defaults["data_dir"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    lr = args.lr if args.lr is not None else defaults["lr"]
    ckpt_name = defaults["ckpt_name"]

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")

    train_ds = GlyphPairDataset(data_dir, split="train", augment=True)
    val_ds = GlyphPairDataset(data_dir, split="val", augment=False)
    unseen_ds = GlyphPairDataset(data_dir, split="test_unseen", augment=False)

    if args.dry_run:
        train_ds = Subset(train_ds, list(range(min(16, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(8, len(val_ds)))))
        unseen_ds = Subset(unseen_ds, list(range(min(8, len(unseen_ds)))))
        epochs = 1
        batch_size = 8
        ckpt_dir = Path("results/smoke_test/pix2pix")
        fig_dir = Path("results/smoke_test/pix2pix")
        res_dir = Path("results/smoke_test/pix2pix")
        sample_every = 1
    else:
        ckpt_dir = args.checkpoints_dir
        fig_dir = args.figures_dir
        res_dir = args.results_dir / args.mode
        sample_every = args.sample_every

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    unseen_loader = DataLoader(
        unseen_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    if args.mode == "finetune":
        if not args.pretrained_ckpt.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint preentrenado '{args.pretrained_ckpt}'."
            )
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["generator_state_dict"])
        if not args.reset_discriminator and "discriminator_state_dict" in ckpt:
            disc.load_state_dict(ckpt["discriminator_state_dict"])
        print(f"[INFO] Pesos preentrenados cargados desde: {args.pretrained_ckpt}")

    criterion = Pix2PixLoss(
        lambda_l1=args.lambda_l1,
        real_label=args.real_label,
        fake_label=0.0,
    )
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # Decaimiento lineal del learning rate en la segunda mitad del entrenamiento (estándar Pix2Pix)
    decay_start = max(epochs // 2, 1)

    def lr_lambda(ep: int) -> float:
        if ep < decay_start or epochs <= 1:
            return 1.0
        return max(0.05, 1.0 - (ep - decay_start) / max(epochs - decay_start, 1))

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda=lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=lr_lambda)

    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = res_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / ckpt_name
    last_ckpt_path = ckpt_dir / f"{Path(ckpt_name).stem}_last.pth"

    history: dict[str, list[float]] = {
        "d_loss": [],
        "g_loss": [],
        "g_adv": [],
        "train_l1": [],
        "val_l1": [],
        "lr": [],
    }

    best_val_l1 = float("inf")
    best_epoch = 0
    start_time = time.time()

    print(
        f"\n=== Entrenando Pix2Pix (mode={args.mode}, epochs={epochs}, "
        f"batch_size={batch_size}, lr={lr}, device={device}) ==="
    )

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        sum_d = sum_g = sum_adv = sum_l1 = 0.0
        steps = 0

        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 1. Paso del Discriminador
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    fake_y_det = gen(x)
                pred_real = disc(x, y)
                pred_fake = disc(x, fake_y_det)
                loss_d, m_d = criterion.discriminator_loss(pred_real, pred_fake)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # 2. Paso del Generador
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_y = gen(x)
                pred_fake_g = disc(x, fake_y)
                loss_g, m_g = criterion.generator_loss(pred_fake_g, fake_y, y)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            sum_d += m_d["d_loss"]
            sum_g += m_g["g_loss"]
            sum_adv += m_g["g_adv"]
            sum_l1 += m_g["g_l1"]
            steps += 1

        current_lr = opt_g.param_groups[0]["lr"]
        sched_g.step()
        sched_d.step()

        avg_d = sum_d / max(steps, 1)
        avg_g = sum_g / max(steps, 1)
        avg_adv = sum_adv / max(steps, 1)
        avg_l1 = sum_l1 / max(steps, 1)
        val_l1 = evaluate_val_l1(gen, val_loader, device)

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["g_adv"].append(avg_adv)
        history["train_l1"].append(avg_l1)
        history["val_l1"].append(val_l1)
        history["lr"].append(current_lr)

        print(
            f"  Época {epoch:03d}/{epochs:03d} | "
            f"d_loss={avg_d:.4f} | g_adv={avg_adv:.4f} | "
            f"train_l1={avg_l1:.4f} | val_l1={val_l1:.4f} | lr={current_lr:.2e}"
        )

        payload = {
            "mode": args.mode,
            "img_size": IMG_SIZE,
            "epoch": epoch,
            "val_l1": val_l1,
            "best_val_l1": min(best_val_l1, val_l1),
            "generator_state_dict": gen.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "history": history,
        }
        torch.save(payload, last_ckpt_path)

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_epoch = epoch
            torch.save(payload, best_ckpt_path)

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            generate_split_grid(
                gen=gen,
                loader=val_loader,
                device=device,
                output_path=samples_dir / f"val_epoch_{epoch:03d}.png",
                max_samples=8,
                title=f"Pix2Pix ({args.mode}) — Val época {epoch}",
            )

    elapsed = time.time() - start_time

    # Cargar mejor checkpoint para generar cuadrículas finales (val y test_unseen)
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(best_state["generator_state_dict"])

    curve_path = fig_dir / f"train_pix2pix_{args.mode}.png"
    val_grid_path = fig_dir / f"pix2pix_{args.mode}_val_grid.png"
    unseen_grid_path = fig_dir / f"pix2pix_{args.mode}_unseen_grid.png"
    history_path = res_dir / f"pix2pix_{args.mode}_history.json"

    save_pix2pix_curves(history, args.mode, curve_path)
    generate_split_grid(
        gen=gen,
        loader=val_loader,
        device=device,
        output_path=val_grid_path,
        max_samples=8,
        title=f"Pix2Pix ({args.mode}) — Mejor modelo (época {best_epoch}) en Val",
    )
    generate_split_grid(
        gen=gen,
        loader=unseen_loader,
        device=device,
        output_path=unseen_grid_path,
        max_samples=8,
        title=f"Pix2Pix ({args.mode}) — Mejor modelo (época {best_epoch}) en Test Unseen (Zero-Shot)",
    )

    summary = {
        "mode": args.mode,
        "data_dir": str(data_dir),
        "best_epoch": best_epoch,
        "best_val_l1": best_val_l1,
        "elapsed_sec": round(elapsed, 2),
        "best_checkpoint": str(best_ckpt_path),
        "curve_path": str(curve_path),
        "val_grid_path": str(val_grid_path),
        "unseen_grid_path": str(unseen_grid_path),
        "history": history,
    }
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n[OK] Pix2Pix ({args.mode}) finalizado en {elapsed:.1f}s | "
        f"mejor época={best_epoch}, best_val_l1={best_val_l1:.4f}"
    )
    print(f"[OK] Mejor checkpoint: {best_ckpt_path}")
    print(f"[OK] Curvas:           {curve_path}")
    print(f"[OK] Grid Val:         {val_grid_path}")
    print(f"[OK] Grid Unseen:      {unseen_grid_path}")


if __name__ == "__main__":
    main()
