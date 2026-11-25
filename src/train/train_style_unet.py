"""Entrenamiento del experimento aditivo Style-Conditioned U-Net (Style Encoder + AdaIN).

Modos soportados:
- `pretrain`: entrena sobre `data/synthetic_pairs/` (14 fuentes en `train`, 3 inéditas en `val`)
  con aumento morfológico consistente de grosor de trazo (`--stroke-augment`) para transferir
  tanto trazos finos (stylus) como gruesos en inferencia Few-Shot sin fine-tuning.
- `finetune`: adaptación ligera sobre `data/real_pairs/` partiendo de `style_unet_pretrained.pth`.

Protocolo anti-fuga:
- `test_unseen/` nunca se usa para entrenar ni seleccionar checkpoints.
- `style_refs` excluye siempre el carácter objetivo y jamás usa `TEST_UNSEEN_CHARS`.

Uso:
    python -m src.train.train_style_unet --mode pretrain --epochs 60
    python -m src.train.train_style_unet --mode finetune --data-dir data/real_pairs --epochs 40 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SEED, STYLE_UNET_CONFIG
from src.data.style_dataset import StyleGlyphPairDataset
from src.models.pix2pix import PatchGANDiscriminator
from src.models.style_unet import StyleUNetGenerator, StyleUNetLoss
from src.utils import save_triplet_grid, seed_everything


def _evaluate_val(
    gen: StyleUNetGenerator,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Devuelve `(val_l1, val_ink_diff, val_score)` sin modificar pesos."""
    gen.eval()
    l1_sum = 0.0
    ink_sum = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, y, refs, _, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            refs = refs.to(device)
            y_hat = gen(x, refs)

            bs = x.size(0)
            l1_sum += torch.abs(y_hat - y).mean().item() * bs
            ink_fake = ((1.0 - y_hat) * 0.5).mean(dim=(1, 2, 3))
            ink_real = ((1.0 - y) * 0.5).mean(dim=(1, 2, 3))
            ink_sum += torch.abs(ink_fake - ink_real).mean().item() * bs
            n_samples += bs

    val_l1 = l1_sum / max(n_samples, 1)
    val_ink = ink_sum / max(n_samples, 1)
    # Penalizar colapso a lienzo en blanco sumando diferencia de densidad de tinta
    val_score = val_l1 + 0.5 * val_ink
    return val_l1, val_ink, val_score


def _save_curves(history: dict[str, list[float]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_l1"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["g_loss"], label="G total", linewidth=1.8)
    axes[0].plot(epochs, history["d_loss"], label="D loss", linewidth=1.8)
    axes[0].plot(epochs, history["g_style"], label="Style loss", linestyle="--", linewidth=1.4)
    axes[0].set_title("Pérdidas Adversarial y Estilo")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_l1"], label="Train L1", linewidth=1.8)
    axes[1].plot(epochs, history["val_l1"], label="Val L1", linewidth=1.8)
    axes[1].plot(epochs, history["val_ink_diff"], label="Val Ink Diff", linestyle=":", linewidth=1.6)
    axes[1].set_title("Reconstrucción L1 y Densidad de Tinta")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Error medio")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena StyleUNetGenerator (Style Encoder + AdaIN).")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=Path("checkpoints/style_unet_pretrained.pth"),
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figuras"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/style_unet"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=STYLE_UNET_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--style-dim", type=int, default=STYLE_UNET_CONFIG["style_dim"])
    parser.add_argument("--num-style-refs", type=int, default=STYLE_UNET_CONFIG["num_style_refs"])
    parser.add_argument("--lambda-l1", type=float, default=STYLE_UNET_CONFIG["lambda_l1"])
    parser.add_argument("--lambda-style", type=float, default=STYLE_UNET_CONFIG["lambda_style"])
    parser.add_argument("--lambda-ink", type=float, default=STYLE_UNET_CONFIG["lambda_ink"])
    parser.add_argument("--no-stroke-augment", action="store_true")
    parser.add_argument("--freeze-style-encoder", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=STYLE_UNET_CONFIG["sample_every"])
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    mode_cfg = STYLE_UNET_CONFIG["modes"][args.mode]
    data_dir = args.data_dir or mode_cfg["data_dir"]
    epochs = args.epochs if args.epochs is not None else mode_cfg["epochs"]
    lr = args.lr if args.lr is not None else mode_cfg["lr"]
    stroke_aug = (args.mode == "pretrain") and (not args.no_stroke_augment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"=== Entrenando StyleUNet (mode={args.mode}, epochs={epochs}, batch={args.batch_size}, "
        f"lr={lr}, stroke_aug={stroke_aug}, device={device}) ==="
    )

    train_ds = StyleGlyphPairDataset(
        data_dir,
        split="train",
        num_style_refs=args.num_style_refs,
        augment=True,
        stroke_augment=stroke_aug,
        seed=args.seed,
    )
    val_ds = StyleGlyphPairDataset(
        data_dir,
        split="val",
        num_style_refs=args.num_style_refs,
        augment=False,
        stroke_augment=False,
        seed=args.seed,
    )
    unseen_ds = StyleGlyphPairDataset(
        data_dir,
        split="test_unseen",
        num_style_refs=args.num_style_refs,
        augment=False,
        stroke_augment=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    unseen_loader = DataLoader(
        unseen_ds,
        batch_size=min(args.batch_size, len(unseen_ds)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    gen = StyleUNetGenerator(style_dim=args.style_dim).to(device)
    disc = PatchGANDiscriminator().to(device)
    criterion = StyleUNetLoss(
        lambda_l1=args.lambda_l1,
        lambda_style=args.lambda_style,
        lambda_ink=args.lambda_ink,
    )

    if args.mode == "finetune":
        if not args.pretrained_ckpt.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint preentrenado: '{args.pretrained_ckpt}'."
            )
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["generator_state_dict"])
        if "discriminator_state_dict" in ckpt:
            disc.load_state_dict(ckpt["discriminator_state_dict"])
        print(f"[INFO] Pesos preentrenados cargados desde: {args.pretrained_ckpt}")

        if args.freeze_style_encoder:
            for p in gen.style_encoder.parameters():
                p.requires_grad = False
            print("[INFO] StyleEncoder congelado durante fine-tuning.")

    opt_g = torch.optim.Adam(
        [p for p in gen.parameters() if p.requires_grad], lr=lr, betas=STYLE_UNET_CONFIG["betas"]
    )
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=STYLE_UNET_CONFIG["betas"])

    # Muestras fijas para visualizar evolución
    val_x_fix, val_y_fix, val_refs_fix, _, val_chars_fix, _ = next(iter(val_loader))
    unseen_x_fix, unseen_y_fix, unseen_refs_fix, _, unseen_chars_fix, _ = next(iter(unseen_loader))

    ckpt_name = mode_cfg["ckpt_name"]
    best_ckpt_path = args.checkpoints_dir / ckpt_name
    last_ckpt_path = args.checkpoints_dir / ckpt_name.replace(".pth", "_last.pth")
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.results_dir / args.mode / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "g_loss": [],
        "d_loss": [],
        "g_style": [],
        "train_l1": [],
        "val_l1": [],
        "val_ink_diff": [],
        "val_score": [],
    }
    best_val_score = float("inf")
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        g_sum = d_sum = st_sum = l1_sum = 0.0
        n_batches = 0

        for x, y, refs, _, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            refs = refs.to(device)

            # 1) Actualizar Discriminador
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_y_det = gen(x, refs)
            pred_real = disc(x, y)
            pred_fake = disc(x, fake_y_det)
            loss_d, d_m = criterion.discriminator_loss(pred_real, pred_fake)
            loss_d.backward()
            opt_d.step()

            # 2) Actualizar Generador + StyleEncoder
            opt_g.zero_grad(set_to_none=True)
            fake_y = gen(x, refs)
            pred_fake_g = disc(x, fake_y)
            z_fake = gen.encode_style(fake_y)
            with torch.no_grad():
                z_real = gen.encode_style(y)
            loss_g, g_m = criterion.generator_loss(pred_fake_g, fake_y, y, z_fake, z_real)
            loss_g.backward()
            opt_g.step()

            g_sum += g_m["g_loss"]
            d_sum += d_m["d_loss"]
            st_sum += g_m["g_style"]
            l1_sum += g_m["g_l1"]
            n_batches += 1

        val_l1, val_ink, val_score = _evaluate_val(gen, val_loader, device)
        history["g_loss"].append(g_sum / max(n_batches, 1))
        history["d_loss"].append(d_sum / max(n_batches, 1))
        history["g_style"].append(st_sum / max(n_batches, 1))
        history["train_l1"].append(l1_sum / max(n_batches, 1))
        history["val_l1"].append(val_l1)
        history["val_ink_diff"].append(val_ink)
        history["val_score"].append(val_score)

        payload = {
            "arch": "StyleUNetGenerator",
            "mode": args.mode,
            "epoch": epoch,
            "style_dim": args.style_dim,
            "num_style_refs": args.num_style_refs,
            "generator_state_dict": gen.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "val_l1": val_l1,
            "val_ink_diff": val_ink,
            "val_score": val_score,
        }
        torch.save(payload, last_ckpt_path)

        improved = ""
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(payload, best_ckpt_path)
            improved = " *"

        if epoch == 1 or epoch % max(1, args.sample_every) == 0 or epoch == epochs:
            print(
                f"  Ep {epoch:03d}/{epochs} | D={history['d_loss'][-1]:.3f} "
                f"| TrainL1={history['train_l1'][-1]:.4f} | ValL1={val_l1:.4f} "
                f"| InkDiff={val_ink:.4f} | Style={history['g_style'][-1]:.4f}{improved}"
            )
            gen.eval()
            with torch.no_grad():
                val_yh = gen(val_x_fix.to(device), val_refs_fix.to(device))
            save_triplet_grid(
                val_x_fix,
                val_yh,
                val_y_fix,
                samples_dir / f"val_epoch_{epoch:03d}.png",
                chars=list(val_chars_fix),
                max_samples=8,
                title=f"StyleUNet ({args.mode}) — Época {epoch}",
            )

    # Cargar mejor checkpoint y generar cuadrículas finales
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(best_state["generator_state_dict"])
    gen.eval()
    with torch.no_grad():
        best_val_yh = gen(val_x_fix.to(device), val_refs_fix.to(device))
        best_unseen_yh = gen(unseen_x_fix.to(device), unseen_refs_fix.to(device))

    fig_curves = args.figures_dir / f"train_style_unet_{args.mode}.png"
    fig_val = args.figures_dir / f"style_unet_{args.mode}_val_grid.png"
    fig_unseen = args.figures_dir / f"style_unet_{args.mode}_unseen_grid.png"

    _save_curves(history, fig_curves, f"StyleUNet ({args.mode}) — Curvas de Entrenamiento")
    save_triplet_grid(
        val_x_fix,
        best_val_yh,
        val_y_fix,
        fig_val,
        chars=list(val_chars_fix),
        max_samples=8,
        title=f"StyleUNet ({args.mode}) — Mejor modelo (ep. {best_epoch}) en Val",
    )
    save_triplet_grid(
        unseen_x_fix,
        best_unseen_yh,
        unseen_y_fix,
        fig_unseen,
        chars=list(unseen_chars_fix),
        max_samples=8,
        title=f"StyleUNet ({args.mode}) — Mejor modelo (ep. {best_epoch}) en Test Unseen",
    )

    elapsed = time.time() - t_start
    hist_path = args.results_dir / f"style_unet_{args.mode}_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
                "elapsed_sec": round(elapsed, 2),
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[OK] StyleUNet ({args.mode}) finalizado en {elapsed:.1f}s | "
        f"mejor época={best_epoch}, best_val_score={best_val_score:.4f}"
    )
    print(f"[OK] Mejor checkpoint: {best_ckpt_path}")
    print(f"[OK] Curvas:           {fig_curves}")
    print(f"[OK] Grid Val:         {fig_val}")
    print(f"[OK] Grid Unseen:      {fig_unseen}")


if __name__ == "__main__":
    main()
