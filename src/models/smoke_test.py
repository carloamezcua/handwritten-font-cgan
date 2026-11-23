"""Checkpoint 2 — Smoke test en CPU para Pix2Pix y clasificadores jueces.

Valida:
1. Shapes exactos y `forward + backward` sin NaNs para:
   - `UNetGenerator` (`[4, 1, 128, 128] -> [4, 1, 128, 128]`)
   - `PatchGANDiscriminator` (`[4, 1, 128, 128] x 2 -> [4, 1, 14, 14]`)
   - `ResNet18Judge` (features `[4, 512]`, logits `[4, 75]`)
   - `EfficientNetB0Judge` (features `[4, 1280]`, logits `[4, 75]`)
2. Entrenamiento de humo de 2 épocas sobre 32 imágenes sintéticas en CPU,
   guardando checkpoint, cuadrícula `x | ŷ | y` y curva de pérdidas en `results/smoke_test/`.

Uso:
    python -m src.models.smoke_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import CHARSET, IMG_SIZE, SEED
from src.data.dataset import GlyphPairDataset
from src.models.judges import EfficientNetB0Judge, ResNet18Judge
from src.models.pix2pix import PatchGANDiscriminator, Pix2PixLoss, UNetGenerator
from src.utils import save_triplet_grid, seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _assert_finite_grads(model: nn.Module, name: str) -> None:
    grad_count = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name}: gradiente nulo en '{param_name}'."
            assert torch.isfinite(param.grad).all(), (
                f"{name}: gradiente NaN/Inf en '{param_name}'."
            )
            grad_count += 1
    assert grad_count > 0, f"{name}: no se encontraron parámetros entrenables."


def run_tensor_smoke_test(device: torch.device) -> None:
    """Verifica shapes y un paso de forward + backward sin NaNs en todos los modelos."""
    print("=== 1. Smoke test de tensores y gradientes ===")
    batch_size = 4
    num_classes = len(CHARSET)
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device).tanh()
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Pix2Pix: Generador + Discriminador
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    criterion = Pix2PixLoss(lambda_l1=100.0, real_label=0.9, fake_label=0.0)

    fake_y = gen(x)
    assert fake_y.shape == (batch_size, 1, IMG_SIZE, IMG_SIZE), (
        f"Shape inesperado en UNetGenerator: {fake_y.shape}"
    )
    assert fake_y.detach().min().item() >= -1.0 and fake_y.detach().max().item() <= 1.0, (
        "Salida de UNetGenerator fuera de [-1, 1]."
    )

    pred_real = disc(x, y)
    pred_fake = disc(x, fake_y.detach())
    assert pred_real.shape == (batch_size, 1, 14, 14), (
        f"Shape inesperado en PatchGANDiscriminator: {pred_real.shape}"
    )
    assert pred_fake.shape == (batch_size, 1, 14, 14)

    loss_d, _ = criterion.discriminator_loss(pred_real, pred_fake)
    assert torch.isfinite(loss_d), f"loss_d no finito: {loss_d}"
    disc.zero_grad()
    loss_d.backward()
    _assert_finite_grads(disc, "PatchGANDiscriminator")

    pred_fake_for_g = disc(x, fake_y)
    loss_g, _ = criterion.generator_loss(pred_fake_for_g, fake_y, y)
    assert torch.isfinite(loss_g), f"loss_g no finito: {loss_g}"
    gen.zero_grad()
    loss_g.backward()
    _assert_finite_grads(gen, "UNetGenerator")
    print(
        f"  [OK] UNetGenerator {tuple(fake_y.shape)} | "
        f"PatchGANDiscriminator {tuple(pred_real.shape)} | "
        f"loss_d={loss_d.item():.4f}, loss_g={loss_g.item():.4f}"
    )

    # Jueces: ResNet-18 y EfficientNet-B0
    ce = nn.CrossEntropyLoss()
    for name, judge, expected_dim in [
        ("ResNet18Judge", ResNet18Judge(num_classes=num_classes, pretrained=False), 512),
        (
            "EfficientNetB0Judge",
            EfficientNetB0Judge(num_classes=num_classes, pretrained=False),
            1280,
        ),
    ]:
        judge = judge.to(device)
        feats = judge.extract_features(x)
        assert feats.shape == (batch_size, expected_dim), (
            f"Shape inesperado en {name}.extract_features: {feats.shape}"
        )
        logits = judge(x)
        assert logits.shape == (batch_size, num_classes), (
            f"Shape inesperado en {name}.forward: {logits.shape}"
        )
        loss_j = ce(logits, labels)
        assert torch.isfinite(loss_j), f"loss no finito en {name}: {loss_j}"
        judge.zero_grad()
        loss_j.backward()
        _assert_finite_grads(judge, name)
        print(
            f"  [OK] {name}: feats={tuple(feats.shape)}, "
            f"logits={tuple(logits.shape)}, ce_loss={loss_j.item():.4f}"
        )


def run_mini_train_smoke_test(
    device: torch.device,
    data_dir: Path,
    checkpoint_path: Path,
    out_dir: Path,
    num_samples: int = 32,
    epochs: int = 2,
    batch_size: int = 8,
) -> None:
    """Corre 2 épocas de prueba sobre 32 muestras sintéticas en CPU y guarda artefactos."""
    print(f"\n=== 2. Smoke test de entrenamiento ({epochs} épocas, {num_samples} muestras) ===")
    full_ds = GlyphPairDataset(data_dir, split="train", augment=True)
    subset = Subset(full_ds, list(range(min(num_samples, len(full_ds)))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)

    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    criterion = Pix2PixLoss(lambda_l1=100.0, real_label=0.9, fake_label=0.0)

    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    history = {"d_loss": [], "g_loss": [], "g_l1": []}
    last_batch = None

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        sum_d = sum_g = sum_l1 = 0.0
        steps = 0

        for x, y, _, chars in loader:
            x, y = x.to(device), y.to(device)

            # Paso Discriminador
            opt_d.zero_grad()
            with torch.no_grad():
                fake_y_det = gen(x)
            loss_d, m_d = criterion.discriminator_loss(
                disc(x, y), disc(x, fake_y_det)
            )
            assert torch.isfinite(loss_d), f"NaN en loss_d (época {epoch})"
            loss_d.backward()
            opt_d.step()

            # Paso Generador
            opt_g.zero_grad()
            fake_y = gen(x)
            loss_g, m_g = criterion.generator_loss(disc(x, fake_y), fake_y, y)
            assert torch.isfinite(loss_g), f"NaN en loss_g (época {epoch})"
            loss_g.backward()
            opt_g.step()

            sum_d += m_d["d_loss"]
            sum_g += m_g["g_loss"]
            sum_l1 += m_g["g_l1"]
            steps += 1
            last_batch = (x.detach(), fake_y.detach(), y.detach(), list(chars))

        avg_d = sum_d / steps
        avg_g = sum_g / steps
        avg_l1 = sum_l1 / steps
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["g_l1"].append(avg_l1)
        print(
            f"  Época {epoch}/{epochs} -> "
            f"d_loss={avg_d:.4f} | g_loss={avg_g:.4f} | g_l1={avg_l1:.4f}"
        )

    # Guardar checkpoint de prueba
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "generator_state_dict": gen.state_dict(),
            "discriminator_state_dict": disc.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    assert checkpoint_path.exists() and checkpoint_path.stat().st_size > 0

    # Guardar curva de pérdidas y cuadrícula x | ŷ | y
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ep_range = range(1, epochs + 1)
    ax.plot(ep_range, history["d_loss"], marker="o", label="D loss")
    ax.plot(ep_range, history["g_loss"], marker="o", label="G total loss")
    ax.plot(ep_range, history["g_l1"], marker="o", label="G L1")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.set_title("Checkpoint 2 — Smoke Test Pix2Pix (CPU)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    grid_path = out_dir / "grid_epoch2.png"
    assert last_batch is not None
    x_b, yhat_b, y_b, chars_b = last_batch
    save_triplet_grid(
        x_b,
        yhat_b,
        y_b,
        grid_path,
        chars=chars_b,
        max_samples=4,
        title="Checkpoint 2 — Smoke Test (2 épocas CPU)",
    )

    print(f"  [OK] Checkpoint guardado: {checkpoint_path}")
    print(f"  [OK] Curva guardada:      {curve_path}")
    print(f"  [OK] Cuadrícula guardada: {grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint 2: Smoke test en CPU.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic_pairs"),
        help="Directorio del dataset sintético.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/smoke_test_pix2pix.pth"),
        help="Ruta del checkpoint de prueba.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/smoke_test"),
        help="Directorio para guardar curva y cuadrícula de prueba.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cpu")

    run_tensor_smoke_test(device)
    run_mini_train_smoke_test(
        device=device,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        num_samples=args.samples,
        epochs=args.epochs,
    )
    print("\nCHECKPOINT 2 COMPLETADO SIN ERRORES.")


if __name__ == "__main__":
    main()
