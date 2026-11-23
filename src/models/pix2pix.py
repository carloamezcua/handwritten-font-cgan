"""Arquitectura Pix2Pix (cGAN) para generación de glifos manuscritos 1x128x128.

Componentes:
- `UNetGenerator`: 7 bloques encoder + 7 bloques decoder con skip connections y salida Tanh.
  Entrada `[B, 1, 128, 128]` -> Salida `[B, 1, 128, 128]` en `[-1, 1]`.
- `PatchGANDiscriminator`: entrada concatenada `[x, y]` (`[B, 2, 128, 128]`),
  salida mapa de logits `[B, 1, 14, 14]`.
- `Pix2PixLoss`: `BCEWithLogitsLoss` con one-sided label smoothing (`0.9` real / `0.0` falso)
  + pérdida de reconstrucción `L1Loss` con `lambda_l1 = 100.0`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def init_pix2pix_weights(model: nn.Module, init_gain: float = 0.02) -> None:
    """Inicialización estándar de Pix2Pix: Normal(0, 0.02) para Conv/ConvTranspose y BN."""

    def _init_fn(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            "Conv" in classname or "Linear" in classname
        ):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_fn)


class UNetDownBlock(nn.Module):
    """Bloque de bajada (encoder) de la U-Net: Conv(k=4, s=2, p=1) [+ BN] + LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Activación no soportada: {activation}")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Bloque de subida (decoder) de la U-Net: ConvTranspose(k=4, s=2, p=1) + BN [+ Dropout] + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """Generador U-Net de 7 niveles encoder + 7 niveles decoder para imágenes 1x128x128."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
    ) -> None:
        super().__init__()
        f = base_features

        # 7 bloques encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.down1 = UNetDownBlock(in_channels, f, use_batchnorm=False)      # [B, 64, 64, 64]
        self.down2 = UNetDownBlock(f, f * 2, use_batchnorm=True)             # [B, 128, 32, 32]
        self.down3 = UNetDownBlock(f * 2, f * 4, use_batchnorm=True)         # [B, 256, 16, 16]
        self.down4 = UNetDownBlock(f * 4, f * 8, use_batchnorm=True)         # [B, 512, 8, 8]
        self.down5 = UNetDownBlock(f * 8, f * 8, use_batchnorm=True)         # [B, 512, 4, 4]
        self.down6 = UNetDownBlock(f * 8, f * 8, use_batchnorm=True)         # [B, 512, 2, 2]
        self.down7 = UNetDownBlock(
            f * 8, f * 8, use_batchnorm=False, activation="relu"
        )                                                                    # [B, 512, 1, 1]

        # 7 bloques decoder con skip connections: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.up1 = UNetUpBlock(f * 8, f * 8, use_dropout=True)               # -> cat d6: [B, 1024, 2, 2]
        self.up2 = UNetUpBlock(f * 16, f * 8, use_dropout=True)              # -> cat d5: [B, 1024, 4, 4]
        self.up3 = UNetUpBlock(f * 16, f * 8, use_dropout=True)              # -> cat d4: [B, 1024, 8, 8]
        self.up4 = UNetUpBlock(f * 16, f * 4, use_dropout=False)             # -> cat d3: [B, 512, 16, 16]
        self.up5 = UNetUpBlock(f * 8, f * 2, use_dropout=False)              # -> cat d2: [B, 256, 32, 32]
        self.up6 = UNetUpBlock(f * 4, f, use_dropout=False)                  # -> cat d1: [B, 128, 64, 64]
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(
                f * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )                                                                    # [B, 1, 128, 128]

        init_pix2pix_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleneck = self.down7(d6)

        u1 = self.up1(bottleneck, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.up7(u6)


class PatchGANDiscriminator(nn.Module):
    """Discriminador condicional PatchGAN (70x70) para pares `[x, y]` 1x128x128 -> logits 14x14."""

    def __init__(
        self,
        in_channels: int = 2,
        base_features: int = 64,
    ) -> None:
        super().__init__()
        f = base_features

        self.net = nn.Sequential(
            # [B, 2, 128, 128] -> [B, 64, 64, 64]
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 64, 64, 64] -> [B, 128, 32, 32]
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 128, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 256, 16, 16] -> [B, 512, 15, 15]
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 512, 15, 15] -> [B, 1, 14, 14] (logits crudos)
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        init_pix2pix_weights(self)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=1)
        return self.net(xy)


class Pix2PixLoss(nn.Module):
    """Pérdidas de Pix2Pix: BCEWithLogitsLoss (one-sided label smoothing) + L1Loss."""

    def __init__(
        self,
        lambda_l1: float = 100.0,
        real_label: float = 0.9,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.real_label = real_label
        self.fake_label = fake_label
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_real = torch.full_like(pred_real, self.real_label)
        target_fake = torch.full_like(pred_fake, self.fake_label)
        loss_real = self.bce(pred_real, target_real)
        loss_fake = self.bce(pred_fake, target_fake)
        loss_d = 0.5 * (loss_real + loss_fake)
        metrics = {
            "d_loss": float(loss_d.item()),
            "d_real": float(loss_real.item()),
            "d_fake": float(loss_fake.item()),
        }
        return loss_d, metrics

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
        fake_y: torch.Tensor,
        real_y: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_real = torch.ones_like(pred_fake)
        loss_adv = self.bce(pred_fake, target_real)
        loss_l1 = self.l1(fake_y, real_y)
        loss_g = loss_adv + self.lambda_l1 * loss_l1
        metrics = {
            "g_loss": float(loss_g.item()),
            "g_adv": float(loss_adv.item()),
            "g_l1": float(loss_l1.item()),
        }
        return loss_g, metrics
