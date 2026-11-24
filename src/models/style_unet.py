"""Experimento aditivo: Generador U-Net condicionado por estilo (Style Encoder + AdaIN).

Desacopla el contenido tipográfico (`x`, fuente base Lato) del estilo caligráfico
(`style_refs`, K=4 glifos de referencia de un mismo autor o fuente) para permitir
generación *Few-Shot* y mejor generalización *Zero-Shot* en trazos finos (stylus).

Arquitectura:
- `StyleEncoder`: CNN ligera SIN InstanceNorm (siguiendo MUNIT/FUNIT) para preservar
  explícitamente densidad de tinta, grosor de trazo, inclinación y curvatura.
- `AdaIN2d`: Adaptive Instance Normalization modulada por `z_style` (inicializada en
  identidad `gamma=0, beta=0` para estabilidad desde la época 1).
- `StyleUNetGenerator`: 7 niveles encoder de contenido + fusión de estilo en bottleneck
  + 6 bloques decoder modulados por `AdaIN2d` + salida `Tanh` (`1×128×128` en `[-1, 1]`).
- `StyleUNetLoss`: pérdida adversarial PatchGAN + `L1` píxel + consistencia de estilo
  en el espacio de embeddings de `StyleEncoder`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pix2pix import PatchGANDiscriminator, Pix2PixLoss, UNetDownBlock, init_pix2pix_weights


class AdaIN2d(nn.Module):
    """Adaptive Instance Normalization modulada por un vector de estilo `z_style`."""

    def __init__(self, num_features: int, style_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.style_proj = nn.Linear(style_dim, num_features * 2)
        # Inicialización en identidad: (1 + 0) * norm(x) + 0
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)

    def forward(self, x: torch.Tensor, z_style: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        gamma_beta = self.style_proj(z_style).view(b, 2 * c, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return (1.0 + gamma) * x_norm + beta


class StyleEncoder(nn.Module):
    """Codificador CNN de estilo (sin InstanceNorm para preservar grosor y densidad de tinta).

    Entrada:
    - `style_refs`: `[B, K, 1, 128, 128]` o `[B, 1, 128, 128]` en `[-1, 1]`.
    Salida:
    - `z_style`: `[B, style_dim]` invariante al orden de las K referencias (mean pooling).
    """

    def __init__(
        self,
        in_channels: int = 1,
        style_dim: int = 128,
        base_features: int = 32,
    ) -> None:
        super().__init__()
        f = base_features
        self.conv_net = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(f * 8, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        init_pix2pix_weights(self)

    def forward(self, style_refs: torch.Tensor) -> torch.Tensor:
        if style_refs.ndim == 4:
            # [B, 1, H, W] -> [B, 1, 1, H, W]
            style_refs = style_refs.unsqueeze(1)
        if style_refs.ndim != 5:
            raise ValueError(
                f"Se esperaba style_refs 4D o 5D, se recibió shape={tuple(style_refs.shape)}"
            )

        b, k, c, h, w = style_refs.shape
        flat = style_refs.reshape(b * k, c, h, w)
        feats = self.conv_net(flat).flatten(1)  # [B*K, f*8]
        feats = feats.view(b, k, -1).mean(dim=1)  # pooling invariante a permutación
        return self.fc(feats)


class StyleUNetUpBlock(nn.Module):
    """Bloque decoder con ConvTranspose2d + AdaIN(z_style) [+ Dropout] + ReLU + concat skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        use_dropout: bool = False,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.adain = AdaIN2d(out_channels, style_dim)
        self.dropout = nn.Dropout(dropout_p) if use_dropout else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        z_style: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        x = self.adain(x, z_style)
        x = self.dropout(x)
        x = self.act(x)
        return torch.cat([x, skip], dim=1)


class StyleUNetGenerator(nn.Module):
    """Generador U-Net 7 niveles con Content Encoder + Style Encoder + AdaIN Decoder."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        style_dim: int = 128,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        f = base_features

        # Rama de estilo
        self.style_encoder = StyleEncoder(
            in_channels=in_channels,
            style_dim=style_dim,
            base_features=32,
        )

        # Rama de contenido (7 bloques: 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        self.down1 = UNetDownBlock(in_channels, f, use_batchnorm=False)       # [B, 64, 64, 64]
        self.down2 = UNetDownBlock(f, f * 2, use_batchnorm=True)              # [B, 128, 32, 32]
        self.down3 = UNetDownBlock(f * 2, f * 4, use_batchnorm=True)          # [B, 256, 16, 16]
        self.down4 = UNetDownBlock(f * 4, f * 8, use_batchnorm=True)          # [B, 512, 8, 8]
        self.down5 = UNetDownBlock(f * 8, f * 8, use_batchnorm=True)          # [B, 512, 4, 4]
        self.down6 = UNetDownBlock(f * 8, f * 8, use_batchnorm=True)          # [B, 512, 2, 2]
        self.down7 = UNetDownBlock(
            f * 8, f * 8, use_batchnorm=False, activation="relu"
        )                                                                     # [B, 512, 1, 1]

        # Fusión de estilo en el bottleneck
        self.bottleneck_style = nn.Linear(style_dim, f * 8)

        # Decoder modulado por AdaIN (1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128)
        self.up1 = StyleUNetUpBlock(f * 8, f * 8, style_dim, use_dropout=True)
        self.up2 = StyleUNetUpBlock(f * 16, f * 8, style_dim, use_dropout=True)
        self.up3 = StyleUNetUpBlock(f * 16, f * 8, style_dim, use_dropout=True)
        self.up4 = StyleUNetUpBlock(f * 16, f * 4, style_dim, use_dropout=False)
        self.up5 = StyleUNetUpBlock(f * 8, f * 2, style_dim, use_dropout=False)
        self.up6 = StyleUNetUpBlock(f * 4, f, style_dim, use_dropout=False)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(
                f * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

        # Inicializar capas Conv/BN siguiendo Pix2Pix y re-inicializar AdaIN en identidad
        init_pix2pix_weights(self)
        for m in self.modules():
            if isinstance(m, AdaIN2d):
                nn.init.zeros_(m.style_proj.weight)
                nn.init.zeros_(m.style_proj.bias)

    def encode_style(self, style_refs: torch.Tensor) -> torch.Tensor:
        """Extrae el vector de estilo `[B, style_dim]` a partir de K glifos de referencia."""
        return self.style_encoder(style_refs)

    def forward(
        self,
        x: torch.Tensor,
        style_refs: torch.Tensor | None = None,
        z_style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_style is None:
            if style_refs is None:
                raise ValueError("Debes proporcionar `style_refs` o `z_style`.")
            z_style = self.encode_style(style_refs)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottleneck = self.down7(d6) + self.bottleneck_style(z_style)[:, :, None, None]

        u1 = self.up1(bottleneck, d6, z_style)
        u2 = self.up2(u1, d5, z_style)
        u3 = self.up3(u2, d4, z_style)
        u4 = self.up4(u3, d3, z_style)
        u5 = self.up5(u4, d2, z_style)
        u6 = self.up6(u5, d1, z_style)
        return self.up7(u6)


class StyleUNetLoss(nn.Module):
    """Pérdida combinada: PatchGAN BCE + λ_l1·L1 + λ_style·StyleConsistency + λ_ink·InkDensity."""

    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_style: float = 5.0,
        lambda_ink: float = 15.0,
    ) -> None:
        super().__init__()
        self.base = Pix2PixLoss(lambda_l1=lambda_l1)
        self.lambda_style = lambda_style
        self.lambda_ink = lambda_ink

    def discriminator_loss(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.base.discriminator_loss(pred_real, pred_fake)

    def generator_loss(
        self,
        pred_fake: torch.Tensor,
        fake_y: torch.Tensor,
        real_y: torch.Tensor,
        z_fake: torch.Tensor,
        z_real: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_g_base, metrics = self.base.generator_loss(pred_fake, fake_y, real_y)

        # Consistencia en espacio de embeddings de estilo (coseno + L1 normalizado)
        z_fake_n = F.normalize(z_fake, dim=1)
        z_real_n = F.normalize(z_real.detach(), dim=1)
        loss_style = F.l1_loss(z_fake_n, z_real_n)

        # Igualación de densidad media de tinta (evita colapso a trazos vacíos o demasiado gruesos)
        ink_fake = ((1.0 - fake_y) * 0.5).mean(dim=(1, 2, 3))
        ink_real = ((1.0 - real_y) * 0.5).mean(dim=(1, 2, 3))
        loss_ink = F.l1_loss(ink_fake, ink_real)

        loss_total = loss_g_base + self.lambda_style * loss_style + self.lambda_ink * loss_ink
        metrics["g_style"] = float(loss_style.item())
        metrics["g_ink"] = float(loss_ink.item())
        metrics["g_loss"] = float(loss_total.item())
        return loss_total, metrics


def _self_test() -> None:
    gen = StyleUNetGenerator(style_dim=128)
    disc = PatchGANDiscriminator()
    crit = StyleUNetLoss()

    x = torch.randn(2, 1, 128, 128)
    y = torch.randn(2, 1, 128, 128)
    refs = torch.randn(2, 4, 1, 128, 128)

    y_hat = gen(x, refs)
    z_fake = gen.encode_style(y_hat)
    z_real = gen.encode_style(y)
    d_fake = disc(x, y_hat)
    loss_g, metrics = crit.generator_loss(d_fake, y_hat, y, z_fake, z_real)

    n_params = sum(p.numel() for p in gen.parameters()) / 1e6
    assert y_hat.shape == (2, 1, 128, 128), f"Shape inesperado: {y_hat.shape}"
    assert d_fake.shape == (2, 1, 14, 14), f"Shape disc inesperado: {d_fake.shape}"
    print(f"[OK] StyleUNetGenerator ({n_params:.2f}M params) | y_hat={tuple(y_hat.shape)} | metrics={metrics}")


if __name__ == "__main__":
    _self_test()
