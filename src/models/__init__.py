"""Modelos del proyecto: Pix2Pix (U-Net + PatchGAN) y jueces (ResNet-18, EfficientNet-B0)."""

from src.models.judges import (
    EfficientNetB0Judge,
    ResNet18Judge,
    build_judge,
    freeze_model,
)
from src.models.pix2pix import (
    PatchGANDiscriminator,
    Pix2PixLoss,
    UNetGenerator,
    init_pix2pix_weights,
)

__all__ = [
    "UNetGenerator",
    "PatchGANDiscriminator",
    "Pix2PixLoss",
    "init_pix2pix_weights",
    "ResNet18Judge",
    "EfficientNetB0Judge",
    "build_judge",
    "freeze_model",
]
