"""Modelos clasificadores jueces (ResNet-18 y EfficientNet-B0) para evaluación de glifos.

Ambos modelos están adaptados a:
- Entrada de 1 canal (`[B, 1, 128, 128]`).
- Salida de `len(CHARSET)` = 75 clases.
- Método `extract_features(x)` que devuelve los embeddings de la penúltima capa
  para el cálculo de **FID de dominio** (`512` dims en ResNet-18, `1280` dims en EfficientNet-B0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from src.config import CHARSET


def _adapt_first_conv_to_grayscale(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """Convierte una capa Conv2d de 3 canales a 1 canal preservando pesos promedio si es preentrenada."""
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    return new_conv


class ResNet18Judge(nn.Module):
    """Clasificador ResNet-18 adaptado a 1 canal y 75 clases, con extracción de embeddings (512-d)."""

    def __init__(
        self,
        num_classes: int = len(CHARSET),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = _adapt_first_conv_to_grayscale(backbone.conv1, pretrained=pretrained)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Linear(self.feature_dim, num_classes)
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve embeddings de la penúltima capa con forma `[B, 512]`."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)

        x = b.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.backbone.fc(feats)


class EfficientNetB0Judge(nn.Module):
    """Clasificador EfficientNet-B0 adaptado a 1 canal y 75 clases, con extracción de embeddings (1280-d)."""

    def __init__(
        self,
        num_classes: int = len(CHARSET),
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        backbone.features[0][0] = _adapt_first_conv_to_grayscale(
            backbone.features[0][0], pretrained=pretrained
        )
        in_features = backbone.classifier[1].in_features  # 1280
        self.feature_dim = in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve embeddings de la penúltima capa con forma `[B, 1280]`."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.backbone.classifier(feats)


def build_judge(
    name: str,
    num_classes: int = len(CHARSET),
    pretrained: bool = False,
) -> nn.Module:
    """Fábrica de clasificadores jueces por nombre (`resnet18` o `efficientnet_b0`)."""
    key = name.strip().lower().replace("-", "_")
    if key == "resnet18":
        return ResNet18Judge(num_classes=num_classes, pretrained=pretrained)
    if key in {"efficientnet_b0", "efficientnetb0"}:
        return EfficientNetB0Judge(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Juez no reconocido: '{name}'. Usa 'resnet18' o 'efficientnet_b0'."
    )


def freeze_model(model: nn.Module) -> nn.Module:
    """Congela parámetros y coloca el modelo en modo evaluación (`eval()`)."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
