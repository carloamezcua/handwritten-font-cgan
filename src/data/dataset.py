"""Datasets y DataLoaders unificados de PyTorch para Pix2Pix y clasificadores jueces.

Contrato de tensores:
- Imágenes en escala de grises `1 x 128 x 128`, float32 en rango `[-1.0, 1.0]`.
- Aumentos solo en `train`: pequeñas transformaciones afines (rotación ±4°,
  traslación ±4 %, escala 0.94–1.06) rellenando con fondo blanco (`255`).
  NUNCA se aplican flips horizontales o verticales porque alteran la identidad
  de los caracteres (p. ej. `b` <-> `d`, `p` <-> `q`, `6` <-> `9`).
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.config import IMG_SIZE


class PairGentleAffine:
    """Aplica exactamente la misma transformación afín suave a un par (x, y) o a y."""

    def __init__(
        self,
        degrees: float = 4.0,
        translate: tuple[float, float] = (0.04, 0.04),
        scale: tuple[float, float] = (0.94, 1.06),
        p: float = 0.5,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(
        self, x_img: Image.Image, y_img: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if torch.rand(1).item() > self.p:
            return x_img, y_img
        params = transforms.RandomAffine.get_params(
            degrees=(-self.degrees, self.degrees),
            translate=self.translate,
            scale_ranges=self.scale,
            shears=None,
            img_size=[x_img.width, x_img.height],
        )
        x_aug = TF.affine(x_img, *params, fill=255)
        y_aug = TF.affine(y_img, *params, fill=255)
        return x_aug, y_aug


def _pil_to_normalized_tensor(img: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Convierte PIL 'L' a tensor `[1, H, W]` en `[-1.0, 1.0]`."""
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img)  # [1, H, W] en [0, 1]
    return tensor * 2.0 - 1.0  # [-1, 1]


class GlyphPairDataset(Dataset):
    """Dataset emparejado `(x, y, class_id, char)` para Pix2Pix (sintético o real)."""

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        split_column: str = "split",
        augment: bool = False,
        img_size: int = IMG_SIZE,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.pair_aug = PairGentleAffine() if self.augment else None

        metadata_path = self.root_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontró '{metadata_path}'.")

        self.samples: list[dict] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row[split_column] == split:
                    self.samples.append(row)

        if not self.samples:
            raise ValueError(
                f"No hay muestras para {split_column}='{split}' en '{metadata_path}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.samples[idx]
        x_img = Image.open(self.root_dir / row["x_path"]).convert("L")
        y_img = Image.open(self.root_dir / row["y_path"]).convert("L")

        if self.pair_aug is not None:
            x_img, y_img = self.pair_aug(x_img, y_img)

        x_tensor = _pil_to_normalized_tensor(x_img, self.img_size)
        y_tensor = _pil_to_normalized_tensor(y_img, self.img_size)
        return x_tensor, y_tensor, int(row["class_id"]), row["char"]


class GlyphJudgeDataset(Dataset):
    """Dataset de clasificación `(y, class_id, char)` para entrenar/evaluar jueces CNN."""

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        split_column: str = "judge_split",
        augment: bool = False,
        img_size: int = IMG_SIZE,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.affine = (
            transforms.RandomAffine(
                degrees=6.0,
                translate=(0.05, 0.05),
                scale=(0.92, 1.08),
                fill=255,
            )
            if self.augment
            else None
        )

        metadata_path = self.root_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontró '{metadata_path}'.")

        self.samples: list[dict] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row[split_column] == split:
                    self.samples.append(row)

        if not self.samples:
            raise ValueError(
                f"No hay muestras para {split_column}='{split}' en '{metadata_path}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.samples[idx]
        y_img = Image.open(self.root_dir / row["y_path"]).convert("L")
        if self.affine is not None:
            y_img = self.affine(y_img)
        y_tensor = _pil_to_normalized_tensor(y_img, self.img_size)
        return y_tensor, int(row["class_id"]), row["char"]


def build_pix2pix_loaders(
    root_dir: Path | str,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
) -> dict[str, DataLoader]:
    """Construye DataLoaders para `train`, `val` y `test_unseen`."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test_unseen"):
        ds = GlyphPairDataset(
            root_dir=root_dir,
            split=split,
            augment=(augment_train and split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
