"""Genera >= 60 imágenes fuera de distribución (OOD) en `data/ood/` para calibración y rechazo.

Categorías OOD generadas (10+ muestras por categoría):
1. `ruido_puro`: ruido gaussiano y sal-pimienta sin carácter válido.
2. `garabatos`: trazos aleatorios y líneas cruzadas no alfanuméricas.
3. `simbolos_ineditos`: caracteres fuera del charset de 75 símbolos (`@ # $ % ñ Ñ ¿ ¡ *`).
4. `rotacion_extrema`: glifos rotados entre ±35° y ±65°.
5. `desenfoque_corte`: glifos fuertemente borrosos o recortados a la mitad.
6. `sombra_bajo_contraste`: degradados fuertes de sombra y bajo contraste.

Uso:
    python -m src.data.04_generar_ood --output-dir data/ood
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.config import CHARSET, IMG_SIZE, SEED
from src.data.glyph_render import (
    font_supports_char,
    get_default_base_font_path,
    render_glyph,
)
from src.utils import seed_everything

UNSEEN_OOD_SYMBOLS = list("@#$%ñÑ¿¡*")


def _make_noise(rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        img = np_rng.normal(loc=180.0, scale=55.0, size=(IMG_SIZE, IMG_SIZE))
        return np.clip(img, 0, 255).astype(np.uint8)
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    mask = np_rng.random((IMG_SIZE, IMG_SIZE))
    img[mask < 0.12] = 0
    img[(mask >= 0.12) & (mask < 0.20)] = 120
    return img


def _make_scribble(rng: random.Random) -> np.ndarray:
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(canvas)
    n_strokes = rng.randint(4, 9)
    for _ in range(n_strokes):
        pts = [
            (rng.randint(8, IMG_SIZE - 8), rng.randint(8, IMG_SIZE - 8))
            for _ in range(rng.randint(3, 6))
        ]
        draw.line(pts, fill=rng.randint(0, 40), width=rng.randint(2, 6))
    return np.array(canvas, dtype=np.uint8)


def _make_extreme_rotation(rng: random.Random, base_font: Path) -> tuple[np.ndarray, str]:
    ch = rng.choice(CHARSET)
    img = render_glyph(ch, base_font, img_size=IMG_SIZE)
    angle = rng.choice([-1, 1]) * rng.uniform(35.0, 65.0)
    center = (IMG_SIZE / 2.0, IMG_SIZE / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, rot_mat, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    return rotated, ch


def _make_blur_or_crop(rng: random.Random, base_font: Path) -> tuple[np.ndarray, str]:
    ch = rng.choice(CHARSET)
    img = render_glyph(ch, base_font, img_size=IMG_SIZE)
    if rng.random() < 0.5:
        k = rng.choice([11, 15, 19])
        out = cv2.GaussianBlur(img, (k, k), 0)
    else:
        out = img.copy()
        # Borrar una mitad del glifo (simulando recorte defectuoso)
        if rng.random() < 0.5:
            out[:, : IMG_SIZE // 2] = 255
        else:
            out[IMG_SIZE // 2 :, :] = 255
    return out, ch


def _make_heavy_shadow(rng: random.Random, base_font: Path) -> tuple[np.ndarray, str]:
    ch = rng.choice(CHARSET)
    img = render_glyph(ch, base_font, img_size=IMG_SIZE).astype(np.float32)
    grad = np.linspace(0.25, 1.0, IMG_SIZE, dtype=np.float32)
    if rng.random() < 0.5:
        shadow = np.tile(grad[None, :], (IMG_SIZE, 1))
    else:
        shadow = np.tile(grad[:, None], (1, IMG_SIZE))
    # Reducir contraste y oscurecer con degradado fuerte
    darkened = img * shadow * rng.uniform(0.55, 0.8) + rng.uniform(10.0, 35.0)
    return np.clip(darkened, 0, 255).astype(np.uint8), ch


def generate_ood_dataset(
    output_dir: Path,
    per_category: int = 12,
    base_font: Path | None = None,
    seed: int = SEED,
) -> Path:
    """Genera >= 60 imágenes OOD y guarda `metadata.csv`."""
    seed_everything(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if base_font is None:
        base_font = get_default_base_font_path()

    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    idx = 0

    def _save(img: np.ndarray, category: str, source_char: str) -> None:
        nonlocal idx
        fname = f"ood_{idx:04d}_{category}.png"
        rel_path = Path("images") / fname
        Image.fromarray(img, mode="L").save(output_dir / rel_path)
        rows.append(
            {
                "sample_id": idx,
                "category": category,
                "source_char": source_char,
                "is_ood": 1,
                "image_path": rel_path.as_posix(),
            }
        )
        idx += 1

    # 1. Ruido puro
    for _ in range(per_category):
        _save(_make_noise(rng, np_rng), "ruido_puro", "")

    # 2. Garabatos aleatorios
    for _ in range(per_category):
        _save(_make_scribble(rng), "garabatos", "")

    # 3. Símbolos inéditos fuera del charset de 75 clases
    supported_unseen = [ch for ch in UNSEEN_OOD_SYMBOLS if font_supports_char(base_font, ch)]
    for i in range(per_category):
        ch = supported_unseen[i % len(supported_unseen)]
        img = render_glyph(
            ch,
            base_font,
            img_size=IMG_SIZE,
            scale=rng.uniform(0.92, 1.08),
            shift_x=rng.randint(-4, 4),
            shift_y=rng.randint(-4, 4),
        )
        _save(img, "simbolos_ineditos", ch)

    # 4. Rotación extrema
    for _ in range(per_category):
        img, ch = _make_extreme_rotation(rng, base_font)
        _save(img, "rotacion_extrema", ch)

    # 5. Desenfoque fuerte o glifo cortado
    for _ in range(per_category):
        img, ch = _make_blur_or_crop(rng, base_font)
        _save(img, "desenfoque_corte", ch)

    # 6. Sombra fuerte / bajo contraste
    for _ in range(per_category):
        img, ch = _make_heavy_shadow(rng, base_font)
        _save(img, "sombra_bajo_contraste", ch)

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "source_char", "is_ood", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset OOD generado en '{output_dir}': {len(rows)} imágenes ({per_category} por categoría).")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera imágenes OOD (>= 50) para evaluación.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ood"))
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--base-font", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    generate_ood_dataset(
        output_dir=args.output_dir,
        per_category=args.per_category,
        base_font=args.base_font,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
