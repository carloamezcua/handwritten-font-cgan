"""Utilidades compartidas para renderizado y normalización de glifos 1x128x128.

Mantiene un contrato visual único (fondo blanco 255, trazo negro 0, línea base
consistente en `baseline_ratio = 0.72`) tanto para el dataset sintético como
para el emparejamiento de muestras reales, pruebas OOD e inferencia.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from src.config import IMG_SIZE

BASELINE_RATIO = 0.72


def get_default_base_font_path(fonts_dir: Path = Path("data/fonts")) -> Path:
    """Devuelve la ruta de la fuente base limpia (prefiere Lato local usada en entrenamiento)."""
    candidates = [
        fonts_dir / "base" / "Lato-Regular.ttf",
        fonts_dir / "Lato-Regular.ttf",
        fonts_dir / "base" / "Roboto-Regular.ttf",
        fonts_dir / "Roboto-Regular.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontró ninguna fuente base .ttf. Ejecuta "
        "`python src/data/03_generar_sinteticos.py --download-fonts` o indica --base-font."
    )


@lru_cache(maxsize=128)
def _load_cmap(font_path_str: str) -> set[int]:
    """Carga y cachea el mapa Unicode (cmap) de una fuente TrueType/OpenType."""
    with TTFont(font_path_str) as tt:
        cmap = tt.getBestCmap() or {}
    return set(cmap.keys())


def font_supports_char(font_path: Path, ch: str) -> bool:
    """Verifica si la fuente contiene realmente el glifo para `ch` (evita cajas vacías)."""
    return ord(ch) in _load_cmap(str(font_path.resolve()))


def font_supports_charset(font_path: Path, charset: list[str]) -> tuple[bool, list[str]]:
    """Verifica cobertura completa del charset; devuelve (ok, faltantes)."""
    cmap = _load_cmap(str(font_path.resolve()))
    missing = [ch for ch in charset if ord(ch) not in cmap]
    return len(missing) == 0, missing


@lru_cache(maxsize=256)
def _calibrated_font_size(font_path_str: str, img_size: int) -> int:
    """Calcula el tamaño de fuente óptimo para que 'Hg' ocupe ~68% de `img_size`."""
    probe_size = int(img_size * 0.75)
    font = ImageFont.truetype(font_path_str, size=probe_size)
    dummy = Image.new("L", (img_size * 2, img_size * 2), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((img_size, img_size), "Hg", font=font, anchor="ls")
    ref_h = max(1, bbox[3] - bbox[1])
    target_h = img_size * 0.68
    scaled = int(round(probe_size * (target_h / ref_h)))
    return max(16, min(scaled, int(img_size * 1.15)))


def render_glyph(
    ch: str,
    font_path: Path,
    img_size: int = IMG_SIZE,
    scale: float = 1.0,
    shift_x: int = 0,
    shift_y: int = 0,
    baseline_ratio: float = BASELINE_RATIO,
) -> np.ndarray:
    """Renderiza un carácter en escala de grises `img_size x img_size` (fondo 255, tinta 0).

    Respeta la proporción tipográfica natural (p. ej. '.' y '-' no se inflan a toda la caja)
    y solo reescala si el glifo desbordaría el margen de seguridad.
    """
    font_path_str = str(font_path.resolve())
    base_size = _calibrated_font_size(font_path_str, img_size)
    font_size = max(12, int(round(base_size * scale)))
    font = ImageFont.truetype(font_path_str, size=font_size)

    # Lienzo amplio para medir sin recorte previo
    work_size = img_size * 2
    canvas = Image.new("L", (work_size, work_size), color=255)
    draw = ImageDraw.Draw(canvas)

    cx = work_size // 2
    by = work_size // 2
    draw.text((cx, by), ch, fill=0, font=font, anchor="ms")

    arr = np.array(canvas, dtype=np.uint8)
    ys, xs = np.where(arr < 245)

    # Si no hubo tinta (fallback de seguridad), devolver blanco
    if len(xs) == 0:
        return np.full((img_size, img_size), 255, dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    glyph_w = x_max - x_min + 1
    glyph_h = y_max - y_min + 1

    # Posición objetivo del lienzo final respetando la línea base
    target_baseline_y = int(round(img_size * baseline_ratio))
    # Desplazamiento vertical relativo de la tinta respecto a la línea base original `by`
    rel_top = y_min - by
    rel_bottom = y_max - by

    margin = 8
    max_span = img_size - 2 * margin

    # Solo encoger si el glifo desborda el lienzo final
    out_top = target_baseline_y + rel_top
    out_bottom = target_baseline_y + rel_bottom
    needs_shrink = (
        glyph_w > max_span
        or glyph_h > max_span
        or out_top < margin
        or out_bottom > img_size - margin
    )

    crop = canvas.crop((x_min, y_min, x_max + 1, y_max + 1))

    if needs_shrink:
        fit_scale = min(max_span / glyph_w, max_span / glyph_h)
        new_w = max(1, int(round(glyph_w * fit_scale)))
        new_h = max(1, int(round(glyph_h * fit_scale)))
        crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
        paste_x = (img_size - new_w) // 2 + shift_x
        paste_y = (img_size - new_h) // 2 + shift_y
    else:
        paste_x = (img_size - glyph_w) // 2 + shift_x
        paste_y = target_baseline_y + rel_top + shift_y

    # Asegurar que no se recorte al aplicar shift_x / shift_y
    paste_x = max(2, min(paste_x, img_size - crop.width - 2))
    paste_y = max(2, min(paste_y, img_size - crop.height - 2))

    out = Image.new("L", (img_size, img_size), color=255)
    out.paste(crop, (paste_x, paste_y))
    return np.array(out, dtype=np.uint8)
