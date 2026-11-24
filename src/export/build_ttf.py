"""Genera una fuente instalable `.ttf` a partir del Generador ONNX (`vtracer` + `fontTools`).

Flujo por carácter (75 símbolos de `CHARSET` + 9 inéditos `@ # $ % ñ Ñ ¿ ¡ *`):
1. Renderiza el glifo base `x` (`1×128×128`) con la misma fuente base del entrenamiento.
2. Genera el glifo manuscrito `ŷ` con `generator.onnx` (ONNX Runtime CPU).
3. Binariza suavemente en alta resolución (256×256) y vectoriza contornos con `vtracer`.
4. Convierte curvas cúbicas SVG a cuadráticas TrueType (`Cu2QuPen`), alinea la línea base
   y calcula anchos proporcionales (`hmtx`).
5. Empaqueta en un archivo `.ttf` válido con `fontTools.fontBuilder.FontBuilder` y genera
   una imagen de prueba (`--preview-out`) escribiendo una frase con la fuente recién creada.

Uso:
    python -m src.export.build_ttf --onnx-model checkpoints/generator.onnx --output results/HandwrittenCGAN.ttf
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import vtracer
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path
from PIL import Image, ImageDraw, ImageFont

from src.config import CHARSET, IMG_SIZE
from src.data.glyph_render import BASELINE_RATIO, get_default_base_font_path, render_glyph

EXTRA_UNSEEN_CHARS = list("@#$%ñÑ¿¡*")
UNITS_PER_EM = 1000
ASCENT = 800
DESCENT = -200
TRACE_SIZE = 256


def _is_dark_fill(fill: str | None) -> bool:
    """Devuelve True si el color SVG de relleno corresponde a tinta oscura (no fondo blanco)."""
    if not fill:
        return True
    f = fill.strip().lower()
    if f in {"none", "white", "#fff", "#ffffff", "rgb(255,255,255)"}:
        return False
    m = re.match(r"^#([0-9a-f]{6})$", f)
    if m:
        hex_val = m.group(1)
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)
        return (r + g + b) / 3.0 < 180.0
    return True


def _parse_translate(transform: str | None) -> tuple[float, float]:
    """Extrae `(tx, ty)` de un atributo SVG `translate(tx, ty)` si existe."""
    if not transform:
        return 0.0, 0.0
    m = re.search(r"translate\(\s*([+-]?\d*\.?\d+)(?:[\s,]+([+-]?\d*\.?\d+))?\s*\)", transform)
    if not m:
        return 0.0, 0.0
    tx = float(m.group(1))
    ty = float(m.group(2)) if m.group(2) is not None else 0.0
    return tx, ty


def _glyph_image_to_ttglyph(
    glyph_u8: np.ndarray,
    threshold: int = 175,
) -> tuple[object, int, int]:
    """Vectoriza un glifo `128x128` uint8 (fondo 255, trazo 0) y devuelve `(ttglyph, advance, lsb)`."""
    # Escalar 2x con suavizado para curvas limpias antes de binarizar
    up = cv2.resize(glyph_u8, (TRACE_SIZE, TRACE_SIZE), interpolation=cv2.INTER_CUBIC)
    binary = np.where(up < threshold, 0, 255).astype(np.uint8)

    ys, xs = np.where(binary == 0)
    if len(xs) < 10:
        # Glifo vacío: devolver glifo vacío con ancho estándar
        empty_pen = TTGlyphPen(None)
        return empty_pen.glyph(), 420, 0

    x_min_px = float(xs.min())
    x_max_px = float(xs.max())

    # Codificar PNG en memoria para vtracer
    buf = io.BytesIO()
    Image.fromarray(binary).convert("RGB").save(buf, format="PNG")
    svg_str = vtracer.convert_raw_image_to_svg(
        buf.getvalue(),
        img_format="png",
        colormode="binary",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=6,
        corner_threshold=60,
        length_threshold=4.0,
        splice_threshold=45,
        path_precision=3,
    )

    root = ET.fromstring(svg_str)
    scale = UNITS_PER_EM / float(TRACE_SIZE)
    baseline_px = BASELINE_RATIO * float(TRACE_SIZE)
    lsb = 55

    tt_pen = TTGlyphPen(None)
    cu2qu_pen = Cu2QuPen(tt_pen, max_err=1.5, reverse_direction=False)

    paths_drawn = 0
    for elem in root.iter():
        if not elem.tag.endswith("path"):
            continue
        d = elem.attrib.get("d", "").strip()
        if not d or not _is_dark_fill(elem.attrib.get("fill")):
            continue
        tx_svg, ty_svg = _parse_translate(elem.attrib.get("transform"))

        # Mapeo SVG (x_px, y_px) -> Font units (x_f, y_f):
        # x_f = (x_px + tx_svg - x_min_px) * scale + lsb
        # y_f = (baseline_px - (y_px + ty_svg)) * scale
        dx = (tx_svg - x_min_px) * scale + lsb
        dy = (baseline_px - ty_svg) * scale
        matrix = (scale, 0.0, 0.0, -scale, dx, dy)
        t_pen = TransformPen(cu2qu_pen, matrix)
        parse_path(d, t_pen)
        paths_drawn += 1

    if paths_drawn == 0:
        empty_pen = TTGlyphPen(None)
        return empty_pen.glyph(), 420, 0

    glyph_width_units = (x_max_px - x_min_px) * scale
    advance_width = max(int(round(glyph_width_units + 2 * lsb)), 240)
    return tt_pen.glyph(), advance_width, lsb


def _make_notdef_glyph():
    """Crea un glifo `.notdef` rectangular estándar."""
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((400, 50))
    pen.lineTo((400, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    return pen.glyph()


def render_font_preview(ttf_path: Path, preview_path: Path) -> None:
    """Renderiza una frase de prueba usando el archivo `.ttf` recién generado."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(ttf_path), size=56)

    lines = [
        "Hola Mundo 2026! (Prueba TTF)",
        "ABCDEFGHIJKLM NOPQRSTUVWXYZ",
        "abcdefghijklm nopqrstuvwxyz",
        "0123456789 .,!?()+-=/:;&",
        "Ineditos: @ # $ % ñ Ñ ¿ ¡ *",
    ]

    img = Image.new("RGB", (1150, 460), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 35
    for line in lines:
        draw.text((40, y), line, fill=(15, 15, 15), font=font)
        y += 80
    img.save(preview_path)


def build_ttf_from_onnx(
    onnx_model: Path,
    output_ttf: Path,
    font_name: str = "HandwrittenCGAN",
    base_font_path: Path | None = None,
    threshold: int = 175,
    preview_out: Path | None = None,
) -> Path:
    """Genera todos los glifos con `generator.onnx` y los empaqueta en un `.ttf` instalable."""
    if not onnx_model.exists():
        raise FileNotFoundError(f"No se encontró el modelo ONNX: '{onnx_model}'.")

    base_font = base_font_path or get_default_base_font_path()
    sess = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])

    chars = list(dict.fromkeys(CHARSET + EXTRA_UNSEEN_CHARS))

    glyph_order = [".notdef", "space"]
    cmap: dict[int, str] = {ord(" "): "space"}
    glyf_table: dict[str, object] = {
        ".notdef": _make_notdef_glyph(),
        "space": TTGlyphPen(None).glyph(),
    }
    hmtx_table: dict[str, tuple[int, int]] = {
        ".notdef": (500, 50),
        "space": (320, 0),
    }

    for char in chars:
        gname = f"uni{ord(char):04X}"
        x_u8 = render_glyph(char, base_font, img_size=IMG_SIZE)
        x_norm = (x_u8.astype(np.float32) / 127.5) - 1.0
        x_in = x_norm[None, None, :, :]

        y_hat = sess.run(["y_hat"], {"x": x_in})[0][0, 0]
        y_u8 = np.clip((y_hat + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

        ttglyph, advance, lsb = _glyph_image_to_ttglyph(y_u8, threshold=threshold)
        glyph_order.append(gname)
        cmap[ord(char)] = gname
        glyf_table[gname] = ttglyph
        hmtx_table[gname] = (advance, lsb)

    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf_table)
    fb.setupHorizontalMetrics(hmtx_table)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable(
        {
            "copyright": "Synthetic AI-generated font (Handwritten Font cGAN / Pix2Pix). Not for identity impersonation or forgery.",
            "familyName": font_name,
            "styleName": "Regular",
            "uniqueFontIdentifier": f"{font_name}-Regular-AI-Generated",
            "fullName": f"{font_name} Regular",
            "version": "Version 1.000",
            "psName": f"{font_name}-Regular",
            "manufacturer": "Handwritten Font cGAN Pipeline (Pix2Pix + vtracer + fontTools)",
            "designer": "Synthetic AI Output (HandwrittenFont-Pix2Pix-cGAN)",
            "description": (
                "AI-generated synthetic handwriting font produced glyph-by-glyph with Pix2Pix cGAN "
                "(1x128x128) and vectorized via vtracer/fontTools. Provenance marker: AI-SYNTHETIC."
            ),
            "licenseDescription": "MIT (code) / SIL OFL 1.1 (synthetic pretraining fonts). Consent required for personal style transfer.",
        }
    )
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        sTypoLineGap=100,
        usWinAscent=ASCENT + 50,
        usWinDescent=abs(DESCENT) + 50,
    )
    fb.setupPost()

    output_ttf.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(output_ttf))
    print(f"[OK] Fuente .ttf generada ({len(chars)} glifos): {output_ttf}")

    if preview_out is not None:
        render_font_preview(output_ttf, preview_out)
        print(f"[OK] Vista previa guardada en: {preview_out}")

    return output_ttf


def _select_medoid_glyph(images: list[np.ndarray]) -> np.ndarray:
    """Selecciona la repetición más consistente (medoide por distancia L1 entre trazos propios)."""
    if len(images) == 1:
        return images[0]
    arr = np.stack([im.astype(np.float32) / 255.0 for im in images], axis=0)
    dists = np.abs(arr[:, None, :, :] - arr[None, :, :, :]).mean(axis=(2, 3))
    best_idx = int(np.argmin(dists.sum(axis=1)))
    return images[best_idx]


def build_ttf_from_real_pairs(
    real_dir: Path,
    output_ttf: Path,
    font_name: str = "MiLetraStylus",
    threshold: int = 175,
    preview_out: Path | None = None,
) -> Path:
    """Construye un `.ttf` seleccionando el medoide (trazo más consistente) de tus muestras reales."""
    meta_path = real_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"No se encontró '{meta_path}'.")

    by_char: dict[str, list[Path]] = defaultdict(list)
    with meta_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_char[row["char"]].append(real_dir / row["y_path"])

    glyph_order = [".notdef", "space"]
    cmap: dict[int, str] = {ord(" "): "space"}
    glyf_table: dict[str, object] = {
        ".notdef": _make_notdef_glyph(),
        "space": TTGlyphPen(None).glyph(),
    }
    hmtx_table: dict[str, tuple[int, int]] = {
        ".notdef": (500, 50),
        "space": (320, 0),
    }

    built = 0
    for char in CHARSET:
        paths = by_char.get(char, [])
        if not paths:
            continue
        imgs = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in paths if p.exists()]
        if not imgs:
            continue
        best_u8 = _select_medoid_glyph(imgs)
        gname = f"uni{ord(char):04X}"
        ttglyph, advance, lsb = _glyph_image_to_ttglyph(best_u8, threshold=threshold)
        glyph_order.append(gname)
        cmap[ord(char)] = gname
        glyf_table[gname] = ttglyph
        hmtx_table[gname] = (advance, lsb)
        built += 1

    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf_table)
    fb.setupHorizontalMetrics(hmtx_table)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable(
        {
            "copyright": "Digitized personal handwriting (Handwritten Font cGAN pipeline). Authorized personal use only.",
            "familyName": font_name,
            "styleName": "Regular",
            "uniqueFontIdentifier": f"{font_name}-Regular-Stylus-Medoid",
            "fullName": f"{font_name} Regular",
            "version": "Version 1.000",
            "psName": f"{font_name}-Regular",
            "manufacturer": "Handwritten Font cGAN Pipeline (L1-Medoid + vtracer + fontTools)",
            "designer": "Author Personal Handwriting (Captured via Stylus)",
            "description": (
                "Personal handwriting font compiled via L1-medoid selection over authorized stylus "
                "glyph captures (1x128x128) and vectorized via vtracer/fontTools. Provenance marker: PERSONAL-STYLUS."
            ),
            "licenseDescription": "Authorized personal handwriting capture; not for third-party impersonation.",
        }
    )
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        sTypoLineGap=100,
        usWinAscent=ASCENT + 50,
        usWinDescent=abs(DESCENT) + 50,
    )
    fb.setupPost()

    output_ttf.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(output_ttf))
    print(f"[OK] Fuente .ttf personal (medoide real) generada ({built} glifos): {output_ttf}")

    if preview_out is not None:
        render_font_preview(output_ttf, preview_out)
        print(f"[OK] Vista previa guardada en: {preview_out}")

    return output_ttf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectoriza glifos desde generator.onnx o desde data/real_pairs y construye un .ttf."
    )
    parser.add_argument(
        "--from-real-dir",
        type=Path,
        default=None,
        help="Si se indica (ej. data/real_pairs), compila el .ttf con el medoide de tus trazos reales.",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("checkpoints/generator.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/HandwrittenCGAN_Pretrain.ttf"),
    )
    parser.add_argument("--font-name", type=str, default="HandwrittenCGAN")
    parser.add_argument("--threshold", type=int, default=175)
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=Path("results/font_preview.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_real_dir is not None:
        build_ttf_from_real_pairs(
            real_dir=args.from_real_dir,
            output_ttf=args.output,
            font_name=args.font_name,
            threshold=args.threshold,
            preview_out=args.preview_out,
        )
    else:
        build_ttf_from_onnx(
            onnx_model=args.onnx_model,
            output_ttf=args.output,
            font_name=args.font_name,
            threshold=args.threshold,
            preview_out=args.preview_out,
        )


if __name__ == "__main__":
    main()
