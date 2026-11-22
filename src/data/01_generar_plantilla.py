"""Genera plantillas imprimibles con marcadores de esquina para captura manuscrita.

Cada hoja (tamaño Carta a 300 DPI: 2550x3300 px) incluye:
- 4 marcadores negros cuadrados en las esquinas para corrección de perspectiva OpenCV.
- Cuadrícula calibrada con etiqueta de carácter arriba del recuadro y línea base tenue.
- Archivo `template_layout.json` con coordenadas exactas y partición del charset.

Uso:
    python src/data/01_generar_plantilla.py --output-dir data/templates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Charset de 75 símbolos
from src.config import CHARSET, TEST_UNSEEN_CHARS

# Geometría de página tamaño Carta a 300 DPI
PAGE_W = 2550
PAGE_H = 3300
MARKER_SIZE = 110
MARKER_MARGIN = 110
COLS = 6
ROWS = 8
CELLS_PER_PAGE = COLS * ROWS


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Carga una fuente sans-serif del sistema o usa la fuente por defecto de Pillow."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def build_sample_list(repetitions: int = 4) -> list[dict]:
    """Construye la lista ordenada de muestras con su partición objetivo."""
    samples: list[dict] = []
    for rep in range(1, repetitions + 1):
        for idx, ch in enumerate(CHARSET):
            if ch in TEST_UNSEEN_CHARS:
                split = "test_unseen"
            elif rep == repetitions:
                split = "val"
            else:
                split = "train"
            samples.append(
                {
                    "sample_index": len(samples),
                    "char": ch,
                    "char_code": ord(ch),
                    "class_id": idx,
                    "repetition": rep,
                    "split": split,
                }
            )
    return samples


def compute_grid_geometry() -> dict:
    """Calcula coordenadas exactas de marcadores y celdas de la cuadrícula."""
    markers = {
        "top_left": [MARKER_MARGIN, MARKER_MARGIN, MARKER_SIZE, MARKER_SIZE],
        "top_right": [
            PAGE_W - MARKER_MARGIN - MARKER_SIZE,
            MARKER_MARGIN,
            MARKER_SIZE,
            MARKER_SIZE,
        ],
        "bottom_left": [
            MARKER_MARGIN,
            PAGE_H - MARKER_MARGIN - MARKER_SIZE,
            MARKER_SIZE,
            MARKER_SIZE,
        ],
        "bottom_right": [
            PAGE_W - MARKER_MARGIN - MARKER_SIZE,
            PAGE_H - MARKER_MARGIN - MARKER_SIZE,
            MARKER_SIZE,
            MARKER_SIZE,
        ],
    }

    grid_left = MARKER_MARGIN + MARKER_SIZE + 70
    grid_right = PAGE_W - MARKER_MARGIN - MARKER_SIZE - 70
    grid_top = MARKER_MARGIN + MARKER_SIZE + 120
    grid_bottom = PAGE_H - MARKER_MARGIN - MARKER_SIZE - 70

    cell_gap_x = 32
    cell_gap_y = 68
    label_height = 46

    total_w = grid_right - grid_left
    total_h = grid_bottom - grid_top
    box_w = (total_w - cell_gap_x * (COLS - 1)) // COLS
    box_h = (total_h - cell_gap_y * (ROWS - 1)) // ROWS - label_height
    box_size = min(box_w, box_h)

    used_w = COLS * box_size + (COLS - 1) * cell_gap_x
    offset_x = grid_left + (total_w - used_w) // 2

    cells: list[dict] = []
    for row in range(ROWS):
        for col in range(COLS):
            x0 = offset_x + col * (box_size + cell_gap_x)
            y_label = grid_top + row * (box_size + label_height + cell_gap_y)
            y0 = y_label + label_height
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "label_pos": [x0, y_label],
                    "box": [x0, y0, x0 + box_size, y0 + box_size],
                }
            )

    return {
        "page_size": [PAGE_W, PAGE_H],
        "cols": COLS,
        "rows": ROWS,
        "box_size": box_size,
        "baseline_ratio": 0.72,
        "markers": markers,
        "cells": cells,
    }


def render_templates(output_dir: Path, repetitions: int = 4) -> dict:
    """Renderiza las hojas PNG, el PDF multipágina y guarda template_layout.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_sample_list(repetitions=repetitions)
    geometry = compute_grid_geometry()

    font_header = load_font(36)
    font_sub = load_font(24)
    font_label = load_font(32)

    pages: list[Image.Image] = []
    layout_pages: list[dict] = []
    total_pages = (len(samples) + CELLS_PER_PAGE - 1) // CELLS_PER_PAGE

    for page_idx in range(total_pages):
        img = Image.new("RGB", (PAGE_W, PAGE_H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 4 marcadores negros en las esquinas
        for mx, my, mw, mh in geometry["markers"].values():
            draw.rectangle([mx, my, mx + mw, my + mh], fill=(0, 0, 0))

        # Encabezado fuera de los marcadores
        header_x = MARKER_MARGIN + MARKER_SIZE + 70
        draw.text(
            (header_x, MARKER_MARGIN + 6),
            f"Plantilla Manuscrita cGAN  |  Hoja {page_idx + 1}/{total_pages}",
            fill=(30, 30, 30),
            font=font_header,
        )
        draw.text(
            (header_x, MARKER_MARGIN + 54),
            "Usa pluma negra (0.5-0.7 mm). Escribe dentro del recuadro respetando la linea base tenue.",
            fill=(110, 110, 110),
            font=font_sub,
        )

        start = page_idx * CELLS_PER_PAGE
        page_samples = samples[start : start + CELLS_PER_PAGE]
        page_cells: list[dict] = []

        for local_idx, sample in enumerate(page_samples):
            cell_geom = geometry["cells"][local_idx]
            lx, ly = cell_geom["label_pos"]
            x0, y0, x1, y1 = cell_geom["box"]

            # Etiqueta del símbolo arriba del recuadro
            ch = sample["char"]
            draw.text((lx + 6, ly + 4), f"{ch}", fill=(90, 90, 90), font=font_label)
            draw.text(
                (lx + 56, ly + 10),
                f"r{sample['repetition']}",
                fill=(165, 165, 165),
                font=font_sub,
            )

            # Recuadro y línea base tenue
            draw.rectangle([x0, y0, x1, y1], outline=(170, 170, 170), width=3)
            baseline_y = int(y0 + (y1 - y0) * geometry["baseline_ratio"])
            draw.line([(x0 + 10, baseline_y), (x1 - 10, baseline_y)], fill=(225, 225, 225), width=2)

            page_cells.append(
                {
                    **sample,
                    "row": cell_geom["row"],
                    "col": cell_geom["col"],
                    "box": cell_geom["box"],
                }
            )

        png_path = output_dir / f"plantilla_hoja_{page_idx + 1:02d}.png"
        img.save(png_path, dpi=(300, 300))
        pages.append(img)
        layout_pages.append({"page_index": page_idx + 1, "image": png_path.name, "cells": page_cells})

    pdf_path = output_dir / "plantilla_completa.pdf"
    if pages:
        pages[0].save(
            pdf_path,
            "PDF",
            resolution=300.0,
            save_all=True,
            append_images=pages[1:],
        )

    layout = {
        "charset_size": len(CHARSET),
        "repetitions": repetitions,
        "total_samples": len(samples),
        "test_unseen_chars": sorted(TEST_UNSEEN_CHARS),
        "geometry": {
            "page_size": geometry["page_size"],
            "cols": geometry["cols"],
            "rows": geometry["rows"],
            "box_size": geometry["box_size"],
            "baseline_ratio": geometry["baseline_ratio"],
            "markers": geometry["markers"],
        },
        "pages": layout_pages,
    }
    layout_path = output_dir / "template_layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")

    return layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera plantillas imprimibles para captura manuscrita.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/templates"),
        help="Carpeta de salida para las plantillas PNG, PDF y template_layout.json.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=4,
        help="Número de repeticiones por cada símbolo (default: 4).",
    )
    args = parser.parse_args()

    layout = render_templates(args.output_dir, repetitions=args.repetitions)
    print(
        f"Plantillas generadas en '{args.output_dir}': "
        f"{len(layout['pages'])} hojas ({layout['total_samples']} celdas) + template_layout.json"
    )


if __name__ == "__main__":
    main()
