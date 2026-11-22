"""Extrae y normaliza recortes manuscritos (1x128x128) desde hojas escaneadas o fotografiadas.

Pipeline OpenCV por hoja:
1. Detecta los 4 marcadores negros cuadrados de las esquinas.
2. Aplica corrección de perspectiva (`warpPerspective`) al lienzo calibrado (2550x3300).
3. Recorta cada celda según `data/templates/template_layout.json` eliminando bordes y línea base.
4. Limpia ruido/sombras con umbral adaptativo, centra horizontalmente respetando proporciones
   verticales y empareja cada glifo real `y` con su glifo base `x` (misma fuente que sintéticos).
5. Guarda `data/real_pairs/{train,val,test_unseen}/{x,y}/` y `metadata.csv`.

Uso real:
    python -m src.data.02_extraer_recortes --scans-dir data/raw_scans --output-dir data/real_pairs

Uso de prueba simulada (sin hojas impresas):
    python -m src.data.02_extraer_recortes --simulate-test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import IMG_SIZE
from src.data.glyph_render import get_default_base_font_path, render_glyph

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def detect_corner_markers(gray: np.ndarray) -> np.ndarray:
    """Detecta los centros (x, y) de los 4 marcadores negros de esquina.

    Devuelve un arreglo float32 de forma (4, 2) ordenado:
    [top_left, top_right, bottom_right, bottom_left].
    """
    h, w = gray.shape[:2]
    img_area = float(h * w)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Umbral adaptativo invertido + Otsu combinado para manejar tanto fotos con sombra como escaneos limpios
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )

    candidates: list[tuple[float, float, float]] = []  # (cx, cy, area)
    for binary in (thresh_otsu, thresh_adapt):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.0002 or area > img_area * 0.03:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0 or bw == 0:
                continue
            aspect = bw / float(bh)
            if not (0.70 <= aspect <= 1.40):
                continue
            fill_ratio = area / float(bw * bh)
            if fill_ratio < 0.72:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.85:
                continue
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            candidates.append((cx, cy, area))

        if len(candidates) >= 4:
            break

    if len(candidates) < 4:
        raise RuntimeError(
            f"Solo se detectaron {len(candidates)} marcadores válidos (se requieren 4). "
            "Verifica que los 4 cuadros negros de esquina sean completamente visibles."
        )

    pts = np.array([[c[0], c[1]] for c in candidates], dtype=np.float32)

    # Seleccionar el punto más cercano a cada una de las 4 esquinas de la imagen
    corners = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float32,
    )
    selected = []
    used_idx: set[int] = set()
    for corner in corners:
        dists = np.linalg.norm(pts - corner, axis=1)
        for idx in np.argsort(dists):
            if int(idx) not in used_idx:
                used_idx.add(int(idx))
                selected.append(pts[int(idx)])
                break

    if len(selected) != 4:
        raise RuntimeError("No se pudieron asignar 4 marcadores únicos a las esquinas.")

    return np.array(selected, dtype=np.float32)


def warp_page_to_template(image_bgr: np.ndarray, layout: dict) -> np.ndarray:
    """Endereza una foto/escaneo al tamaño calibrado de la plantilla usando los 4 marcadores."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    src_pts = detect_corner_markers(gray)

    geom = layout["geometry"]
    page_w, page_h = geom["page_size"]
    markers = geom["markers"]

    def _center(rect: list[int]) -> list[float]:
        mx, my, mw, mh = rect
        return [mx + mw * 0.5, my + mh * 0.5]

    dst_pts = np.array(
        [
            _center(markers["top_left"]),
            _center(markers["top_right"]),
            _center(markers["bottom_right"]),
            _center(markers["bottom_left"]),
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        gray,
        matrix,
        (page_w, page_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped


def clean_and_normalize_cell(
    cell_gray: np.ndarray,
    img_size: int = IMG_SIZE,
    inner_margin: int = 14,
) -> tuple[np.ndarray, bool]:
    """Limpia el recorte de una celda y lo normaliza a `128x128` (fondo 255, tinta 0).

    Devuelve `(imagen_128x128, tiene_tinta)`.
    Preserva la proporción vertical respecto a la celda original (para que signos
    pequeños como `.` o `-` no se agranden artificialmente) y centra horizontalmente.
    """
    h, w = cell_gray.shape[:2]
    m = min(inner_margin, h // 6, w // 6)
    roi = cell_gray[m : h - m, m : w - m]

    # Umbral adaptativo: elimina línea base tenue (gris ~225) y sombras suaves
    ink_mask = cv2.adaptiveThreshold(
        roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        18,
    )

    # También requerir que el píxel sea realmente más oscuro que papel claro (< 200)
    ink_mask = cv2.bitwise_and(ink_mask, (roi < 200).astype(np.uint8) * 255)

    # Eliminar motas diminutas de ruido (< 12 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    cleaned_mask = np.zeros_like(ink_mask)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 12:
            cleaned_mask[labels == lbl] = 255

    ys, xs = np.where(cleaned_mask > 0)
    if len(xs) == 0:
        return np.full((img_size, img_size), 255, dtype=np.uint8), False

    # Fondo blanco puro fuera de la tinta, preservando antialiasing/intensidad de tinta real
    cleaned_gray = np.full_like(roi, 255, dtype=np.uint8)
    dilated = cv2.dilate(cleaned_mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned_gray[dilated > 0] = roi[dilated > 0]
    # Estirar contraste de tinta para que el negro sea nítido
    ink_pixels = cleaned_gray[cleaned_mask > 0]
    if ink_pixels.size > 0:
        p5 = float(np.percentile(ink_pixels, 5))
        if p5 < 200:
            norm = np.clip((cleaned_gray.astype(np.float32) - p5) / max(1.0, 220.0 - p5), 0.0, 1.0)
            cleaned_gray = (norm * 255.0).astype(np.uint8)
    cleaned_gray[dilated == 0] = 255

    # Escalar de forma proporcional al tamaño de la celda para conservar escala tipográfica
    roi_h, roi_w = cleaned_gray.shape[:2]
    scale = (img_size - 12) / float(max(roi_h, roi_w))
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    glyph_crop = cleaned_gray[y_min : y_max + 1, x_min : x_max + 1]

    new_w = max(1, int(round(glyph_crop.shape[1] * scale)))
    new_h = max(1, int(round(glyph_crop.shape[0] * scale)))

    # Si aun así excede el margen de seguridad, ajustar manteniendo relación de aspecto
    max_span = img_size - 12
    if new_w > max_span or new_h > max_span:
        fit = min(max_span / new_w, max_span / new_h)
        new_w = max(1, int(round(new_w * fit)))
        new_h = max(1, int(round(new_h * fit)))

    resized = cv2.resize(glyph_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centrar horizontalmente y conservar posición vertical relativa dentro de la celda
    paste_x = (img_size - new_w) // 2
    rel_y_center = ((y_min + y_max) * 0.5) / float(roi_h)
    paste_y = int(round(rel_y_center * img_size - new_h * 0.5))
    paste_x = max(2, min(paste_x, img_size - new_w - 2))
    paste_y = max(2, min(paste_y, img_size - new_h - 2))

    out = np.full((img_size, img_size), 255, dtype=np.uint8)
    out[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
    return out, True


def extract_from_scans(
    scans_dir: Path,
    layout_path: Path,
    output_dir: Path,
    base_font: Path | None = None,
) -> Path:
    """Procesa las hojas escaneadas/fotografiadas en orden y genera `data/real_pairs/`."""
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if base_font is None:
        base_font = get_default_base_font_path()

    scan_files = sorted(
        p for p in scans_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not scan_files:
        raise FileNotFoundError(
            f"No se encontraron imágenes en '{scans_dir}'. Guarda ahí tus hojas escaneadas."
        )

    for split in ("train", "val", "test_unseen"):
        (output_dir / split / "x").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "y").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    empty_cells = 0

    for page_meta, scan_path in zip(layout["pages"], scan_files):
        img_bgr = cv2.imread(str(scan_path))
        if img_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {scan_path}")

        warped = warp_page_to_template(img_bgr, layout)
        page_idx = page_meta["page_index"]

        for cell in page_meta["cells"]:
            x0, y0, x1, y1 = cell["box"]
            cell_crop = warped[y0:y1, x0:x1]
            y_img, has_ink = clean_and_normalize_cell(cell_crop, img_size=IMG_SIZE)
            if not has_ink:
                empty_cells += 1
                continue

            ch = cell["char"]
            split = cell["split"]
            sample_idx = cell["sample_index"]
            class_id = cell["class_id"]
            rep = cell["repetition"]

            x_img = render_glyph(ch, base_font, img_size=IMG_SIZE)

            fname = f"{sample_idx:04d}_c{class_id:02d}_r{rep}_p{page_idx:02d}.png"
            x_rel = Path(split) / "x" / fname
            y_rel = Path(split) / "y" / fname

            Image.fromarray(x_img, mode="L").save(output_dir / x_rel)
            Image.fromarray(y_img, mode="L").save(output_dir / y_rel)

            rows.append(
                {
                    "sample_id": sample_idx,
                    "char": ch,
                    "char_code": cell["char_code"],
                    "class_id": class_id,
                    "repetition": rep,
                    "page_index": page_idx,
                    "split": split,
                    "x_path": x_rel.as_posix(),
                    "y_path": y_rel.as_posix(),
                }
            )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "char",
                "char_code",
                "class_id",
                "repetition",
                "page_index",
                "split",
                "x_path",
                "y_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Extracción completada en '{output_dir}': {len(rows)} pares guardados "
        f"({empty_cells} celdas vacías omitidas)."
    )
    return metadata_path


def run_simulated_test(
    layout_path: Path = Path("data/templates/template_layout.json"),
    template_png: Path = Path("data/templates/plantilla_hoja_01.png"),
    out_preview: Path = Path("docs/figuras/checkpoint1_opencv_simulado.png"),
    num_cells: int = 6,
) -> bool:
    """Prueba de humo del extractor OpenCV sobre una hoja simulada con perspectiva."""
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    page_img = cv2.imread(str(template_png), cv2.IMREAD_GRAYSCALE)
    if page_img is None:
        raise FileNotFoundError(f"No se encontró '{template_png}'.")

    base_font = get_default_base_font_path()
    page_meta = layout["pages"][0]
    test_cells = page_meta["cells"][:num_cells]

    # Dibujar glifos sintéticos dentro de las primeras `num_cells` celdas
    simulated = page_img.copy()
    for cell in test_cells:
        x0, y0, x1, y1 = cell["box"]
        box_w, box_h = x1 - x0, y1 - y0
        glyph = render_glyph(cell["char"], base_font, img_size=box_w - 36)
        gx0 = x0 + 18
        gy0 = y0 + 18
        roi = simulated[gy0 : gy0 + glyph.shape[0], gx0 : gx0 + glyph.shape[1]]
        simulated[gy0 : gy0 + glyph.shape[0], gx0 : gx0 + glyph.shape[1]] = np.minimum(roi, glyph)

    # Aplicar deformación de perspectiva + rotación simulada (~4°) + ligera sombra
    h, w = simulated.shape[:2]
    src_quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_quad = np.array(
        [[65, 90], [w - 110, 40], [w - 45, h - 85], [105, h - 40]],
        dtype=np.float32,
    )
    warp_sim = cv2.warpPerspective(
        simulated,
        cv2.getPerspectiveTransform(src_quad, dst_quad),
        (w, h),
        borderValue=235,
    )

    # Recuperar con el pipeline real
    recovered = warp_page_to_template(warp_sim, layout)
    strips: list[np.ndarray] = []
    recovered_count = 0

    for cell in test_cells:
        x0, y0, x1, y1 = cell["box"]
        y_img, has_ink = clean_and_normalize_cell(recovered[y0:y1, x0:x1], img_size=IMG_SIZE)
        x_img = render_glyph(cell["char"], base_font, img_size=IMG_SIZE)
        if has_ink:
            recovered_count += 1
        pair = np.hstack([x_img, np.full((IMG_SIZE, 4), 180, dtype=np.uint8), y_img])
        strips.append(pair)

    grid = np.vstack(strips)
    out_preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(out_preview)
    print(
        f"[SIMULACIÓN OPENCV] Recuperados {recovered_count}/{num_cells} glifos tras perspectiva. "
        f"Vista previa guardada en: {out_preview}"
    )
    return recovered_count == num_cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae glifos 128x128 desde hojas manuscritas escaneadas o fotografiadas."
    )
    parser.add_argument("--scans-dir", type=Path, default=Path("data/raw_scans"))
    parser.add_argument("--layout", type=Path, default=Path("data/templates/template_layout.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_pairs"))
    parser.add_argument("--base-font", type=Path, default=None)
    parser.add_argument(
        "--simulate-test",
        action="store_true",
        help="Ejecuta una prueba simulada con perspectiva sobre plantilla_hoja_01.png.",
    )
    args = parser.parse_args()

    if args.simulate_test:
        ok = run_simulated_test(layout_path=args.layout)
        if not ok:
            raise SystemExit("La simulación OpenCV no recuperó todos los glifos de prueba.")
        return

    extract_from_scans(
        scans_dir=args.scans_dir,
        layout_path=args.layout,
        output_dir=args.output_dir,
        base_font=args.base_font,
    )


if __name__ == "__main__":
    main()
