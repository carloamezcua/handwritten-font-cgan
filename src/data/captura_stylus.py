"""Captura guiada de glifos manuscritos con stylus / tableta digitalizadora (o mouse).

Por defecto abre una interfaz local fluida en el navegador (Windows Ink +
`getCoalescedEvents` + presión del lápiz + suavizado Bézier estilo OneNote, con
grosor fino y tamaño de recuadro ajustable). También conserva `--opencv` como
modo clásico de respaldo.

Muestra carácter por carácter según `data/templates/template_layout.json` (300 muestras:
75 símbolos × 4 repeticiones), guarda el progreso automáticamente en
`data/raw_scans/_stylus_cells/` y compone cada glifo dentro de las 7 plantillas originales
(`data/templates/plantilla_hoja_01.png` … `07.png`) para guardarlas en
`data/raw_scans/hoja_01.png` … `hoja_07.png`.

Así el pipeline completo de extracción OpenCV (`src.data.02_extraer_recortes`) funciona
exactamente igual que con hojas impresas y escaneadas.

Controles:
    - Stylus / clic izquierdo: dibujar en el lienzo.
    - Espacio / Enter / botón [Guardar y Sig.]: guarda el trazo y avanza al siguiente.
    - Z / borrador del stylus / clic derecho / botón [Deshacer]: deshace el último trazo.
    - R / C / botón [Limpiar]: borra el lienzo actual.
    - Flecha Izq / Backspace / botón [Anterior]: regresa al glifo anterior.
    - Flecha Der / botón [Saltar]: avanza sin modificar (útil al revisar glifos).
    - + / -: ajusta el grosor del trazo (default 6.5 px en plantilla ≈ pluma 0.5 mm).
    - [ / ]: reduce o agranda el tamaño del recuadro de dibujo en pantalla.
    - E: exporta en ese momento las 7 hojas rellenas a `data/raw_scans/`.

Uso:
    python -m src.data.captura_stylus
    python -m src.data.captura_stylus --brush 6.5 --canvas-size 320
    python -m src.data.captura_stylus --opencv
    python -m src.data.captura_stylus --export-only
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from src.data.glyph_render import get_default_base_font_path, render_glyph
from src.data.stylus_web_ui import run_web_capture

WINDOW_NAME = "Captura Stylus — Plantillas Manuscritas (300 glifos)"

# Geometría de la interfaz (canvas 2x respecto a box_size=228 -> 456x456 px)
CANVAS_SCALE = 2
LEFT_PANEL_W = 300
PAD = 20
HEADER_H = 56
BUTTON_BAR_H = 66
FOOTER_H = 38

# Margen interno que recorta `02_extraer_recortes.py` (inner_margin=14 en espacio 228 px)
EXTRACTOR_INNER_MARGIN = 14

SYMBOL_DESCRIPTIONS: dict[str, str] = {
    ".": "Punto",
    ",": "Coma",
    "!": "Signo de exclamacion",
    "?": "Signo de interrogacion",
    "(": "Parentesis izquierdo (abre)",
    ")": "Parentesis derecho (cierra)",
    "+": "Signo mas",
    "-": "Guion / signo menos",
    "=": "Signo igual",
    "/": "Diagonal (barra)",
    ":": "Dos puntos",
    ";": "Punto y coma",
    "&": "Ampersand (y comercial)",
}


def describe_char(ch: str) -> str:
    """Devuelve una etiqueta sin ambigüedad para distinguir I/l/1, O/0/o, signos, etc."""
    if ch in SYMBOL_DESCRIPTIONS:
        return SYMBOL_DESCRIPTIONS[ch]
    if ch.isdigit():
        return f"Digito {ch}"
    if ch.isupper():
        return f"Mayuscula '{ch}'"
    if ch.islower():
        return f"Minuscula '{ch}'"
    return f"Simbolo '{ch}'"


def load_layout(layout_path: Path) -> tuple[dict, list[dict]]:
    """Carga `template_layout.json` y aplana la lista ordenada de 300 celdas."""
    if not layout_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{layout_path}'. Ejecuta primero `python -m src.data.01_generar_plantilla`."
        )
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    flat_cells: list[dict] = []
    for page in layout["pages"]:
        for cell in page["cells"]:
            item = dict(cell)
            item["page_index"] = page["page_index"]
            item["page_image"] = page["image"]
            flat_cells.append(item)
    flat_cells.sort(key=lambda c: c["sample_index"])
    return layout, flat_cells


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    """Lee una imagen soportando rutas Windows con acentos/Unicode (evita fallo de cv2.imread)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except OSError:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """Escribe una imagen soportando rutas Windows con acentos/Unicode."""
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return True


def cell_path(cells_dir: Path, sample_index: int) -> Path:
    return cells_dir / f"cell_{sample_index:04d}.png"


def has_ink(canvas_gray: np.ndarray, threshold: int = 235) -> bool:
    return int(np.count_nonzero(canvas_gray < threshold)) >= 8


def export_filled_templates(
    layout: dict,
    templates_dir: Path,
    cells_dir: Path,
    scans_dir: Path,
) -> tuple[int, int]:
    """Estampa las celdas capturadas sobre las 7 plantillas y las guarda en `data/raw_scans/`."""
    scans_dir.mkdir(parents=True, exist_ok=True)
    total_placed = 0
    pages_written = 0

    for page in layout["pages"]:
        page_idx = page["page_index"]
        tpl_path = templates_dir / page["image"]
        if not tpl_path.exists():
            raise FileNotFoundError(f"No se encontró la plantilla base: '{tpl_path}'.")

        page_img = imread_unicode(tpl_path, cv2.IMREAD_COLOR)
        if page_img is None:
            raise RuntimeError(f"No se pudo leer la plantilla: '{tpl_path}'.")

        for cell in page["cells"]:
            cpath = cell_path(cells_dir, cell["sample_index"])
            if not cpath.exists():
                continue
            cell_gray = imread_unicode(cpath, cv2.IMREAD_GRAYSCALE)
            if cell_gray is None:
                continue

            x0, y0, x1, y1 = cell["box"]
            box_w = x1 - x0
            box_h = y1 - y0
            if cell_gray.shape != (box_h, box_w):
                cell_gray = cv2.resize(cell_gray, (box_w, box_h), interpolation=cv2.INTER_AREA)

            # Evitar pisar el borde gris de 2 px de la celda
            clean_cell = cell_gray.copy()
            clean_cell[:3, :] = 255
            clean_cell[-3:, :] = 255
            clean_cell[:, :3] = 255
            clean_cell[:, -3:] = 255

            cell_bgr = cv2.cvtColor(clean_cell, cv2.COLOR_GRAY2BGR)
            roi = page_img[y0:y1, x0:x1]
            # Mezcla tipo "multiply/darken": conserva guías claras y estampa trazos negros encima
            page_img[y0:y1, x0:x1] = np.minimum(roi, cell_bgr)
            total_placed += 1

        out_path = scans_dir / f"hoja_{page_idx:02d}.png"
        imwrite_unicode(out_path, page_img)
        pages_written += 1

    return pages_written, total_placed


class StylusCaptureApp:
    """Aplicación interactiva OpenCV para capturar los 300 glifos con stylus."""

    def __init__(
        self,
        layout: dict,
        flat_cells: list[dict],
        templates_dir: Path,
        cells_dir: Path,
        scans_dir: Path,
        output_pairs_dir: Path,
        brush_tpl_px: int = 12,
        start_index: int | None = None,
    ) -> None:
        self.layout = layout
        self.cells = flat_cells
        self.templates_dir = templates_dir
        self.cells_dir = cells_dir
        self.scans_dir = scans_dir
        self.output_pairs_dir = output_pairs_dir
        self.cells_dir.mkdir(parents=True, exist_ok=True)

        self.box_size = int(layout["geometry"]["box_size"])
        self.baseline_ratio = float(layout["geometry"].get("baseline_ratio", 0.72))
        self.canvas_size = self.box_size * CANVAS_SCALE

        # Posición exacta del lienzo dentro de la ventana
        self.canvas_x0 = PAD + LEFT_PANEL_W + PAD
        self.canvas_y0 = HEADER_H + PAD
        self.canvas_x1 = self.canvas_x0 + self.canvas_size
        self.canvas_y1 = self.canvas_y0 + self.canvas_size

        self.win_w = self.canvas_x1 + PAD
        self.win_h = self.canvas_y1 + PAD + BUTTON_BAR_H + FOOTER_H

        # Grosor en píxeles de la plantilla (228 px). Rango seguro para adaptiveThreshold(35): 4..20
        self.brush_tpl_px = max(4, min(20, int(round(brush_tpl_px))))

        # Fuente base para renderizar el glifo de referencia
        self.base_font = get_default_base_font_path()
        self._ref_cache: dict[str, np.ndarray] = {}

        # Estado del lienzo y deshacer
        self.canvas = np.full((self.canvas_size, self.canvas_size), 255, dtype=np.uint8)
        self.undo_stack: list[np.ndarray] = []
        self.drawing = False
        self.prev_pt: tuple[int, int] | None = None

        # Mensaje transitorio de estado
        self.status_msg = "Dibuja el caracter con el stylus y pulsa ESPACIO o [Guardar y Sig.]"
        self.status_color = (180, 210, 230)
        self.status_until = 0.0

        # Definir botones clicables en la barra inferior
        self.buttons: list[dict] = []
        self._build_buttons()

        # Índice inicial: primer glifo aún no capturado (reanudación automática)
        if start_index is not None:
            self.idx = max(0, min(len(self.cells) - 1, start_index))
        else:
            self.idx = self._first_missing_index()

        self._load_current_cell()

    def _build_buttons(self) -> None:
        y0 = self.canvas_y1 + PAD + 8
        y1 = y0 + 46
        btn_specs = [
            ("prev", "< Anterior", (75, 80, 92), 125),
            ("undo", "Deshacer (Z)", (85, 85, 110), 145),
            ("clear", "Limpiar (R)", (70, 70, 145), 130),
            ("save_next", "Guardar y Sig. (Espacio)", (55, 135, 65), 240),
            ("export", "Exportar (E)", (140, 100, 40), 125),
        ]
        total_w = sum(w for _, _, _, w in btn_specs) + 10 * (len(btn_specs) - 1)
        x = max(PAD, (self.win_w - total_w) // 2)
        self.buttons.clear()
        for action, label, color, w in btn_specs:
            self.buttons.append(
                {
                    "action": action,
                    "label": label,
                    "color": color,
                    "rect": (x, y0, x + w, y1),
                }
            )
            x += w + 10

    def _first_missing_index(self) -> int:
        for i, cell in enumerate(self.cells):
            if not cell_path(self.cells_dir, cell["sample_index"]).exists():
                return i
        return 0

    def _count_completed(self) -> int:
        return sum(
            1 for cell in self.cells if cell_path(self.cells_dir, cell["sample_index"]).exists()
        )

    def _set_status(self, msg: str, color: tuple[int, int, int] = (140, 230, 160), duration: float = 3.0) -> None:
        self.status_msg = msg
        self.status_color = color
        self.status_until = time.time() + duration

    def _get_ref_glyph(self, ch: str) -> np.ndarray:
        if ch not in self._ref_cache:
            self._ref_cache[ch] = render_glyph(ch, self.base_font, img_size=148)
        return self._ref_cache[ch]

    def _get_prev_rep_preview(self, cell: dict) -> np.ndarray | None:
        """Devuelve la repetición anterior del mismo carácter si ya fue dibujada."""
        ch = cell["char"]
        cur_rep = cell["repetition"]
        for prev_cell in reversed(self.cells):
            if prev_cell["char"] == ch and prev_cell["repetition"] < cur_rep:
                p = cell_path(self.cells_dir, prev_cell["sample_index"])
                if p.exists():
                    img = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        return img
        return None

    def _load_current_cell(self) -> None:
        cell = self.cells[self.idx]
        p = cell_path(self.cells_dir, cell["sample_index"])
        self.undo_stack.clear()
        self.drawing = False
        self.prev_pt = None
        if p.exists():
            saved = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
            if saved is not None:
                self.canvas = cv2.resize(
                    saved,
                    (self.canvas_size, self.canvas_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                return
        self.canvas.fill(255)

    def _save_current_cell(self) -> bool:
        if not has_ink(self.canvas):
            self._set_status(
                "Lienzo vacio: dibuja el caracter antes de guardar.",
                color=(80, 110, 255),
                duration=2.5,
            )
            return False

        cell = self.cells[self.idx]
        downscaled = cv2.resize(
            self.canvas,
            (self.box_size, self.box_size),
            interpolation=cv2.INTER_AREA,
        )
        p = cell_path(self.cells_dir, cell["sample_index"])
        imwrite_unicode(p, downscaled)
        return True

    def _action_save_and_next(self) -> None:
        if not self._save_current_cell():
            return
        completed = self._count_completed()
        if self.idx < len(self.cells) - 1:
            self.idx += 1
            self._load_current_cell()
            if completed == len(self.cells):
                self._set_status("300/300 completos! Pulsa E o ESC para exportar.", (90, 240, 130))
            else:
                self._set_status(f"Guardado ({completed}/{len(self.cells)}).", (120, 230, 150), 1.5)
        else:
            pages, placed = export_filled_templates(
                self.layout, self.templates_dir, self.cells_dir, self.scans_dir
            )
            self._set_status(
                f"Completado ({placed}/{len(self.cells)})! {pages} hojas exportadas en data/raw_scans/",
                (90, 245, 130),
                6.0,
            )

    def _action_prev(self) -> None:
        if has_ink(self.canvas):
            self._save_current_cell()
        if self.idx > 0:
            self.idx -= 1
            self._load_current_cell()
            self._set_status(f"Editando muestra {self.idx + 1}/{len(self.cells)}.", (200, 210, 230), 1.5)

    def _action_skip_next(self) -> None:
        if self.idx < len(self.cells) - 1:
            self.idx += 1
            self._load_current_cell()

    def _action_undo(self) -> None:
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()
            self._set_status("Trazo deshecho.", (210, 210, 150), 1.2)
        else:
            self._set_status("No hay trazos para deshacer.", (170, 170, 180), 1.2)

    def _action_clear(self) -> None:
        if has_ink(self.canvas):
            self.undo_stack.append(self.canvas.copy())
            if len(self.undo_stack) > 40:
                self.undo_stack.pop(0)
        self.canvas.fill(255)
        self._set_status("Lienzo limpiado.", (210, 210, 150), 1.2)

    def _action_export(self) -> None:
        if has_ink(self.canvas):
            self._save_current_cell()
        pages, placed = export_filled_templates(
            self.layout, self.templates_dir, self.cells_dir, self.scans_dir
        )
        self._set_status(
            f"Exportadas {pages} plantillas ({placed}/{len(self.cells)} glifos) -> data/raw_scans/",
            (100, 235, 170),
            4.0,
        )

    def _handle_button_click(self, x: int, y: int) -> bool:
        for btn in self.buttons:
            bx0, by0, bx1, by1 = btn["rect"]
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                action = btn["action"]
                if action == "prev":
                    self._action_prev()
                elif action == "undo":
                    self._action_undo()
                elif action == "clear":
                    self._action_clear()
                elif action == "save_next":
                    self._action_save_and_next()
                elif action == "export":
                    self._action_export()
                return True
        return False

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        in_canvas = (
            self.canvas_x0 <= x < self.canvas_x1 and self.canvas_y0 <= y < self.canvas_y1
        )
        thickness = max(4, self.brush_tpl_px * CANVAS_SCALE)
        radius = max(2, thickness // 2)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._handle_button_click(x, y):
                return
            if in_canvas:
                self.undo_stack.append(self.canvas.copy())
                if len(self.undo_stack) > 40:
                    self.undo_stack.pop(0)
                self.drawing = True
                cx = x - self.canvas_x0
                cy = y - self.canvas_y0
                self.prev_pt = (cx, cy)
                cv2.circle(self.canvas, (cx, cy), radius, 0, -1, cv2.LINE_AA)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
                cx = int(np.clip(x - self.canvas_x0, 0, self.canvas_size - 1))
                cy = int(np.clip(y - self.canvas_y0, 0, self.canvas_size - 1))
                if self.prev_pt is not None:
                    cv2.line(self.canvas, self.prev_pt, (cx, cy), 0, thickness, cv2.LINE_AA)
                else:
                    cv2.circle(self.canvas, (cx, cy), radius, 0, -1, cv2.LINE_AA)
                self.prev_pt = (cx, cy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.prev_pt = None

    def _render_frame(self) -> np.ndarray:
        frame = np.full((self.win_h, self.win_w, 3), (32, 34, 40), dtype=np.uint8)
        cell = self.cells[self.idx]
        completed = self._count_completed()
        total = len(self.cells)

        # ---- Encabezado ----
        cv2.rectangle(frame, (0, 0), (self.win_w, HEADER_H), (22, 24, 30), -1)
        header_text = (
            f"Glifo {self.idx + 1}/{total}  |  Hoja {cell['page_index']}/7  |  "
            f"Rep {cell['repetition']}/4  |  Completados: {completed}/{total}"
        )
        cv2.putText(
            frame,
            header_text,
            (PAD, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )

        # Barra de progreso debajo del header
        prog_w = int((completed / max(1, total)) * self.win_w)
        cv2.rectangle(frame, (0, HEADER_H - 5), (self.win_w, HEADER_H), (45, 48, 56), -1)
        cv2.rectangle(frame, (0, HEADER_H - 5), (prog_w, HEADER_H), (75, 195, 110), -1)

        # ---- Panel izquierdo (Referencia + info) ----
        lx0, ly0 = PAD, HEADER_H + PAD
        lx1, ly1 = lx0 + LEFT_PANEL_W, self.canvas_y1
        cv2.rectangle(frame, (lx0, ly0), (lx1, ly1), (42, 45, 54), -1)
        cv2.rectangle(frame, (lx0, ly0), (lx1, ly1), (75, 80, 95), 1)

        # Etiqueta de partición (split)
        split = cell["split"]
        if split == "test_unseen":
            split_label, split_col = "TEST UNSEEN (Zero-Shot)", (60, 145, 235)
        elif split == "val":
            split_label, split_col = "VALIDACION (val)", (190, 150, 60)
        else:
            split_label, split_col = "ENTRENAMIENTO (train)", (70, 155, 95)

        cv2.rectangle(frame, (lx0 + 14, ly0 + 12), (lx1 - 14, ly0 + 40), split_col, -1)
        cv2.putText(
            frame,
            split_label,
            (lx0 + 24, ly0 + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Glifo de referencia (fuente base)
        ref = self._get_ref_glyph(cell["char"])
        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
        rx0 = lx0 + (LEFT_PANEL_W - ref.shape[1]) // 2
        ry0 = ly0 + 54
        frame[ry0 : ry0 + ref.shape[0], rx0 : rx0 + ref.shape[1]] = ref_bgr
        cv2.rectangle(
            frame,
            (rx0 - 1, ry0 - 1),
            (rx0 + ref.shape[1], ry0 + ref.shape[0]),
            (110, 115, 130),
            1,
        )

        # Descripción inequívoca del carácter
        desc = describe_char(cell["char"])
        cv2.putText(
            frame,
            desc,
            (lx0 + 16, ry0 + ref.shape[0] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (245, 245, 250),
            1,
            cv2.LINE_AA,
        )

        # Grosor de pincel actual
        cv2.putText(
            frame,
            f"Grosor pincel: {self.brush_tpl_px} px  (+ / -)",
            (lx0 + 16, ry0 + ref.shape[0] + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (185, 195, 210),
            1,
            cv2.LINE_AA,
        )

        # Indicador si esta celda ya estaba guardada
        is_saved = cell_path(self.cells_dir, cell["sample_index"]).exists()
        saved_txt = "Estado: GUARDADO" if is_saved else "Estado: Pendiente"
        saved_col = (100, 225, 130) if is_saved else (175, 180, 195)
        cv2.putText(
            frame,
            saved_txt,
            (lx0 + 16, ry0 + ref.shape[0] + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            saved_col,
            1,
            cv2.LINE_AA,
        )

        # Vista previa de repetición anterior del mismo carácter (si existe)
        prev_rep = self._get_prev_rep_preview(cell)
        py_label = ry0 + ref.shape[0] + 106
        if prev_rep is not None:
            cv2.putText(
                frame,
                "Tu trazo anterior (referencia):",
                (lx0 + 16, py_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (175, 190, 210),
                1,
                cv2.LINE_AA,
            )
            thumb = cv2.resize(prev_rep, (112, 112), interpolation=cv2.INTER_AREA)
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            tx0 = lx0 + (LEFT_PANEL_W - 112) // 2
            ty0 = py_label + 10
            frame[ty0 : ty0 + 112, tx0 : tx0 + 112] = thumb_bgr
            cv2.rectangle(frame, (tx0 - 1, ty0 - 1), (tx0 + 112, ty0 + 112), (110, 115, 130), 1)
        else:
            cv2.putText(
                frame,
                "Consejo: escribe dentro del",
                (lx0 + 16, py_label + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (165, 175, 190),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "recuadro azul usando la linea",
                (lx0 + 16, py_label + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (165, 175, 190),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "base como guia de renglon.",
                (lx0 + 16, py_label + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (165, 175, 190),
                1,
                cv2.LINE_AA,
            )

        # ---- Lienzo de dibujo (con guías solo visuales, no se guardan) ----
        canvas_bgr = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)

        # Guías suaves sobre el lienzo blanco (sin tapar los trazos negros)
        guides = np.full_like(canvas_bgr, 255)
        safe_m = EXTRACTOR_INNER_MARGIN * CANVAS_SCALE
        # Marco de margen seguro (zona que no recorta 02_extraer_recortes)
        cv2.rectangle(
            guides,
            (safe_m, safe_m),
            (self.canvas_size - safe_m, self.canvas_size - safe_m),
            (235, 220, 210),
            1,
        )
        # Línea base (renglón) exactamente igual a la plantilla (72 %)
        y_base = int(round(self.canvas_size * self.baseline_ratio))
        cv2.line(guides, (safe_m, y_base), (self.canvas_size - safe_m, y_base), (235, 195, 165), 2)
        # Guías tenues de altura de mayúsculas (~22 %) y minúsculas (~44 %) + centro vertical
        y_cap = int(round(self.canvas_size * 0.22))
        y_x = int(round(self.canvas_size * 0.44))
        x_mid = self.canvas_size // 2
        for x_dash in range(safe_m, self.canvas_size - safe_m, 14):
            cv2.line(guides, (x_dash, y_cap), (min(x_dash + 6, self.canvas_size - safe_m), y_cap), (238, 238, 244), 1)
            cv2.line(guides, (x_dash, y_x), (min(x_dash + 6, self.canvas_size - safe_m), y_x), (238, 238, 244), 1)
        for y_dash in range(safe_m, self.canvas_size - safe_m, 14):
            cv2.line(guides, (x_mid, y_dash), (x_mid, min(y_dash + 6, self.canvas_size - safe_m)), (240, 240, 245), 1)

        cv2.putText(
            guides,
            "linea base",
            (safe_m + 6, y_base - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (215, 185, 165),
            1,
            cv2.LINE_AA,
        )

        # Mezclar trazos negros por encima de las guías
        display_canvas = np.minimum(guides, canvas_bgr)
        frame[self.canvas_y0 : self.canvas_y1, self.canvas_x0 : self.canvas_x1] = display_canvas
        cv2.rectangle(
            frame,
            (self.canvas_x0 - 2, self.canvas_y0 - 2),
            (self.canvas_x1 + 1, self.canvas_y1 + 1),
            (100, 140, 190),
            2,
        )

        # ---- Botones clicables ----
        for btn in self.buttons:
            bx0, by0, bx1, by1 = btn["rect"]
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), btn["color"], -1)
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), (200, 205, 215), 1)
            (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            tx = bx0 + (bx1 - bx0 - tw) // 2
            ty = by0 + (by1 - by0 + th) // 2
            cv2.putText(
                frame,
                btn["label"],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # ---- Barra de estado inferior ----
        fy0 = self.win_h - FOOTER_H
        cv2.rectangle(frame, (0, fy0), (self.win_w, self.win_h), (22, 24, 30), -1)
        if time.time() <= self.status_until:
            msg, col = self.status_msg, self.status_color
        else:
            msg = "Teclas: [Espacio/Enter] Guardar+Sig | [Z] Deshacer | [R] Limpiar | [Izq/Der] Navegar | [+/-] Grosor | [ESC] Salir"
            col = (175, 182, 195)
        cv2.putText(
            frame,
            msg,
            (PAD, fy0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            col,
            1,
            cv2.LINE_AA,
        )

        return frame

    def run(self) -> tuple[int, int]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        try:
            while True:
                frame = self._render_frame()
                cv2.imshow(WINDOW_NAME, frame)

                # Detectar cierre desde la 'X' de la ventana
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKeyEx(10)
                if key == -1:
                    continue

                # Teclas especiales (flechas Windows / Linux)
                if key in (2424832, 81, 8):  # Flecha izquierda / Backspace
                    self._action_prev()
                    continue
                if key in (2555904, 83):  # Flecha derecha
                    self._action_skip_next()
                    continue

                k = key & 0xFF
                if k in (27, ord("q"), ord("Q")):  # ESC / Q
                    if has_ink(self.canvas):
                        self._save_current_cell()
                    break
                elif k in (32, 13):  # Espacio / Enter
                    self._action_save_and_next()
                elif k in (ord("z"), ord("Z")):
                    self._action_undo()
                elif k in (ord("r"), ord("R"), ord("c"), ord("C")):
                    self._action_clear()
                elif k in (ord("+"), ord("="), ord("]")):
                    self.brush_tpl_px = min(20, self.brush_tpl_px + 1)
                    self._set_status(f"Grosor del pincel: {self.brush_tpl_px} px", (180, 220, 240), 1.5)
                elif k in (ord("-"), ord("_"), ord("[")):
                    self.brush_tpl_px = max(4, self.brush_tpl_px - 1)
                    self._set_status(f"Grosor del pincel: {self.brush_tpl_px} px", (180, 220, 240), 1.5)
                elif k in (ord("e"), ord("E")):
                    self._action_export()
        finally:
            cv2.destroyAllWindows()

        return export_filled_templates(
            self.layout, self.templates_dir, self.cells_dir, self.scans_dir
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura guiada de los 300 glifos con stylus e integración en plantillas."
    )
    parser.add_argument("--templates-dir", type=Path, default=Path("data/templates"))
    parser.add_argument("--scans-dir", type=Path, default=Path("data/raw_scans"))
    parser.add_argument("--output-pairs-dir", type=Path, default=Path("data/real_pairs"))
    parser.add_argument(
        "--brush",
        type=float,
        default=6.5,
        help="Grosor inicial del trazo en px de plantilla (3.5..16, default 6.5 ≈ pluma 0.5 mm).",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=320,
        help="Tamaño inicial del recuadro de dibujo en pantalla en px (200..540, default 320).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Puerto local para la interfaz fluida en navegador (0 = automático).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abre el navegador automáticamente (imprime la URL local).",
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Usa la ventana clásica de OpenCV en lugar de la interfaz fluida en navegador.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Índice de muestra (0..299) desde donde abrir. Por defecto reanuda en la primera pendiente.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Solo recompone `data/raw_scans/hoja_01..07.png` desde las celdas ya guardadas sin abrir ventana.",
    )
    parser.add_argument(
        "--auto-extract",
        action="store_true",
        help="Al terminar los 300 glifos, ejecuta automáticamente `02_extraer_recortes` sobre `data/raw_scans/`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout_path = args.templates_dir / "template_layout.json"
    cells_dir = args.scans_dir / "_stylus_cells"
    layout, flat_cells = load_layout(layout_path)

    if args.export_only:
        pages, placed = export_filled_templates(layout, args.templates_dir, cells_dir, args.scans_dir)
        print(f"Plantillas exportadas: {pages} hojas ({placed}/{len(flat_cells)} glifos) en {args.scans_dir}")
        return

    if args.opencv:
        app = StylusCaptureApp(
            layout=layout,
            flat_cells=flat_cells,
            templates_dir=args.templates_dir,
            cells_dir=cells_dir,
            scans_dir=args.scans_dir,
            output_pairs_dir=args.output_pairs_dir,
            brush_tpl_px=int(round(args.brush)),
            start_index=args.start_index,
        )
        pages, placed = app.run()
    else:
        pages, placed = run_web_capture(
            layout=layout,
            flat_cells=flat_cells,
            templates_dir=args.templates_dir,
            cells_dir=cells_dir,
            scans_dir=args.scans_dir,
            describe_char_fn=describe_char,
            cell_path_fn=cell_path,
            has_ink_fn=has_ink,
            export_fn=export_filled_templates,
            default_brush_px=args.brush,
            default_canvas_px=args.canvas_size,
            start_index=args.start_index,
            port=args.port,
            open_browser=not args.no_browser,
        )

    print(f"\n[OK] Exportadas {pages} plantillas con {placed}/{len(flat_cells)} glifos en '{args.scans_dir}'.")

    if placed == len(flat_cells):
        if args.auto_extract:
            extractor = importlib.import_module("src.data.02_extraer_recortes")
            extractor.extract_from_scans(
                scans_dir=args.scans_dir,
                layout_path=layout_path,
                output_dir=args.output_pairs_dir,
            )
            print(f"[OK] Pares reales generados en '{args.output_pairs_dir}'.")
        else:
            print("Siguiente paso para extraer y normalizar los 300 pares reales:")
            print(
                f"  python -m src.data.02_extraer_recortes --scans-dir {args.scans_dir} "
                f"--output-dir {args.output_pairs_dir}"
            )
    else:
        print(
            f"Quedan {len(flat_cells) - placed} glifos pendientes. "
            "Vuelve a ejecutar `python -m src.data.captura_stylus` para reanudar justo donde te quedaste."
        )


if __name__ == "__main__":
    main()
