"""Demo interactiva en tiempo real (OpenCV + ONNX Runtime).

Ventana triple:
    [ Entrada x (Canvas / Webcam ROI) | Glifo generado ŷ | Predicción + Confianza + FPS ]

Controles:
    - Cualquier carácter imprimible (`A-Z`, `a-z`, `0-9`, signos): cambia el glifo en Canvas.
    - Mouse (clic izquierdo arrastrando en panel Entrada x): dibujo libre sobre el Canvas.
    - `M` / `Tab`: alterna entre modo CANVAS y modo WEBCAM ROI.
    - `+` / `-`: ajusta el umbral de confianza OOD (default 0.70).
    - `C`: limpia el dibujo libre y vuelve al carácter base.
    - `S`: guarda una captura de la ventana en `results/demo_capture.png`.
    - `ESC`: salir (también `Ctrl+Q` o cerrar ventana).

Arranque en 1 solo comando:
    python app/demo_realtime.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Garantizar importación de `src.*` al ejecutar `python app/demo_realtime.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import onnxruntime as ort

from src.config import CHARSET, IDX_TO_CHAR, IMG_SIZE
from src.data.glyph_render import get_default_base_font_path, render_glyph

PANEL_SIZE = 320
HEADER_H = 44
FOOTER_H = 54
WINDOW_NAME = "Handwritten Font cGAN — Demo Tiempo Real (ONNX)"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _create_ort_session(onnx_path: Path, intra_threads: int = 4) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"No se encontró '{onnx_path}'. Descarga los pesos con "
            "`python -m src.data.download_artifacts` o expórtalos con `python -m src.export.export_onnx`."
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = min(intra_threads, os.cpu_count() or intra_threads)
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _preprocess_webcam_roi(roi_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Limpia un ROI BGR de webcam, recorta por bounding box y centra el glifo en 128x128.

    Pasos:
    1. Detecta cámara tapada o escena plana sin contraste (`is_valid = False`).
    2. Combina umbral adaptativo + contraste mínimo respecto al papel (`p85`) para ignorar sombras.
    3. Elimina ruido y manchas pegadas al borde mediante componentes conexos, preservando
       glifos multiparte (`i`, `j`, `?`, `!`, `=`, `:`, `;`, `ñ`).
    4. Recorta el bounding box del carácter, escala conservando relación de aspecto (~64 % de 128 px)
       y centra sobre fondo blanco puro `255`.
    """
    blank = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    if roi_bgr is None or roi_bgr.size == 0:
        return blank, False

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # 1. Detectar cámara tapada o escena sin contraste
    if float(gray.std()) < 10.0 or float(gray.mean()) < 25.0:
        return blank, False

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bg_level = float(np.percentile(blur, 85))
    if bg_level < 45.0:
        return blank, False

    # 2. Máscara de tinta (255 = trazo, 0 = fondo) robusta a sombras suaves
    h, w = blur.shape
    block_size = max(15, (min(h, w) // 6) | 1)
    ink_adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        14,
    )
    ink_contrast = (blur < (bg_level - 26.0)).astype(np.uint8) * 255
    ink = cv2.bitwise_and(ink_adapt, ink_contrast)

    # Limpiar un margen exterior del 5 % para evitar bordes de hoja/dedos
    margin = max(2, int(min(h, w) * 0.05))
    ink[:margin, :] = 0
    ink[-margin:, :] = 0
    ink[:, :margin] = 0
    ink[:, -margin:] = 0

    # Cerrar pequeños huecos en el trazo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=1)

    total_area = float(h * w)
    if float(np.count_nonzero(ink)) / total_area > 0.60:
        return blank, False

    # 3. Filtrado de componentes conexos (descarta motas minúsculas y bloques pegados al borde)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(14, int(total_area * 0.0005))
    max_area = int(total_area * 0.45)

    valid_ids: list[int] = []
    for idx in range(1, num_labels):
        cx0, cy0, cw, ch, area = stats[idx]
        if area < min_area or area > max_area:
            continue
        touches_border = (
            cx0 <= margin
            or cy0 <= margin
            or (cx0 + cw) >= (w - margin)
            or (cy0 + ch) >= (h - margin)
        )
        if touches_border and area > total_area * 0.15:
            continue
        valid_ids.append(idx)

    if not valid_ids:
        return blank, False

    # Tomar el componente principal (más cercano al centro ponderado por área) y conservar sus vecinos
    img_cx, img_cy = w / 2.0, h / 2.0

    def _score_comp(cid: int) -> float:
        area = float(stats[cid, cv2.CC_STAT_AREA])
        dist = float(np.hypot(centroids[cid][0] - img_cx, centroids[cid][1] - img_cy))
        return area / (1.0 + dist / max(min(h, w) * 0.35, 1.0))

    main_id = max(valid_ids, key=_score_comp)
    main_cx, main_cy = centroids[main_id]
    keep_radius = max(h, w) * 0.48

    clean_mask = np.zeros_like(ink)
    for cid in valid_ids:
        dist_main = float(np.hypot(centroids[cid][0] - main_cx, centroids[cid][1] - main_cy))
        if dist_main <= keep_radius:
            clean_mask[labels == cid] = 255

    ys, xs = np.where(clean_mask > 0)
    if len(xs) < min_area:
        return blank, False

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 and bh < 6:
        return blank, False

    # 4. Recorte, escalado proporcional (~54 % = ~69 px, igual a Lato) y alineación a línea base 0.72
    crop_ink = clean_mask[y0:y1, x0:x1]
    target_box = int(round(IMG_SIZE * 0.54))
    scale = target_box / float(max(bw, bh))
    # Evitar que trazos finos desaparezcan al reducir fuertemente el ROI
    if scale < 0.5:
        crop_ink = cv2.dilate(crop_ink, kernel, iterations=1)
    crop_glyph = 255 - crop_ink  # trazo negro (0), fondo blanco (255)

    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop_glyph, (new_w, new_h), interpolation=interp)

    # Contraste suave conservando antialiasing (evita romper trazos finos como un umbral duro)
    resized = np.clip((resized.astype(np.float32) - 40.0) * (255.0 / 185.0), 0.0, 255.0).astype(
        np.uint8
    )

    canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    offset_x = (IMG_SIZE - new_w) // 2
    offset_y = int(round(IMG_SIZE * 0.72) - new_h)
    offset_y = int(np.clip(offset_y, 8, IMG_SIZE - new_h - 8))
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    ink_ratio = float(np.mean(canvas < 128))
    is_valid = 0.005 <= ink_ratio <= 0.65
    return canvas, is_valid


def _normalize_freehand_canvas(raw_canvas: np.ndarray) -> tuple[np.ndarray, bool]:
    """Recorta, escala (~56 %) y alinea a línea base 0.72 un dibujo libre hecho con mouse."""
    blank = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    ys, xs = np.where(raw_canvas < 200)
    if len(xs) < 12:
        return blank, False

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 and bh < 4:
        return blank, False

    crop = raw_canvas[y0:y1, x0:x1]
    target_box = int(round(IMG_SIZE * 0.56))
    scale = target_box / float(max(bw, bh))
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    if scale > 1.3:
        resized = cv2.GaussianBlur(resized, (3, 3), 0)

    canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    offset_x = (IMG_SIZE - new_w) // 2
    offset_y = int(round(IMG_SIZE * 0.72) - new_h)
    offset_y = int(np.clip(offset_y, 8, IMG_SIZE - new_h - 8))
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

    ink_ratio = float(np.mean(canvas < 180))
    is_valid = 0.003 <= ink_ratio <= 0.85
    return canvas, is_valid


def _draw_panel_header(panel: np.ndarray, text: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (35, 35, 40), -1)
    cv2.putText(
        panel,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def _build_info_panel(
    pred_char: str,
    confidence: float,
    top3: list[tuple[str, float]],
    threshold: float,
    is_valid_input: bool,
    latency_ms: float,
    fps: float,
    mode: str,
    target_char: str,
    judge_source_label: str = "y_hat",
    webcam_error: str | None = None,
) -> np.ndarray:
    """Construye el 3er panel: Predicción, Confianza, Alerta OOD y métricas de tiempo real."""
    panel = np.full((PANEL_SIZE, PANEL_SIZE, 3), 24, dtype=np.uint8)
    _draw_panel_header(panel, "3. Evaluacion & Tiempo Real")

    is_ok = is_valid_input and (confidence >= threshold)
    status_color = (60, 200, 80) if is_ok else (55, 65, 235)
    if not is_valid_input:
        status_text = "SIN ENTRADA / OOD"
    elif is_ok:
        status_text = "OK (CONFIABLE)"
    else:
        status_text = "BAJA CONFIANZA (OOD)"

    # Banner de estado
    cv2.rectangle(panel, (14, 42), (PANEL_SIZE - 14, 82), status_color, -1)
    cv2.putText(
        panel,
        status_text,
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Predicción principal
    shown_pred = pred_char if is_valid_input else "-"
    cv2.putText(
        panel,
        f"Prediccion: '{shown_pred}'",
        (16, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Confianza:  {confidence * 100:5.1f}%  (Umbral: {threshold:.2f})",
        (16, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    # Barra de confianza con marca del umbral
    bar_x1, bar_x2, bar_y1, bar_y2 = 16, PANEL_SIZE - 16, 150, 170
    cv2.rectangle(panel, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 65), -1)
    fill_x = bar_x1 + int((bar_x2 - bar_x1) * np.clip(confidence, 0.0, 1.0))
    cv2.rectangle(panel, (bar_x1, bar_y1), (fill_x, bar_y2), status_color, -1)
    thr_x = bar_x1 + int((bar_x2 - bar_x1) * np.clip(threshold, 0.0, 1.0))
    cv2.line(panel, (thr_x, bar_y1 - 3), (thr_x, bar_y2 + 3), (0, 220, 255), 2)

    # Top-3 predicciones
    cv2.putText(
        panel,
        "Top-3 Juez (ResNet-18):",
        (16, 198),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 185),
        1,
        cv2.LINE_AA,
    )
    for i, (ch, prob) in enumerate(top3[:3]):
        cv2.putText(
            panel,
            f"  {i + 1}. '{ch}'  ->  {prob * 100:5.1f}%",
            (18, 220 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

    # FPS y latencia
    fps_color = (70, 220, 100) if fps >= 10.0 else (60, 140, 255)
    cv2.line(panel, (14, 276), (PANEL_SIZE - 14, 276), (70, 70, 75), 1)
    cv2.putText(
        panel,
        f"FPS: {fps:5.1f}  |  Latencia: {latency_ms:5.1f} ms",
        (16, 298),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        fps_color,
        1,
        cv2.LINE_AA,
    )
    extra = (
        webcam_error
        if webcam_error
        else f"Modo: {mode} | Juez: {judge_source_label} | '{target_char}'"
    )
    cv2.putText(
        panel,
        extra[:42],
        (16, 314),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (170, 170, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


class RealtimeDemo:
    """Controlador de la demo triple en tiempo real con Canvas y Webcam ROI."""

    def __init__(
        self,
        generator_onnx: Path,
        judge_onnx: Path,
        threshold: float = 0.70,
        camera_index: int = 0,
    ) -> None:
        self.gen_sess = _create_ort_session(generator_onnx)
        self.judge_sess = _create_ort_session(judge_onnx)
        self.base_font = get_default_base_font_path()

        self.threshold = float(np.clip(threshold, 0.10, 0.95))
        self.camera_index = camera_index
        self.mode = "CANVAS"  # "CANVAS" | "WEBCAM"
        self.char_idx = 0
        self.current_char = CHARSET[self.char_idx]
        self.freehand_active = False
        self.drawing = False
        self.last_pt: tuple[int, int] | None = None

        self.canvas_u8 = render_glyph(self.current_char, self.base_font, img_size=IMG_SIZE)
        self.cap: cv2.VideoCapture | None = None
        self.webcam_error: str | None = None
        self.flip_webcam: bool = False
        # AUTO: evalúa y_hat en teclado (Lato), y x normalizado directo en dibujo libre/webcam
        self.judge_mode: str = "AUTO"  # "AUTO" | "Y_HAT"
        self.ema_ms: float | None = None

    def set_char(self, ch: str) -> None:
        self.current_char = ch
        if ch in CHARSET:
            self.char_idx = CHARSET.index(ch)
        self.freehand_active = False
        self.canvas_u8 = render_glyph(ch, self.base_font, img_size=IMG_SIZE)

    def clear_canvas(self) -> None:
        self.freehand_active = True
        self.canvas_u8 = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    def toggle_mode(self) -> None:
        if self.mode == "CANVAS":
            self.mode = "WEBCAM"
            self.webcam_error = None
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.webcam_error = "Webcam no disponible (pulsa Tab/M)"
        else:
            self.mode = "CANVAS"
            self.webcam_error = None
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None

    def adjust_threshold(self, delta: float) -> None:
        self.threshold = round(float(np.clip(self.threshold + delta, 0.10, 0.95)), 2)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Permite dibujar a mano alzada en el panel 1 cuando está en modo CANVAS."""
        if self.mode != "CANVAS":
            return
        local_y = y - HEADER_H
        if not (0 <= x < PANEL_SIZE and 32 <= local_y < PANEL_SIZE):
            if event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.last_pt = None
            return

        gx = int(np.clip(x * IMG_SIZE / PANEL_SIZE, 0, IMG_SIZE - 1))
        gy = int(np.clip((local_y - 32) * IMG_SIZE / (PANEL_SIZE - 32), 0, IMG_SIZE - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.freehand_active:
                self.clear_canvas()
            self.drawing = True
            self.last_pt = (gx, gy)
            cv2.circle(self.canvas_u8, (gx, gy), 3, 0, -1, lineType=cv2.LINE_AA)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.last_pt is not None:
            cv2.line(self.canvas_u8, self.last_pt, (gx, gy), 0, thickness=5, lineType=cv2.LINE_AA)
            self.last_pt = (gx, gy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_pt = None

    def step_frame(self) -> tuple[np.ndarray, float, float]:
        """Ejecuta 1 ciclo completo (adquisición + inferencia ONNX + renderizado de ventana)."""
        t0 = time.perf_counter()

        # 1. Obtener entrada x (128x128 uint8) y panel izquierdo
        if self.mode == "WEBCAM":
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
            else:
                ok, frame = False, None

            if ok and frame is not None:
                if self.flip_webcam:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                side = min(h, w) // 2
                cx, cy = w // 2, h // 2
                x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
                x2, y2 = min(x1 + side, w), min(y1 + side, h)
                roi = frame[y1:y2, x1:x2]
                x_u8, is_valid = _preprocess_webcam_roi(roi)

                vis = frame.copy()
                roi_color = (0, 220, 90) if is_valid else (55, 65, 235)
                cv2.rectangle(vis, (x1, y1), (x2, y2), roi_color, 2)
                left_body = cv2.resize(vis, (PANEL_SIZE, PANEL_SIZE - 32))
                # Miniatura del glifo recortado y centrado (lo que realmente entra a Pix2Pix)
                thumb_sz = 96
                thumb = cv2.cvtColor(cv2.resize(x_u8, (thumb_sz, thumb_sz)), cv2.COLOR_GRAY2BGR)
                left_body[-thumb_sz - 4 : -4, -thumb_sz - 4 : -4] = thumb
                cv2.rectangle(
                    left_body,
                    (PANEL_SIZE - thumb_sz - 4, PANEL_SIZE - 32 - thumb_sz - 4),
                    (PANEL_SIZE - 4, PANEL_SIZE - 32 - 4),
                    roi_color,
                    2,
                )
            else:
                self.webcam_error = "Webcam no disponible (pulsa Tab/M)"
                x_u8 = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
                is_valid = False
                left_body = np.full((PANEL_SIZE - 32, PANEL_SIZE, 3), 40, dtype=np.uint8)
                cv2.putText(
                    left_body,
                    "Sin senal de camara",
                    (55, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (120, 120, 240),
                    2,
                    cv2.LINE_AA,
                )
            left_title = "1. Entrada x (Webcam ROI)"
        else:
            if self.freehand_active:
                x_u8, is_valid = _normalize_freehand_canvas(self.canvas_u8)
                left_body = cv2.cvtColor(
                    cv2.resize(
                        self.canvas_u8,
                        (PANEL_SIZE, PANEL_SIZE - 32),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    cv2.COLOR_GRAY2BGR,
                )
                # Miniatura del dibujo libre ya recortado y centrado en 128x128
                thumb_sz = 84
                thumb = cv2.cvtColor(cv2.resize(x_u8, (thumb_sz, thumb_sz)), cv2.COLOR_GRAY2BGR)
                thumb_color = (0, 220, 90) if is_valid else (55, 65, 235)
                left_body[-thumb_sz - 4 : -4, -thumb_sz - 4 : -4] = thumb
                cv2.rectangle(
                    left_body,
                    (PANEL_SIZE - thumb_sz - 4, PANEL_SIZE - 32 - thumb_sz - 4),
                    (PANEL_SIZE - 4, PANEL_SIZE - 32 - 4),
                    thumb_color,
                    2,
                )
                left_title = "1. Entrada x (Dibujo libre)"
            else:
                x_u8 = self.canvas_u8
                ink_ratio = float(np.mean(x_u8 < 180))
                is_valid = 0.003 <= ink_ratio <= 0.85
                left_body = cv2.cvtColor(
                    cv2.resize(x_u8, (PANEL_SIZE, PANEL_SIZE - 32), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR,
                )
                left_title = f"1. Entrada x (Canvas: '{self.current_char}')"

        # 2. Inferencia ONNX (Generador + Juez)
        x_norm = (x_u8.astype(np.float32) / 127.5) - 1.0
        x_in = x_norm[None, None, :, :]

        y_hat = self.gen_sess.run(["y_hat"], {"x": x_in})[0]
        eval_direct = (self.judge_mode == "AUTO") and (
            self.mode == "WEBCAM" or self.freehand_active
        )
        judge_input = x_in if eval_direct else y_hat
        judge_source_label = "x directo" if eval_direct else "y_hat"
        logits = self.judge_sess.run(["logits"], {"glyph": judge_input})[0][0]
        probs = _softmax(logits)

        top_indices = np.argsort(probs)[::-1][:3]
        top3 = [(IDX_TO_CHAR[int(i)], float(probs[int(i)])) for i in top_indices]
        pred_char, confidence = top3[0]
        if not is_valid:
            confidence = 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.ema_ms = elapsed_ms if self.ema_ms is None else (0.8 * self.ema_ms + 0.2 * elapsed_ms)
        fps = 1000.0 / max(self.ema_ms, 1e-6)

        # 3. Componer los 3 paneles
        panel_left = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
        _draw_panel_header(panel_left, left_title)
        panel_left[32:, :] = left_body

        y_u8 = np.clip((y_hat[0, 0] + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        panel_mid = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
        _draw_panel_header(panel_mid, "2. Glifo generado y_hat (Pix2Pix)")
        panel_mid[32:, :] = cv2.cvtColor(
            cv2.resize(y_u8, (PANEL_SIZE, PANEL_SIZE - 32), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_GRAY2BGR,
        )

        panel_right = _build_info_panel(
            pred_char=pred_char,
            confidence=confidence,
            top3=top3,
            threshold=self.threshold,
            is_valid_input=is_valid,
            latency_ms=self.ema_ms,
            fps=fps,
            mode=self.mode,
            target_char=self.current_char,
            judge_source_label=judge_source_label,
            webcam_error=self.webcam_error,
        )

        # 4. Encabezado y barra de controles inferior
        total_w = PANEL_SIZE * 3
        canvas = np.full((HEADER_H + PANEL_SIZE + FOOTER_H, total_w, 3), 18, dtype=np.uint8)
        cv2.putText(
            canvas,
            "Handwritten Font cGAN (Pix2Pix + ResNet-18 ONNX) — Tiempo Real",
            (16, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        canvas[HEADER_H : HEADER_H + PANEL_SIZE, 0:PANEL_SIZE] = panel_left
        canvas[HEADER_H : HEADER_H + PANEL_SIZE, PANEL_SIZE : PANEL_SIZE * 2] = panel_mid
        canvas[HEADER_H : HEADER_H + PANEL_SIZE, PANEL_SIZE * 2 : total_w] = panel_right

        # Separadores verticales
        cv2.line(canvas, (PANEL_SIZE, HEADER_H), (PANEL_SIZE, HEADER_H + PANEL_SIZE), (75, 75, 80), 2)
        cv2.line(
            canvas,
            (PANEL_SIZE * 2, HEADER_H),
            (PANEL_SIZE * 2, HEADER_H + PANEL_SIZE),
            (75, 75, 80),
            2,
        )

        footer_y = HEADER_H + PANEL_SIZE + 22
        cv2.putText(
            canvas,
            "[Teclas] Letra/digito | Mouse: dibuja | Tab: Canvas/Webcam | Enter: Juez x/y_hat | +/-: Umbral | Backspace: Borrar | ESC: Salir",
            (14, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Tip WEBCAM: 'M'/'Tab' alterna modo | 'F' voltea espejo | Centra 1 letra grande en el cuadro | Ctrl+S guarda captura en results/",
            (14, footer_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (150, 155, 165),
            1,
            cv2.LINE_AA,
        )
        return canvas, self.ema_ms, fps

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        try:
            while True:
                frame, _, _ = self.step_frame()
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                if key == 27:  # ESC
                    break
                if key == 9:  # Tab -> alternar modo siempre
                    self.toggle_mode()
                elif key == 13:  # Enter -> alternar fuente del Juez (AUTO vs siempre y_hat)
                    self.judge_mode = "Y_HAT" if self.judge_mode == "AUTO" else "AUTO"
                elif key == 8:  # Backspace -> limpiar canvas
                    self.clear_canvas()
                elif key == 32:  # Space -> siguiente carácter del charset
                    self.char_idx = (self.char_idx + 1) % len(CHARSET)
                    self.set_char(CHARSET[self.char_idx])
                elif key in (ord("+"), ord("=")):
                    self.adjust_threshold(+0.05)
                elif key in (ord("-"), ord("_")):
                    self.adjust_threshold(-0.05)
                elif self.mode == "WEBCAM" and key in (ord("m"), ord("M")):
                    self.toggle_mode()
                elif self.mode == "WEBCAM" and key in (ord("f"), ord("F")):
                    self.flip_webcam = not self.flip_webcam
                elif key == 19:  # Ctrl+S -> guardar captura
                    out_p = ROOT_DIR / "results" / "demo_capture.png"
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    ok_enc, buf = cv2.imencode(".png", frame)
                    if ok_enc:
                        buf.tofile(str(out_p))
                        print(f"[OK] Captura guardada en: {out_p}")
                elif 32 < key < 127:
                    ch = chr(key)
                    if ch in ("M",) and self.freehand_active:
                        self.toggle_mode()
                    else:
                        self.set_char(ch)
        finally:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo OpenCV en tiempo real (ONNX).")
    parser.add_argument(
        "--generator-onnx",
        type=Path,
        default=ROOT_DIR / "checkpoints" / "generator.onnx",
    )
    parser.add_argument(
        "--judge-onnx",
        type=Path,
        default=ROOT_DIR / "checkpoints" / "judge_resnet18.onnx",
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--headless-test",
        action="store_true",
        help="Ejecuta 25 cuadros sin abrir ventana, guarda captura de prueba y verifica >=10 FPS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = RealtimeDemo(
        generator_onnx=args.generator_onnx,
        judge_onnx=args.judge_onnx,
        threshold=args.threshold,
        camera_index=args.camera,
    )

    if args.headless_test:
        test_chars = ["A", "B", "k", "4", "?", "G"]
        last_frame = None
        last_ms = 0.0
        last_fps = 0.0
        for i in range(25):
            demo.set_char(test_chars[i % len(test_chars)])
            last_frame, last_ms, last_fps = demo.step_frame()
        # Probar también entrada vacía (OOD) y ajuste de umbral
        demo.clear_canvas()
        demo.adjust_threshold(+0.05)
        ood_frame, _, _ = demo.step_frame()

        out_dir = ROOT_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        assert last_frame is not None
        cv2.imencode(".png", last_frame)[1].tofile(str(out_dir / "demo_headless_ok.png"))
        cv2.imencode(".png", ood_frame)[1].tofile(str(out_dir / "demo_headless_ood.png"))
        print(
            f"[OK] Headless test (25 cuadros) | Latencia EMA: {last_ms:.2f} ms | "
            f"FPS: {last_fps:.2f} | Target >=10 FPS: {'CUMPLE' if last_fps >= 10.0 else 'NO CUMPLE'}"
        )
        return

    demo.run()


if __name__ == "__main__":
    main()
