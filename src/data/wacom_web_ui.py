"""Interfaz web local de captura fluida con tableta Wacom (estilo OneNote).

Usa Pointer Events nativos del navegador (`getCoalescedEvents`, presión del lápiz,
`desynchronized: true` e interpolación cuadrática Bézier + estabilizador) para
obtener trazos suaves de baja latencia sin dependencias externas adicionales.

Guarda exactamente el mismo formato que el modo OpenCV:
- Celdas individuales: `data/raw_scans/_wacom_cells/cell_XXXX.png` (228x228)
- Hojas compuestas: `data/raw_scans/hoja_01.png` … `hoja_07.png`
"""

from __future__ import annotations

import base64
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from src.data.glyph_render import get_default_base_font_path, render_glyph

EXTRACTOR_INNER_MARGIN = 14

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Captura Wacom Fluida — 300 Glifos</title>
<style>
  :root {
    --bg: #15171e;
    --panel: #1e212b;
    --panel-border: #2e3342;
    --text: #eef1f7;
    --muted: #98a2b8;
    --accent: #4c8dff;
    --green: #2fa866;
    --green-hover: #37be74;
    --red: #d9534f;
    --amber: #e2a738;
    --btn-bg: #2a2f3d;
    --btn-hover: #363c4f;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  /* Header */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 18px;
    background: #101218;
    border-bottom: 1px solid var(--panel-border);
    flex-shrink: 0;
  }
  .header-info {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    font-size: 13.5px;
  }
  .pill {
    background: #232734;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--panel-border);
    font-weight: 600;
    font-size: 12.5px;
  }
  .pill-ok { border-color: #2c6d49; color: #7ae2a6; background: #172b20; }
  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .progress-track {
    height: 4px;
    background: #222633;
    width: 100%;
    flex-shrink: 0;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #35b46e, #5ad48e);
    width: 0%;
    transition: width 0.2s ease;
  }

  /* Main layout */
  .main {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* Left sidebar */
  .sidebar {
    width: 310px;
    flex-shrink: 0;
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }
  .split-badge {
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 6px 10px;
    border-radius: 7px;
    color: #fff;
  }
  .split-train { background: #2d7a4f; }
  .split-val { background: #a87b22; }
  .split-test_unseen { background: #2c6ec4; }

  .ref-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #171a23;
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 10px;
  }
  .ref-img {
    width: 88px;
    height: 88px;
    background: #fff;
    border-radius: 6px;
    border: 1px solid #3c4254;
    object-fit: contain;
    flex-shrink: 0;
  }
  .ref-meta {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }
  .char-title {
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    line-height: 1.25;
  }
  .char-sub {
    font-size: 12px;
    color: var(--muted);
  }
  .saved-tag {
    display: inline-block;
    margin-top: 2px;
    font-size: 11.5px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 5px;
    width: fit-content;
  }
  .saved-yes { background: #183625; color: #7be3a6; }
  .saved-no { background: #282c3a; color: #b5bfd4; }

  .prev-box {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #171a23;
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 8px 10px;
  }
  .prev-box img {
    width: 64px;
    height: 64px;
    background: #fff;
    border-radius: 5px;
    border: 1px solid #3c4254;
    object-fit: contain;
  }
  .prev-box .prev-text {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.35;
  }

  .card {
    background: #171a23;
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .card-title {
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted);
  }
  .control-row {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .control-label {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    color: #dce2ef;
  }
  .control-label span.val {
    color: #82b4ff;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
    margin: 0;
  }
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12.5px;
    color: #dce2ef;
    cursor: pointer;
  }
  .toggle-row input {
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
    cursor: pointer;
  }

  /* Center workspace */
  .stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 16px;
    min-width: 0;
    position: relative;
  }

  .canvas-frame {
    position: relative;
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.45), 0 0 0 2px #4a6b9c;
    overflow: hidden;
    touch-action: none;
    flex-shrink: 0;
  }
  .canvas-frame canvas {
    display: block;
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    touch-action: none;
  }
  #guideCanvas {
    pointer-events: none;
    z-index: 1;
  }
  #inkCanvas {
    z-index: 2;
    cursor: crosshair;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
  }
  button {
    border: 1px solid var(--panel-border);
    background: var(--btn-bg);
    color: var(--text);
    padding: 9px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.12s ease, transform 0.06s ease;
    font-family: inherit;
  }
  button:hover { background: var(--btn-hover); }
  button:active { transform: translateY(1px); }
  button.primary {
    background: var(--green);
    border-color: #39c377;
    color: #fff;
    padding: 9px 20px;
  }
  button.primary:hover { background: var(--green-hover); }
  button.danger { background: #5a272b; border-color: #8c3d43; }
  button.danger:hover { background: #733136; }
  button.amber { background: #6b501c; border-color: #9e772b; }
  button.amber:hover { background: #836222; }
  button.small { padding: 6px 10px; font-size: 12px; }

  .hint-bar {
    font-size: 12px;
    color: var(--muted);
    text-align: center;
    max-width: 720px;
  }
  .status-toast {
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: #1d2738;
    border: 1px solid #3c5884;
    color: #cfe3ff;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
    z-index: 10;
    white-space: nowrap;
  }
  .status-toast.show { opacity: 1; }
  .status-toast.ok { background: #163322; border-color: #2f7d4f; color: #8af0b4; }
  .status-toast.warn { background: #3b2222; border-color: #8f4242; color: #ffadad; }

  .jump-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }
  .jump-row input[type="number"] {
    width: 68px;
    background: #101218;
    border: 1px solid var(--panel-border);
    color: var(--text);
    border-radius: 6px;
    padding: 5px 7px;
    font-size: 12.5px;
  }
  .pen-indicator {
    font-size: 11.5px;
    color: #7ee3a8;
    text-align: center;
    min-height: 16px;
  }
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="header-info">
      <strong style="font-size:14.5px;">Captura Wacom Fluida</strong>
      <span class="pill" id="hdrGlyph">Glifo 1 / 300</span>
      <span class="pill" id="hdrPage">Hoja 1 / 7 · Rep 1 / 4</span>
      <span class="pill pill-ok" id="hdrDone">Completados: 0 / 300</span>
    </div>
    <div class="header-actions">
      <button class="small" id="btnNextMissing" title="Saltar al primer glifo pendiente">Ir a pendiente</button>
      <button class="small amber" id="btnExport" title="Exportar las 7 hojas (E)">Exportar hojas (E)</button>
      <button class="small danger" id="btnQuit" title="Guardar, exportar y cerrar servidor">Terminar</button>
    </div>
  </div>
  <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>

  <div class="main">
    <aside class="sidebar">
      <div class="split-badge split-train" id="splitBadge">ENTRENAMIENTO (train)</div>

      <div class="ref-row">
        <img class="ref-img" id="refImg" alt="Referencia" />
        <div class="ref-meta">
          <div class="char-title" id="charDesc">Mayúscula 'A'</div>
          <div class="char-sub" id="charCode">Muestra #0 · Repetición 1/4</div>
          <span class="saved-tag saved-no" id="savedTag">Pendiente</span>
        </div>
      </div>

      <div class="prev-box" id="prevBox" style="display:none;">
        <img id="prevImg" alt="Repetición anterior" />
        <div class="prev-text">
          <strong style="color:#dce2ef;">Tu trazo anterior</strong><br />
          Intenta mantener tamaño, inclinación y estilo parecidos.
        </div>
      </div>

      <div class="card">
        <div class="card-title">Ajustes de tinta y lienzo</div>

        <div class="control-row">
          <div class="control-label">
            <span>Grosor de pluma</span>
            <span class="val" id="brushVal">6.5 px</span>
          </div>
          <input type="range" id="brushSlider" min="3.5" max="16" step="0.5" value="6.5" />
        </div>

        <div class="control-row">
          <div class="control-label">
            <span>Tamaño del recuadro</span>
            <span class="val" id="sizeVal">320 px</span>
          </div>
          <input type="range" id="sizeSlider" min="200" max="540" step="10" value="320" />
        </div>

        <div class="control-row">
          <div class="control-label">
            <span>Suavizado (estabilizador)</span>
            <span class="val" id="smoothVal">45 %</span>
          </div>
          <input type="range" id="smoothSlider" min="0" max="100" step="5" value="45" />
        </div>

        <label class="toggle-row">
          <span>Presión del lápiz (Wacom)</span>
          <input type="checkbox" id="pressureToggle" checked />
        </label>

        <label class="toggle-row">
          <span>Mostrar guías de renglón</span>
          <input type="checkbox" id="guidesToggle" checked />
        </label>
      </div>

      <div class="card">
        <div class="card-title">Navegación rápida</div>
        <div class="jump-row">
          <span style="font-size:12.5px;color:var(--muted);">Ir al #</span>
          <input type="number" id="jumpInput" min="1" max="300" value="1" />
          <button class="small" id="btnJump">Ir</button>
        </div>
        <div style="font-size:11.5px;color:var(--muted);line-height:1.4;">
          Apoya letras sobre la <strong>línea base naranja</strong>; colitas de <code>g, j, p, q, y</code> y comas bajan de la línea.
        </div>
      </div>
    </aside>

    <main class="stage">
      <div class="status-toast" id="toast">Listo</div>

      <div class="canvas-frame" id="canvasFrame" style="width:320px;height:320px;">
        <canvas id="guideCanvas" width="912" height="912"></canvas>
        <canvas id="inkCanvas" width="912" height="912"></canvas>
      </div>

      <div class="pen-indicator" id="penIndicator"></div>

      <div class="toolbar">
        <button id="btnPrev" title="Glifo anterior (←)">← Anterior</button>
        <button id="btnUndo" title="Deshacer último trazo (Z)">Deshacer (Z)</button>
        <button class="danger" id="btnClear" title="Limpiar lienzo (R)">Limpiar (R)</button>
        <button class="primary" id="btnSaveNext" title="Guardar y avanzar (Espacio / Enter)">Guardar y Sig. (Espacio)</button>
        <button id="btnSkip" title="Saltar sin guardar (→)">Saltar →</button>
      </div>

      <div class="hint-bar">
        Atajos: <strong>Espacio / Enter</strong> Guardar y sig. · <strong>Z</strong> Deshacer · <strong>R</strong> Limpiar ·
        <strong>← / →</strong> Navegar · <strong>+ / -</strong> Grosor · <strong>[ / ]</strong> Tamaño recuadro ·
        Botón derecho / borrador del lápiz = Deshacer
      </div>
    </main>
  </div>
</div>

<script>
(() => {
  const INTERNAL_SIZE = 912; // 4x de box_size=228 para trazos ultra nítidos
  const TEMPLATE_BOX = 228;
  const SCALE_TO_INTERNAL = INTERNAL_SIZE / TEMPLATE_BOX; // 4.0

  // Estado global
  let state = {
    cells: [],
    idx: 0,
    completed: 0,
    boxSize: 228,
    baselineRatio: 0.72,
    innerMargin: 14,
  };

  // Config persistente en localStorage
  const LS_KEY = "wacom_smooth_capture_prefs_v1";
  let prefs = {
    brushTplPx: 6.5,
    canvasCssPx: 320,
    smoothing: 45,
    usePressure: true,
    showGuides: true,
  };

  // Trazos del glifo actual
  let strokes = []; // Cada trazo: { points: [{x, y, p}], baseWidth: number, usePressure: boolean }
  let currentStroke = null;
  let loadedBgImage = null; // Si el glifo ya tenía PNG previo guardado
  let isDrawing = false;
  let activePointerId = null;
  let dirty = false;
  let saving = false;

  // Capa cacheada de trazos ya terminados para 120+ FPS sin lag durante pointermove
  const committedCanvas = document.createElement("canvas");
  committedCanvas.width = INTERNAL_SIZE;
  committedCanvas.height = INTERNAL_SIZE;
  const committedCtx = committedCanvas.getContext("2d");

  // DOM
  const frameEl = document.getElementById("canvasFrame");
  const guideCanvas = document.getElementById("guideCanvas");
  const inkCanvas = document.getElementById("inkCanvas");
  const guideCtx = guideCanvas.getContext("2d");
  const inkCtx = inkCanvas.getContext("2d", { desynchronized: true }) || inkCanvas.getContext("2d");

  const hdrGlyph = document.getElementById("hdrGlyph");
  const hdrPage = document.getElementById("hdrPage");
  const hdrDone = document.getElementById("hdrDone");
  const progressFill = document.getElementById("progressFill");
  const splitBadge = document.getElementById("splitBadge");
  const refImg = document.getElementById("refImg");
  const charDesc = document.getElementById("charDesc");
  const charCode = document.getElementById("charCode");
  const savedTag = document.getElementById("savedTag");
  const prevBox = document.getElementById("prevBox");
  const prevImg = document.getElementById("prevImg");
  const jumpInput = document.getElementById("jumpInput");
  const penIndicator = document.getElementById("penIndicator");
  const toastEl = document.getElementById("toast");

  const brushSlider = document.getElementById("brushSlider");
  const brushVal = document.getElementById("brushVal");
  const sizeSlider = document.getElementById("sizeSlider");
  const sizeVal = document.getElementById("sizeVal");
  const smoothSlider = document.getElementById("smoothSlider");
  const smoothVal = document.getElementById("smoothVal");
  const pressureToggle = document.getElementById("pressureToggle");
  const guidesToggle = document.getElementById("guidesToggle");

  let toastTimer = null;
  function showToast(msg, kind = "", ms = 2200) {
    toastEl.textContent = msg;
    toastEl.className = "status-toast show" + (kind ? " " + kind : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastEl.classList.remove("show");
    }, ms);
  }

  function loadPrefs(defaults) {
    prefs.brushTplPx = defaults.brushTplPx;
    prefs.canvasCssPx = defaults.canvasCssPx;
    try {
      const saved = JSON.parse(localStorage.getItem(LS_KEY) || "null");
      if (saved && typeof saved === "object") {
        if (typeof saved.brushTplPx === "number") prefs.brushTplPx = saved.brushTplPx;
        if (typeof saved.canvasCssPx === "number") prefs.canvasCssPx = saved.canvasCssPx;
        if (typeof saved.smoothing === "number") prefs.smoothing = saved.smoothing;
        if (typeof saved.usePressure === "boolean") prefs.usePressure = saved.usePressure;
        if (typeof saved.showGuides === "boolean") prefs.showGuides = saved.showGuides;
      }
    } catch (_) {}
    syncPrefControls();
  }

  function savePrefs() {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(prefs));
    } catch (_) {}
  }

  function syncPrefControls() {
    prefs.brushTplPx = Math.max(3.5, Math.min(16, Math.round(prefs.brushTplPx * 2) / 2));
    prefs.canvasCssPx = Math.max(200, Math.min(540, Math.round(prefs.canvasCssPx / 10) * 10));
    prefs.smoothing = Math.max(0, Math.min(100, Math.round(prefs.smoothing / 5) * 5));

    brushSlider.value = String(prefs.brushTplPx);
    brushVal.textContent = `${prefs.brushTplPx.toFixed(1)} px`;
    sizeSlider.value = String(prefs.canvasCssPx);
    sizeVal.textContent = `${prefs.canvasCssPx} px`;
    smoothSlider.value = String(prefs.smoothing);
    smoothVal.textContent = `${prefs.smoothing} %`;
    pressureToggle.checked = prefs.usePressure;
    guidesToggle.checked = prefs.showGuides;

    frameEl.style.width = `${prefs.canvasCssPx}px`;
    frameEl.style.height = `${prefs.canvasCssPx}px`;
    drawGuides();
  }

  // ---- Guías visuales (nunca se exportan) ----
  function drawGuides() {
    const ctx = guideCtx;
    const S = INTERNAL_SIZE;
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, S, S);

    if (!prefs.showGuides) return;

    const safe = state.innerMargin * (S / state.boxSize);
    const yBase = Math.round(S * state.baselineRatio);
    const yCap = Math.round(S * 0.22);
    const yX = Math.round(S * 0.44);
    const xMid = Math.round(S * 0.5);

    // Margen seguro interno
    ctx.strokeStyle = "#e6ebf3";
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.strokeRect(safe, safe, S - 2 * safe, S - 2 * safe);

    // Guías tenues de altura mayúsculas, minúsculas y centro
    ctx.strokeStyle = "#e8ecf2";
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 12]);
    ctx.beginPath();
    ctx.moveTo(safe, yCap); ctx.lineTo(S - safe, yCap);
    ctx.moveTo(safe, yX);   ctx.lineTo(S - safe, yX);
    ctx.moveTo(xMid, safe); ctx.lineTo(xMid, S - safe);
    ctx.stroke();

    // Línea base (renglón)
    ctx.setLineDash([]);
    ctx.strokeStyle = "#eba878";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(safe, yBase);
    ctx.lineTo(S - safe, yBase);
    ctx.stroke();

    // Etiqueta discreta
    ctx.fillStyle = "#d49568";
    ctx.font = "600 22px Segoe UI, sans-serif";
    ctx.fillText("línea base", safe + 10, yBase - 10);
  }

  // ---- Geometría y suavizado estilo OneNote ----
  function eventToCanvasPoint(e) {
    const rect = inkCanvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * INTERNAL_SIZE;
    const y = ((e.clientY - rect.top) / rect.height) * INTERNAL_SIZE;
    let p = 0.5;
    if (e.pointerType === "pen") {
      p = typeof e.pressure === "number" && e.pressure > 0 ? e.pressure : 0.5;
    } else {
      p = 0.5;
    }
    return {
      x: Math.max(0, Math.min(INTERNAL_SIZE, x)),
      y: Math.max(0, Math.min(INTERNAL_SIZE, y)),
      p,
      isPen: e.pointerType === "pen",
    };
  }

  function strokeWidthAt(baseWidth, pressure, usePressure) {
    if (!usePressure) return baseWidth;
    // Rango sutil estilo OneNote: mantiene legibilidad en binarización sin perder respuesta táctil
    const factor = 0.72 + Math.min(1, Math.max(0, pressure)) * 0.48;
    return baseWidth * factor;
  }

  function chaikinSmooth(points, passes) {
    if (points.length < 3 || passes <= 0) return points;
    let pts = points;
    for (let k = 0; k < passes; k++) {
      const next = [pts[0]];
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i];
        const b = pts[i + 1];
        next.push({
          x: 0.75 * a.x + 0.25 * b.x,
          y: 0.75 * a.y + 0.25 * b.y,
          p: 0.75 * a.p + 0.25 * b.p,
        });
        next.push({
          x: 0.25 * a.x + 0.75 * b.x,
          y: 0.25 * a.y + 0.75 * b.y,
          p: 0.25 * a.p + 0.75 * b.p,
        });
      }
      next.push(pts[pts.length - 1]);
      pts = next;
    }
    return pts;
  }

  function renderSingleStroke(ctx, stroke) {
    const pts = stroke.points;
    if (!pts || pts.length === 0) return;

    ctx.fillStyle = "#000000";
    ctx.strokeStyle = "#000000";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Un solo toque (punto '.', 'i', '!', ':', ';'): garantizar área mínima segura para OpenCV
    if (pts.length === 1) {
      const w = Math.max(stroke.baseWidth, 5.2 * SCALE_TO_INTERNAL);
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, w * 0.52, 0, Math.PI * 2);
      ctx.fill();
      return;
    }

    if (!stroke.usePressure) {
      // Curva cuadrática Bézier continua ultra suave con ancho uniforme
      ctx.lineWidth = stroke.baseWidth;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      if (pts.length === 2) {
        ctx.lineTo(pts[1].x, pts[1].y);
      } else {
        for (let i = 1; i < pts.length - 1; i++) {
          const midX = (pts[i].x + pts[i + 1].x) * 0.5;
          const midY = (pts[i].y + pts[i + 1].y) * 0.5;
          ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
        }
        const last = pts[pts.length - 1];
        ctx.lineTo(last.x, last.y);
      }
      ctx.stroke();
      return;
    }

    // Trazo suave con presión variable interpolando a lo largo de Bézier cuadrática
    let prevX = pts[0].x;
    let prevY = pts[0].y;
    let prevP = pts[0].p;

    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = i === 0 ? pts[0] : {
        x: (pts[i - 1].x + pts[i].x) * 0.5,
        y: (pts[i - 1].y + pts[i].y) * 0.5,
        p: (pts[i - 1].p + pts[i].p) * 0.5,
      };
      const ctrl = pts[i];
      const p2 = i === pts.length - 2 ? pts[pts.length - 1] : {
        x: (pts[i].x + pts[i + 1].x) * 0.5,
        y: (pts[i].y + pts[i + 1].y) * 0.5,
        p: (pts[i].p + pts[i + 1].p) * 0.5,
      };

      const dist = Math.hypot(p2.x - p0.x, p2.y - p0.y) + Math.hypot(ctrl.x - p0.x, ctrl.y - p0.y);
      const steps = Math.max(1, Math.min(14, Math.ceil(dist / 5.0)));

      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        const omt = 1 - t;
        const x = omt * omt * p0.x + 2 * omt * t * ctrl.x + t * t * p2.x;
        const y = omt * omt * p0.y + 2 * omt * t * ctrl.y + t * t * p2.y;
        const p = omt * omt * p0.p + 2 * omt * t * ctrl.p + t * t * p2.p;
        const w = strokeWidthAt(stroke.baseWidth, (prevP + p) * 0.5, true);

        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();

        prevX = x;
        prevY = y;
        prevP = p;
      }
    }
  }

  function rebuildCommittedLayer() {
    committedCtx.clearRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    if (loadedBgImage) {
      committedCtx.drawImage(loadedBgImage, 0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    }
    for (const s of strokes) {
      renderSingleStroke(committedCtx, s);
    }
  }

  function redrawInkCanvas(rebuildCommitted = true) {
    if (rebuildCommitted) {
      rebuildCommittedLayer();
    }
    inkCtx.clearRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    inkCtx.drawImage(committedCanvas, 0, 0);
    if (currentStroke) {
      renderSingleStroke(inkCtx, currentStroke);
    }
  }

  // Hacer transparente el blanco del PNG previo para que se sigan viendo las guías
  function makeWhiteTransparentCanvas(img) {
    const off = document.createElement("canvas");
    off.width = INTERNAL_SIZE;
    off.height = INTERNAL_SIZE;
    const octx = off.getContext("2d");
    octx.drawImage(img, 0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    const imgData = octx.getImageData(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const lum = d[i]; // escala de grises
      if (lum >= 245) {
        d[i + 3] = 0;
      } else {
        d[i] = 0; d[i + 1] = 0; d[i + 2] = 0;
        d[i + 3] = Math.min(255, Math.round((255 - lum) * 1.15));
      }
    }
    octx.putImageData(imgData, 0, 0);
    return off;
  }

  function hasAnyInk() {
    return Boolean(loadedBgImage) || strokes.length > 0;
  }

  function exportCleanPngDataUrl() {
    const off = document.createElement("canvas");
    off.width = INTERNAL_SIZE;
    off.height = INTERNAL_SIZE;
    const octx = off.getContext("2d");
    octx.fillStyle = "#ffffff";
    octx.fillRect(0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    if (loadedBgImage) {
      octx.drawImage(loadedBgImage, 0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    }
    for (const s of strokes) {
      renderSingleStroke(octx, s);
    }
    return off.toDataURL("image/png");
  }

  // ---- Eventos Pointer (Wacom / Windows Ink / Mouse) ----
  inkCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

  inkCanvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();

    // Borrador del lápiz Wacom (button 5) o clic derecho (button 2) -> Deshacer
    if (e.button === 5 || e.button === 2 || (e.buttons & 32)) {
      actionUndo();
      return;
    }
    if (e.button !== 0) return;

    inkCanvas.setPointerCapture(e.pointerId);
    activePointerId = e.pointerId;
    isDrawing = true;

    const pt = eventToCanvasPoint(e);
    if (pt.isPen) {
      penIndicator.textContent = "Lápiz Wacom detectado (Windows Ink activo)";
    }

    currentStroke = {
      points: [{ x: pt.x, y: pt.y, p: pt.p }],
      baseWidth: prefs.brushTplPx * SCALE_TO_INTERNAL,
      usePressure: prefs.usePressure && pt.isPen,
    };
    dirty = true;
    redrawInkCanvas(false);
  });

  inkCanvas.addEventListener("pointermove", (e) => {
    if (!isDrawing || e.pointerId !== activePointerId || !currentStroke) return;
    e.preventDefault();

    const events = typeof e.getCoalescedEvents === "function" ? e.getCoalescedEvents() : [e];
    const smoothRatio = prefs.smoothing / 100.0; // 0..1
    // Factor EMA: 0% -> alpha=1.0 (sin filtro), 100% -> alpha=0.22 (muy estabilizado)
    const alpha = 1.0 - smoothRatio * 0.78;
    const minDist = 1.4 + smoothRatio * 2.4;

    let added = false;
    for (const ev of events) {
      const raw = eventToCanvasPoint(ev);
      const pts = currentStroke.points;
      const last = pts[pts.length - 1];

      const sx = last.x + (raw.x - last.x) * alpha;
      const sy = last.y + (raw.y - last.y) * alpha;
      const sp = last.p + (raw.p - last.p) * 0.35;

      if (Math.hypot(sx - last.x, sy - last.y) >= minDist) {
        pts.push({ x: sx, y: sy, p: sp });
        added = true;
      }
    }

    if (added) {
      redrawInkCanvas(false);
    }
  });

  function finishStroke(e) {
    if (!isDrawing || (activePointerId !== null && e.pointerId !== activePointerId)) return;
    isDrawing = false;
    activePointerId = null;
    if (currentStroke) {
      // Capturar también el punto exacto donde se levantó el lápiz para no acortar trazos rápidos
      const endPt = eventToCanvasPoint(e);
      const pts = currentStroke.points;
      const last = pts[pts.length - 1];
      if (Math.hypot(endPt.x - last.x, endPt.y - last.y) >= 1.0) {
        pts.push({ x: endPt.x, y: endPt.y, p: last.p * 0.85 });
      }
      // Suavizado ligero post-trazo (efecto OneNote al levantar el lápiz)
      const passes = prefs.smoothing >= 65 ? 2 : (prefs.smoothing >= 20 ? 1 : 0);
      if (currentStroke.points.length >= 4 && passes > 0) {
        currentStroke.points = chaikinSmooth(currentStroke.points, passes);
      }
      strokes.push(currentStroke);
      currentStroke = null;
      redrawInkCanvas(true);
    }
  }

  inkCanvas.addEventListener("pointerup", finishStroke);
  inkCanvas.addEventListener("pointercancel", finishStroke);

  // ---- Carga y navegación de celdas ----
  function updateHeaderAndMeta() {
    const cell = state.cells[state.idx];
    const total = state.cells.length;
    hdrGlyph.textContent = `Glifo ${state.idx + 1} / ${total}`;
    hdrPage.textContent = `Hoja ${cell.page_index} / 7 · Rep ${cell.repetition} / 4`;
    hdrDone.textContent = `Completados: ${state.completed} / ${total}`;
    progressFill.style.width = `${(state.completed / Math.max(1, total)) * 100}%`;
    jumpInput.value = String(state.idx + 1);

    // Split badge
    splitBadge.className = "split-badge split-" + cell.split;
    if (cell.split === "test_unseen") splitBadge.textContent = "TEST UNSEEN (Zero-Shot)";
    else if (cell.split === "val") splitBadge.textContent = "VALIDACIÓN (val)";
    else splitBadge.textContent = "ENTRENAMIENTO (train)";

    charDesc.textContent = cell.desc;
    charCode.textContent = `Muestra #${cell.sample_index} · Repetición ${cell.repetition}/4`;
    if (cell.saved) {
      savedTag.textContent = "GUARDADO";
      savedTag.className = "saved-tag saved-yes";
    } else {
      savedTag.textContent = "Pendiente";
      savedTag.className = "saved-tag saved-no";
    }

    const cacheBust = Date.now();
    refImg.src = `/api/ref/${state.idx}`;

    // Vista previa de repetición anterior
    fetch(`/api/prev_rep/${state.idx}?t=${cacheBust}`).then(async (res) => {
      if (res.status === 200) {
        const blob = await res.blob();
        prevImg.src = URL.createObjectURL(blob);
        prevBox.style.display = "flex";
      } else {
        prevBox.style.display = "none";
      }
    }).catch(() => { prevBox.style.display = "none"; });
  }

  async function loadCell(index) {
    state.idx = Math.max(0, Math.min(state.cells.length - 1, index));
    strokes = [];
    currentStroke = null;
    loadedBgImage = null;
    dirty = false;
    updateHeaderAndMeta();
    redrawInkCanvas();

    try {
      const res = await fetch(`/api/cell/${state.idx}?t=${Date.now()}`);
      if (res.status === 200) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          loadedBgImage = makeWhiteTransparentCanvas(img);
          URL.revokeObjectURL(url);
          redrawInkCanvas();
        };
        img.src = url;
      }
    } catch (_) {}
  }

  async function saveCurrentCell() {
    if (!hasAnyInk()) {
      showToast("Lienzo vacío: dibuja el carácter antes de guardar.", "warn");
      return false;
    }
    if (saving) return false;
    saving = true;
    try {
      const dataUrl = exportCleanPngDataUrl();
      const res = await fetch(`/api/cell/${state.idx}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        showToast(data.error || "Error al guardar el trazo.", "warn");
        return false;
      }
      state.cells[state.idx].saved = true;
      state.completed = data.completed;
      dirty = false;
      return true;
    } catch (err) {
      showToast("Error de conexión al guardar.", "warn");
      return false;
    } finally {
      saving = false;
    }
  }

  // ---- Acciones ----
  async function actionSaveAndNext() {
    const ok = await saveCurrentCell();
    if (!ok) return;
    if (state.idx < state.cells.length - 1) {
      await loadCell(state.idx + 1);
      if (state.completed === state.cells.length) {
        showToast("¡300/300 completos! Exportando hojas...", "ok", 3500);
        await actionExport();
      } else {
        showToast(`Guardado (${state.completed}/${state.cells.length})`, "ok", 1200);
      }
    } else {
      updateHeaderAndMeta();
      await actionExport();
    }
  }

  async function actionPrev() {
    if (dirty && hasAnyInk()) {
      await saveCurrentCell();
    }
    if (state.idx > 0) {
      await loadCell(state.idx - 1);
    }
  }

  async function actionSkip() {
    if (state.idx < state.cells.length - 1) {
      await loadCell(state.idx + 1);
    }
  }

  function actionUndo() {
    if (strokes.length > 0) {
      strokes.pop();
      dirty = true;
      redrawInkCanvas();
      showToast("Trazo deshecho", "", 900);
    } else if (loadedBgImage) {
      loadedBgImage = null;
      dirty = true;
      redrawInkCanvas();
      showToast("Trazo anterior limpiado", "", 900);
    } else {
      showToast("No hay trazos para deshacer", "", 900);
    }
  }

  function actionClear() {
    strokes = [];
    currentStroke = null;
    loadedBgImage = null;
    dirty = true;
    redrawInkCanvas();
    showToast("Lienzo limpiado", "", 900);
  }

  async function actionExport() {
    if (dirty && hasAnyInk()) {
      await saveCurrentCell();
    }
    try {
      const res = await fetch("/api/export", { method: "POST" });
      const data = await res.json();
      showToast(`Exportadas ${data.pages} hojas (${data.placed}/${data.total} glifos)`, "ok", 3000);
    } catch (_) {
      showToast("No se pudo exportar.", "warn");
    }
  }

  async function actionQuit() {
    if (dirty && hasAnyInk()) {
      await saveCurrentCell();
    }
    try {
      const res = await fetch("/api/shutdown", { method: "POST" });
      const data = await res.json();
      document.body.innerHTML = `
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;gap:12px;text-align:center;padding:24px;">
          <h2 style="margin:0;color:#7be3a6;">Sesión guardada y hojas exportadas</h2>
          <p style="margin:0;color:#b8c2d6;">Se guardaron <strong>${data.placed}/${data.total}</strong> glifos en <code>data/raw_scans/</code> (${data.pages} hojas).</p>
          <p style="margin:0;color:#8e98ad;font-size:13px;">Ya puedes cerrar esta pestaña y revisar tu terminal.</p>
        </div>
      `;
    } catch (_) {
      window.close();
    }
  }

  // ---- Controles UI ----
  brushSlider.addEventListener("input", () => {
    prefs.brushTplPx = parseFloat(brushSlider.value);
    syncPrefControls();
    savePrefs();
  });
  sizeSlider.addEventListener("input", () => {
    prefs.canvasCssPx = parseInt(sizeSlider.value, 10);
    syncPrefControls();
    savePrefs();
  });
  smoothSlider.addEventListener("input", () => {
    prefs.smoothing = parseInt(smoothSlider.value, 10);
    syncPrefControls();
    savePrefs();
  });
  pressureToggle.addEventListener("change", () => {
    prefs.usePressure = pressureToggle.checked;
    savePrefs();
  });
  guidesToggle.addEventListener("change", () => {
    prefs.showGuides = guidesToggle.checked;
    drawGuides();
    savePrefs();
  });

  document.getElementById("btnPrev").addEventListener("click", actionPrev);
  document.getElementById("btnUndo").addEventListener("click", actionUndo);
  document.getElementById("btnClear").addEventListener("click", actionClear);
  document.getElementById("btnSaveNext").addEventListener("click", actionSaveAndNext);
  document.getElementById("btnSkip").addEventListener("click", actionSkip);
  document.getElementById("btnExport").addEventListener("click", actionExport);
  document.getElementById("btnQuit").addEventListener("click", actionQuit);

  document.getElementById("btnJump").addEventListener("click", () => {
    const v = parseInt(jumpInput.value, 10);
    if (!Number.isNaN(v)) loadCell(v - 1);
  });
  jumpInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.stopPropagation();
      const v = parseInt(jumpInput.value, 10);
      if (!Number.isNaN(v)) loadCell(v - 1);
      jumpInput.blur();
    }
  });

  document.getElementById("btnNextMissing").addEventListener("click", () => {
    const idx = state.cells.findIndex((c) => !c.saved);
    if (idx === -1) {
      showToast("¡Todos los glifos ya están guardados!", "ok");
    } else {
      loadCell(idx);
    }
  });

  // ---- Atajos de teclado ----
  window.addEventListener("keydown", (e) => {
    if (document.activeElement && document.activeElement.tagName === "INPUT" && document.activeElement.type === "number") {
      return;
    }
    if (e.code === "Space" || e.key === "Enter") {
      e.preventDefault();
      actionSaveAndNext();
    } else if (e.key === "z" || e.key === "Z") {
      e.preventDefault();
      actionUndo();
    } else if (e.key === "r" || e.key === "R" || e.key === "c" || e.key === "C" || e.key === "Delete") {
      e.preventDefault();
      actionClear();
    } else if (e.key === "ArrowLeft" || e.key === "Backspace") {
      e.preventDefault();
      actionPrev();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      actionSkip();
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      prefs.brushTplPx += 0.5;
      syncPrefControls();
      savePrefs();
      showToast(`Grosor: ${prefs.brushTplPx.toFixed(1)} px`, "", 900);
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault();
      prefs.brushTplPx -= 0.5;
      syncPrefControls();
      savePrefs();
      showToast(`Grosor: ${prefs.brushTplPx.toFixed(1)} px`, "", 900);
    } else if (e.key === "]") {
      e.preventDefault();
      prefs.canvasCssPx += 20;
      syncPrefControls();
      savePrefs();
      showToast(`Recuadro: ${prefs.canvasCssPx} px`, "", 900);
    } else if (e.key === "[") {
      e.preventDefault();
      prefs.canvasCssPx -= 20;
      syncPrefControls();
      savePrefs();
      showToast(`Recuadro: ${prefs.canvasCssPx} px`, "", 900);
    } else if (e.key === "e" || e.key === "E") {
      e.preventDefault();
      actionExport();
    }
  });

  // ---- Inicialización ----
  async function init() {
    const res = await fetch("/api/state");
    const data = await res.json();
    state.cells = data.cells;
    state.completed = data.completed;
    state.boxSize = data.box_size;
    state.baselineRatio = data.baseline_ratio;
    state.innerMargin = data.inner_margin;
    loadPrefs({
      brushTplPx: data.default_brush_px,
      canvasCssPx: data.default_canvas_px,
    });
    await loadCell(data.initial_index);
  }

  init();
})();
</script>
</body>
</html>
"""


def run_web_capture(
    layout: dict,
    flat_cells: list[dict],
    templates_dir: Path,
    cells_dir: Path,
    scans_dir: Path,
    describe_char_fn: Callable[[str], str],
    cell_path_fn: Callable[[Path, int], Path],
    has_ink_fn: Callable[[np.ndarray], bool],
    export_fn: Callable[[dict, Path, Path, Path], tuple[int, int]],
    default_brush_px: float = 6.5,
    default_canvas_px: int = 320,
    start_index: int | None = None,
    port: int = 0,
    open_browser: bool = True,
) -> tuple[int, int]:
    """Lanza un servidor HTTP local en localhost para captura fluida con Wacom."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    base_font = get_default_base_font_path()
    box_size = int(layout["geometry"]["box_size"])
    baseline_ratio = float(layout["geometry"].get("baseline_ratio", 0.72))

    ref_png_cache: dict[str, bytes] = {}
    lock = threading.Lock()
    shutdown_result: dict[str, tuple[int, int] | None] = {"value": None}

    def _get_ref_png(ch: str) -> bytes:
        if ch not in ref_png_cache:
            img = render_glyph(ch, base_font, img_size=148)
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError(f"No se pudo codificar el glifo de referencia '{ch}'.")
            ref_png_cache[ch] = buf.tobytes()
        return ref_png_cache[ch]

    def _count_completed() -> int:
        return sum(1 for c in flat_cells if cell_path_fn(cells_dir, c["sample_index"]).exists())

    def _first_missing() -> int:
        for i, c in enumerate(flat_cells):
            if not cell_path_fn(cells_dir, c["sample_index"]).exists():
                return i
        return 0

    initial_index = (
        max(0, min(len(flat_cells) - 1, start_index))
        if start_index is not None
        else _first_missing()
    )

    class CaptureHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # Silenciar logs HTTP verbosos en consola
            return

        def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path

            if path == "/" or path == "/index.html":
                self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/state":
                with lock:
                    cells_payload = [
                        {
                            "idx": i,
                            "sample_index": c["sample_index"],
                            "char": c["char"],
                            "desc": describe_char_fn(c["char"]),
                            "split": c["split"],
                            "page_index": c["page_index"],
                            "repetition": c["repetition"],
                            "saved": cell_path_fn(cells_dir, c["sample_index"]).exists(),
                        }
                        for i, c in enumerate(flat_cells)
                    ]
                    payload = {
                        "cells": cells_payload,
                        "completed": sum(1 for c in cells_payload if c["saved"]),
                        "initial_index": initial_index,
                        "box_size": box_size,
                        "baseline_ratio": baseline_ratio,
                        "inner_margin": EXTRACTOR_INNER_MARGIN,
                        "default_brush_px": default_brush_px,
                        "default_canvas_px": default_canvas_px,
                    }
                self._send_json(payload)
                return

            if path.startswith("/api/ref/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    ch = flat_cells[idx]["char"]
                    self._send_bytes(_get_ref_png(ch), "image/png")
                except (ValueError, IndexError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                return

            if path.startswith("/api/cell/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    c = flat_cells[idx]
                    p = cell_path_fn(cells_dir, c["sample_index"])
                    if p.exists():
                        self._send_bytes(p.read_bytes(), "image/png")
                    else:
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.end_headers()
                except (ValueError, IndexError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                return

            if path.startswith("/api/prev_rep/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    cur = flat_cells[idx]
                    found: Path | None = None
                    for prev_c in reversed(flat_cells):
                        if prev_c["char"] == cur["char"] and prev_c["repetition"] < cur["repetition"]:
                            p = cell_path_fn(cells_dir, prev_c["sample_index"])
                            if p.exists():
                                found = p
                                break
                    if found is not None:
                        self._send_bytes(found.read_bytes(), "image/png")
                    else:
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.end_headers()
                except (ValueError, IndexError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path

            if path.startswith("/api/cell/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    cell = flat_cells[idx]
                except (ValueError, IndexError):
                    self._send_json({"ok": False, "error": "Índice inválido."}, HTTPStatus.BAD_REQUEST)
                    return

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    b64_data = payload["image_base64"]
                    if "," in b64_data:
                        b64_data = b64_data.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                except Exception:
                    self._send_json({"ok": False, "error": "Payload PNG inválido."}, HTTPStatus.BAD_REQUEST)
                    return

                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if decoded is None:
                    self._send_json({"ok": False, "error": "No se pudo decodificar PNG."}, HTTPStatus.BAD_REQUEST)
                    return

                downscaled = cv2.resize(decoded, (box_size, box_size), interpolation=cv2.INTER_AREA)
                if not has_ink_fn(downscaled):
                    self._send_json(
                        {"ok": False, "error": "Lienzo vacío: dibuja el carácter antes de guardar."},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return

                with lock:
                    out_p = cell_path_fn(cells_dir, cell["sample_index"])
                    ok_enc, enc_buf = cv2.imencode(".png", downscaled)
                    if not ok_enc:
                        self._send_json({"ok": False, "error": "No se pudo guardar PNG."}, HTTPStatus.INTERNAL_SERVER_ERROR)
                        return
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_bytes(enc_buf.tobytes())
                    completed = _count_completed()

                self._send_json({"ok": True, "completed": completed, "total": len(flat_cells)})
                return

            if path == "/api/export":
                with lock:
                    pages, placed = export_fn(layout, templates_dir, cells_dir, scans_dir)
                self._send_json({"ok": True, "pages": pages, "placed": placed, "total": len(flat_cells)})
                return

            if path == "/api/shutdown":
                with lock:
                    pages, placed = export_fn(layout, templates_dir, cells_dir, scans_dir)
                    shutdown_result["value"] = (pages, placed)
                self._send_json({"ok": True, "pages": pages, "placed": placed, "total": len(flat_cells)})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer(("127.0.0.1", port), CaptureHandler)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"

    print(f"\n[Captura Wacom Fluida] Servidor activo en: {url}")
    print("  - Usa tu lápiz Wacom directo en el navegador (Windows Ink + suavizado tipo OneNote).")
    print("  - Ajusta 'Grosor de pluma' y 'Tamaño del recuadro' en vivo desde el panel izquierdo.")
    print("  - Pulsa el botón [Terminar] en la página (o Ctrl+C aquí) para guardar y exportar las 7 hojas.\n")

    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterrupción recibida; guardando y exportando hojas...")
    finally:
        server.server_close()

    if shutdown_result["value"] is not None:
        return shutdown_result["value"]
    return export_fn(layout, templates_dir, cells_dir, scans_dir)
