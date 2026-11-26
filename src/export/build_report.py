"""Generador reproducible del Reporte Web Autocontenido (`docs/reporte_completo.html`).

Lee dinámicamente los artefactos reales del repositorio:
- `results/eval/metrics_pretrain.json`
- `results/eval/auditoria_subgrupos_pretrain.json`
- `results/eval/gradcam_pretrain.json`
- `results/eval/robustez_ood.json`
- `results/benchmark_onnx.json`
- Hiperparámetros y vocabulario desde `src/config.py`
- Conteo exacto de parámetros instanciando modelos de `src/models/`
- Metadatos OpenType (`name` table, glifos y tamaño) desde `results/*.ttf`

Ensambla un único archivo HTML 100 % offline (cero CDNs: CSS, JS y gráficas SVG inline)
ideal para presentación en defensa técnica o portafolio.

Uso:
    python -m src.export.build_report
    python -m src.export.build_report --output docs/reporte_completo.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from fontTools.ttLib import TTFont

from src.config import (
    CHARSET,
    DIGITS,
    IMG_SIZE,
    JUDGES_CONFIG,
    LOWERCASE,
    PIX2PIX_CONFIG,
    PUNCTUATION,
    SEED,
    STYLE_UNET_CONFIG,
    TEST_UNSEEN_CHARS,
    UPPERCASE,
)
from src.models.judges import EfficientNetB0Judge, ResNet18Judge
from src.models.pix2pix import PatchGANDiscriminator, UNetGenerator
from src.models.style_unet import StyleEncoder, StyleUNetGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "reporte_completo.html"

OPENTYPE_NAME_IDS: dict[int, str] = {
    0: "copyright",
    1: "familyName",
    2: "styleName",
    3: "uniqueFontIdentifier",
    4: "fullName",
    5: "version",
    6: "psName",
    8: "manufacturer",
    9: "designer",
    10: "description",
    13: "licenseDescription",
}

# Resultados consolidados del dominio personal real con stylus (data/real_pairs, 300 muestras)
# Documentados en README.md, MODEL_CARD.md, notebooks/03_colab_finetuning.ipynb y 04_colab_style_encoder.ipynb.
REAL_STYLUS_RESULTS: dict[str, list[dict[str, Any]]] = {
    "val": [
        {
            "key": "reference_y",
            "label": "Referencia y (techo stylus real)",
            "tag": "Techo humano",
            "r18_acc": 89.06,
            "eff_acc": 84.38,
            "r18_f1": 86.20,
            "eff_f1": 81.25,
            "l1": None,
            "fid": 0.00,
            "highlight": False,
        },
        {
            "key": "pix2pix_pretrained",
            "label": "pix2pix_pretrained",
            "tag": "Base sintética",
            "r18_acc": 71.88,
            "eff_acc": 71.88,
            "r18_f1": 67.45,
            "eff_f1": 67.97,
            "l1": 0.0947,
            "fid": 145.25,
            "highlight": False,
        },
        {
            "key": "pix2pix_scratch",
            "label": "pix2pix_scratch",
            "tag": "150 ep. desde cero",
            "r18_acc": 75.00,
            "eff_acc": 67.19,
            "r18_f1": 69.27,
            "eff_f1": 59.64,
            "l1": 0.0570,
            "fid": 94.87,
            "highlight": False,
        },
        {
            "key": "pix2pix_finetuned",
            "label": "pix2pix_finetuned",
            "tag": "100 ep, lr=1e-4",
            "r18_acc": 62.50,
            "eff_acc": 56.25,
            "r18_f1": 56.38,
            "eff_f1": 47.50,
            "l1": 0.0550,
            "fid": 121.14,
            "highlight": False,
        },
        {
            "key": "style_unet_fewshot",
            "label": "style_unet_fewshot",
            "tag": "0 ep. FT · 4 refs",
            "r18_acc": 84.38,
            "eff_acc": 73.44,
            "r18_f1": 80.99,
            "eff_f1": 69.79,
            "l1": 0.0639,
            "fid": 141.47,
            "highlight": False,
        },
        {
            "key": "style_unet_finetuned",
            "label": "style_unet_finetuned",
            "tag": "Mejor ep. 4 · StyleEnc cong.",
            "r18_acc": 89.06,
            "eff_acc": 79.69,
            "r18_f1": 85.94,
            "eff_f1": 74.95,
            "l1": 0.0646,
            "fid": 126.18,
            "highlight": True,
        },
    ],
    "test_unseen": [
        {
            "key": "reference_y",
            "label": "Referencia y (techo stylus real)",
            "tag": "Techo humano",
            "r18_acc": 86.36,
            "eff_acc": 81.82,
            "r18_f1": 85.28,
            "eff_f1": 78.79,
            "l1": None,
            "fid": 0.00,
            "highlight": False,
        },
        {
            "key": "pix2pix_pretrained",
            "label": "pix2pix_pretrained",
            "tag": "Base sintética",
            "r18_acc": 27.27,
            "eff_acc": 45.45,
            "r18_f1": 27.27,
            "eff_f1": 42.42,
            "l1": 0.0981,
            "fid": 407.37,
            "highlight": False,
        },
        {
            "key": "pix2pix_scratch",
            "label": "pix2pix_scratch",
            "tag": "150 ep. desde cero",
            "r18_acc": 27.27,
            "eff_acc": 9.09,
            "r18_f1": 27.27,
            "eff_f1": 3.03,
            "l1": 0.0691,
            "fid": 469.96,
            "highlight": False,
        },
        {
            "key": "pix2pix_finetuned",
            "label": "pix2pix_finetuned",
            "tag": "100 ep, lr=1e-4",
            "r18_acc": 9.09,
            "eff_acc": 27.27,
            "r18_f1": 9.09,
            "eff_f1": 21.82,
            "l1": 0.0667,
            "fid": 499.83,
            "highlight": False,
        },
        {
            "key": "style_unet_fewshot",
            "label": "style_unet_fewshot",
            "tag": "0 ep. FT · 4 refs",
            "r18_acc": 54.55,
            "eff_acc": 45.45,
            "r18_f1": 54.55,
            "eff_f1": 42.42,
            "l1": 0.0676,
            "fid": 307.42,
            "highlight": False,
        },
        {
            "key": "style_unet_finetuned",
            "label": "style_unet_finetuned",
            "tag": "Mejor ep. 4 · StyleEnc cong.",
            "r18_acc": 72.73,
            "eff_acc": 45.45,
            "r18_f1": 72.73,
            "eff_f1": 42.42,
            "l1": 0.0728,
            "fid": 302.42,
            "highlight": True,
        },
    ],
}

# Costos medidos en NVIDIA Tesla T4 (docs/costos.md)
GPU_COSTS_ROWS: list[dict[str, str]] = [
    {
        "stage": "Juez ResNet-18",
        "dataset": "Sintético (3,825)",
        "epochs": "30 (mejor: ep. 23)",
        "time": "99.9 s (1.67 min)",
        "speed": "3.33 s/época",
        "size": "42.84 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.010 USD",
    },
    {
        "stage": "Juez EfficientNet-B0",
        "dataset": "Sintético (3,825)",
        "epochs": "22 (early stop, ep. 15)",
        "time": "117.3 s (1.96 min)",
        "speed": "5.33 s/época",
        "size": "15.95 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.011 USD",
    },
    {
        "stage": "Pix2Pix Pretrain",
        "dataset": "Sintético (2,688 train)",
        "epochs": "60 (mejor: ep. 11)",
        "time": "832.6 s (13.88 min)",
        "speed": "13.88 s/época",
        "size": "170.16 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.081 USD",
    },
    {
        "stage": "Pix2Pix Fine-Tune",
        "dataset": "Real stylus (192 train)",
        "epochs": "100 (mejor: ep. 51)",
        "time": "235.0 s (3.92 min)",
        "speed": "2.35 s/época",
        "size": "170.16 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.023 USD",
    },
    {
        "stage": "Pix2Pix Scratch",
        "dataset": "Real stylus (192 train)",
        "epochs": "150 (mejor: ep. 120)",
        "time": "309.1 s (5.15 min)",
        "speed": "2.06 s/época",
        "size": "170.16 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.030 USD",
    },
    {
        "stage": "StyleUNet Pretrain (aditivo)",
        "dataset": "Sintético (2,688 train)",
        "epochs": "60 (mejor: ep. 59)",
        "time": "2,782.9 s (46.38 min)",
        "speed": "46.38 s/época",
        "size": "~180 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.271 USD",
    },
    {
        "stage": "StyleUNet Fine-Tune (aditivo)",
        "dataset": "Real stylus (192 train)",
        "epochs": "40 (mejor: ep. 4)",
        "time": "157.0 s (2.62 min)",
        "speed": "3.93 s/época",
        "size": "~180 MB",
        "real_cost": "$0.00",
        "cloud_cost": "$0.015 USD",
    },
]


def _load_json(rel_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo JSON requerido: {path}")
    raw = path.read_text(encoding="utf-8")
    # auditoria_subgrupos_pretrain.json puede contener NaN de pandas; json.loads en Python lo acepta.
    return json.loads(raw)


def _clean_nan(obj: Any) -> Any:
    """Reemplaza NaN/Inf por None para serialización JSON estricta compatible con JS."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


def _count_model_params() -> list[dict[str, Any]]:
    """Instancia los modelos reales de `src/models/` en CPU y calcula sus parámetros exactos."""
    specs = [
        (
            "UNetGenerator",
            "Generador Pix2Pix",
            UNetGenerator(in_channels=1, out_channels=1, base_features=64),
            "[B, 1, 128, 128]",
            "[B, 1, 128, 128]",
            "7 enc + 7 dec, skip connections, Tanh",
            False,
        ),
        (
            "PatchGANDiscriminator",
            "Discriminador cGAN",
            PatchGANDiscriminator(in_channels=2, base_features=64),
            "[B, 2, 128, 128]",
            "[B, 1, 14, 14]",
            "Campo receptivo 70×70, solo entrenamiento",
            False,
        ),
        (
            "StyleUNetGenerator",
            "Generador Few-Shot",
            StyleUNetGenerator(
                in_channels=1,
                out_channels=1,
                base_features=64,
                style_dim=STYLE_UNET_CONFIG["style_dim"],
            ),
            "x + 4 refs [B, 4, 1, 128, 128]",
            "[B, 1, 128, 128]",
            "Content U-Net + StyleEncoder + 6 bloques AdaIN2d",
            True,
        ),
        (
            "StyleEncoder (submódulo)",
            "Rama de estilo",
            StyleEncoder(
                in_channels=1,
                style_dim=STYLE_UNET_CONFIG["style_dim"],
                base_features=32,
            ),
            "[B, 4, 1, 128, 128]",
            "[B, 128]",
            "5 Conv2d SIN InstanceNorm + MeanPool (preserva grosor)",
            False,
        ),
        (
            "ResNet18Judge",
            "Juez principal + OOD",
            ResNet18Judge(num_classes=len(CHARSET), pretrained=False),
            "[B, 1, 128, 128]",
            "Logits 75 / Emb 512-d",
            "Conv1 adaptada a 1 canal; Grad-CAM en layer4[-1]",
            False,
        ),
        (
            "EfficientNetB0Judge",
            "Juez validación cruzada",
            EfficientNetB0Judge(num_classes=len(CHARSET), pretrained=False),
            "[B, 1, 128, 128]",
            "Logits 75 / Emb 1280-d",
            "Conv separables + SE; Grad-CAM en features[-1]",
            False,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, role, model, in_shape, out_shape, notes, is_highlight in specs:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append(
            {
                "name": name,
                "role": role,
                "total": total,
                "trainable": trainable,
                "millions": round(total / 1e6, 2),
                "in_shape": in_shape,
                "out_shape": out_shape,
                "notes": notes,
                "highlight": is_highlight,
            }
        )
    return rows


def _extract_ttf_metadata() -> list[dict[str, Any]]:
    """Extrae en vivo la tabla OpenType `name`, número de glifos y tamaño de cada `results/*.ttf`."""
    ttf_dir = PROJECT_ROOT / "results"
    entries: list[dict[str, Any]] = []
    for ttf_path in sorted(ttf_dir.glob("*.ttf")):
        font = TTFont(str(ttf_path))
        glyph_count = len(font.getGlyphOrder())
        char_count = max(0, glyph_count - 2)  # excluye .notdef y space
        names_map: dict[str, str] = {}
        for rec in font["name"].names:
            # Preferir plataforma Windows Unicode (3, 1); fallback si no existe
            key = OPENTYPE_NAME_IDS.get(rec.nameID)
            if not key:
                continue
            if rec.platformID == 3 and rec.platEncID == 1:
                names_map[key] = rec.toUnicode()
            elif key not in names_map:
                names_map[key] = rec.toUnicode()
        desc = names_map.get("description", "")
        if "AI-SYNTHETIC" in desc:
            provenance = "AI-SYNTHETIC"
        elif "PERSONAL-STYLUS" in desc:
            provenance = "PERSONAL-STYLUS"
        else:
            provenance = "BASE-OPENTYPE (sin extensión)"
        entries.append(
            {
                "filename": ttf_path.name,
                "size_kb": round(ttf_path.stat().st_size / 1024.0, 2),
                "glyph_count": glyph_count,
                "char_count": char_count,
                "provenance": provenance,
                "fields": names_map,
            }
        )
    return entries


def _build_synthetic_experiment_rows(metrics_pretrain: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convierte `results/eval/metrics_pretrain.json` al formato unificado de la sección 4."""
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("val", "test_unseen"):
        sources = metrics_pretrain["splits"][split]["sources"]
        rows: list[dict[str, Any]] = []
        for src_key, label, tag, is_hl in [
            ("reference_y", "Referencia y (techo sintético)", "Techo fuentes OFL", False),
            ("pretrained", "pix2pix_pretrained", "60 ep, 14 fuentes", True),
        ]:
            s = sources[src_key]
            r18 = s["judges"]["resnet18"]
            eff = s["judges"]["efficientnet_b0"]
            fid = r18["domain_fid"]
            rows.append(
                {
                    "key": src_key if src_key == "reference_y" else "pix2pix_pretrained",
                    "label": label,
                    "tag": tag,
                    "r18_acc": round(r18["accuracy"] * 100.0, 2),
                    "eff_acc": round(eff["accuracy"] * 100.0, 2),
                    "r18_f1": round(r18["macro_f1"] * 100.0, 2),
                    "eff_f1": round(eff["macro_f1"] * 100.0, 2),
                    "l1": round(s["mean_l1"], 4) if s["mean_l1"] is not None else None,
                    "fid": round(fid, 2) if fid is not None else 0.00,
                    "highlight": is_hl,
                }
            )
        result[split] = rows
    return result


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _svg_architecture_diagram() -> str:
    """Diagrama de arquitectura end-to-end completamente en SVG inline (armonizado en modo oscuro)."""
    return """
    <svg class="arch-svg" viewBox="0 0 1180 398" role="img" aria-label="Diagrama de arquitectura End-to-End de Handwritten Font cGAN" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--svg-arrow)"/>
        </marker>
        <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent)"/>
        </marker>
        <marker id="arrow-teal" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--accent-2)"/>
        </marker>
        <marker id="arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--text-muted)"/>
        </marker>
      </defs>

      <!-- Placa de fondo oscura unificada -->
      <rect x="2" y="2" width="1176" height="394" rx="14" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>

      <!-- Zonas lógicas -->
      <rect x="214" y="16" width="336" height="238" rx="14" fill="var(--surface)" stroke="var(--border-strong)" stroke-dasharray="5 5"/>
      <text x="230" y="36" font-size="10.5" font-weight="700" fill="var(--text-muted)" letter-spacing="0.07em">SÍNTESIS CONDICIONAL (G)</text>

      <rect x="694" y="16" width="470" height="364" rx="14" fill="var(--surface)" stroke="var(--border-strong)" stroke-dasharray="5 5"/>
      <text x="710" y="36" font-size="10.5" font-weight="700" fill="var(--text-muted)" letter-spacing="0.07em">AUDITORÍA, EXPLICABILIDAD Y EXPORTACIÓN</text>

      <!-- Nodo 1: Entrada x -->
      <g transform="translate(18, 92)">
        <rect width="166" height="98" rx="12" fill="var(--svg-card)" stroke="var(--accent)" stroke-width="1.8"/>
        <text x="83" y="27" text-anchor="middle" font-size="13" font-weight="700" fill="var(--text)">Entrada x</text>
        <text x="83" y="47" text-anchor="middle" font-size="11.5" font-family="var(--font-mono)" fill="var(--accent)">1 × 128 × 128</text>
        <text x="83" y="68" text-anchor="middle" font-size="11" fill="var(--text-muted)">Lato-Regular / Canvas</text>
        <text x="83" y="84" text-anchor="middle" font-size="11" fill="var(--text-muted)">/ Webcam ROI [-1, 1]</text>
      </g>

      <!-- Nodo 1b: Referencias estilo -->
      <g transform="translate(18, 226)">
        <rect width="166" height="74" rx="12" fill="var(--svg-card)" stroke="var(--accent-2)" stroke-width="1.6" stroke-dasharray="4 3"/>
        <text x="83" y="25" text-anchor="middle" font-size="12" font-weight="700" fill="var(--text)">K=4 Refs de Estilo</text>
        <text x="83" y="44" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--accent-2)">[4, 1, 128, 128]</text>
        <text x="83" y="61" text-anchor="middle" font-size="10.5" fill="var(--text-muted)">Solo StyleUNet + AdaIN</text>
      </g>

      <!-- Nodo 2: Generador G -->
      <g transform="translate(232, 54)">
        <rect width="300" height="182" rx="14" fill="var(--svg-card)" stroke="var(--accent)" stroke-width="2"/>
        <rect x="0" y="0" width="300" height="33" rx="14" fill="var(--accent-solid)"/>
        <rect x="0" y="19" width="300" height="14" fill="var(--accent-solid)"/>
        <text x="150" y="21" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">Generador G (Pix2Pix / StyleUNet)</text>
        <text x="150" y="57" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)">U-Net 7 Enc + 7 Dec + Skip Connections</text>
        <text x="150" y="77" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--text-muted)">Pix2Pix: 41.82M · StyleUNet: 44.18M</text>
        <line x1="18" y1="91" x2="282" y2="91" stroke="var(--border-strong)"/>
        <text x="150" y="111" text-anchor="middle" font-size="11" fill="var(--text)">StyleEncoder (sin InstanceNorm) → z_style (128-d)</text>
        <text x="150" y="129" text-anchor="middle" font-size="11" fill="var(--text)">6 bloques AdaIN2d + pérdida L1 + InkDiff</text>
        <rect x="34" y="143" width="232" height="25" rx="7" fill="var(--accent-soft)" stroke="rgba(96,165,250,0.28)"/>
        <text x="150" y="159.5" text-anchor="middle" font-size="11" font-family="var(--font-mono)" font-weight="700" fill="var(--accent)">ONNX CPU: 16.86 ms (59.30 FPS)</text>
      </g>

      <!-- Nodo 3: Discriminador PatchGAN (solo train) -->
      <g transform="translate(254, 288)">
        <rect width="256" height="78" rx="12" fill="var(--svg-card)" stroke="var(--text-muted)" stroke-width="1.4" stroke-dasharray="5 4"/>
        <text x="128" y="25" text-anchor="middle" font-size="12" font-weight="700" fill="var(--text)">Discriminador PatchGAN 70×70</text>
        <text x="128" y="44" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--text-muted)">[x, y/ŷ] → logits 14×14 (2.76M)</text>
        <text x="128" y="62" text-anchor="middle" font-size="10.5" fill="var(--text-muted)">Solo entrenamiento · BCE (0.9) + λL1=100</text>
      </g>

      <!-- Nodo 4: Salida y_hat -->
      <g transform="translate(566, 98)">
        <rect width="108" height="86" rx="12" fill="var(--svg-card)" stroke="var(--accent-2)" stroke-width="2"/>
        <text x="54" y="29" text-anchor="middle" font-size="14" font-weight="800" fill="var(--text)">Glifo ŷ</text>
        <text x="54" y="50" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--accent-2)">1×128×128</text>
        <text x="54" y="68" text-anchor="middle" font-size="10.5" fill="var(--text-muted)">Salida Tanh</text>
      </g>

      <!-- Nodo 5: Jueces CNN (ampliado para margen holgado) -->
      <g transform="translate(710, 54)">
        <rect width="234" height="122" rx="12" fill="var(--svg-card)" stroke="var(--accent)" stroke-width="1.8"/>
        <text x="117" y="24" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--text)">Jueces Congelados</text>
        <text x="117" y="46" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--accent)">ResNet-18 (11.21M)</text>
        <text x="117" y="62" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="var(--text-muted)">75 clases · Emb 512-d · 4.69 ms</text>
        <line x1="18" y1="74" x2="216" y2="74" stroke="var(--border-strong)"/>
        <text x="117" y="92" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--accent-2)">EfficientNet-B0 (4.10M)</text>
        <text x="117" y="108" text-anchor="middle" font-size="10" font-family="var(--font-mono)" fill="var(--text-muted)">75 clases · Emb 1280-d (X-val)</text>
      </g>

      <!-- Nodo 6: Grad-CAM + OOD -->
      <g transform="translate(964, 54)">
        <rect width="184" height="122" rx="12" fill="var(--svg-card)" stroke="var(--accent-2)" stroke-width="1.6"/>
        <text x="92" y="24" text-anchor="middle" font-size="12" font-weight="700" fill="var(--text)">Explicabilidad &amp; OOD</text>
        <text x="92" y="46" text-anchor="middle" font-size="11" fill="var(--text)">Grad-CAM Espacial</text>
        <text x="92" y="62" text-anchor="middle" font-size="10.2" font-family="var(--font-mono)" fill="var(--accent-2)">cos = 0.9895 (ZS)</text>
        <line x1="16" y1="74" x2="168" y2="74" stroke="var(--border-strong)"/>
        <text x="92" y="92" text-anchor="middle" font-size="11" fill="var(--text)">Filtro OOD (τ = 0.70)</text>
        <text x="92" y="108" text-anchor="middle" font-size="10.2" font-family="var(--font-mono)" fill="var(--accent)">AUROC 0.9435 · 81.9%</text>
      </g>

      <!-- Nodo 7: Demo en vivo ONNX -->
      <g transform="translate(710, 196)">
        <rect width="438" height="74" rx="12" fill="var(--svg-card)" stroke="var(--accent)" stroke-width="1.6"/>
        <text x="219" y="24" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--text)">Demo OpenCV Tiempo Real (ONNX Runtime CPU)</text>
        <text x="219" y="44" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--accent)">21.07 ms (p95: 24.00 ms) · 47.45 FPS · Speedup 1.69×</text>
        <text x="219" y="62" text-anchor="middle" font-size="10.5" fill="var(--text-muted)">Canvas teclado / Dibujo libre / Webcam ROI · Alerta OK / OOD</text>
      </g>

      <!-- Nodo 8: Vectorización TTF -->
      <g transform="translate(710, 288)">
        <rect width="438" height="78" rx="12" fill="var(--svg-card)" stroke="var(--accent-3)" stroke-width="1.8"/>
        <text x="219" y="25" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--text)">Exportación Tipografía Instalable (.ttf)</text>
        <text x="219" y="45" text-anchor="middle" font-size="11" fill="var(--text)">vtracer (SVG) → Cu2QuPen → fontTools (75–84 glifos)</text>
        <text x="219" y="63" text-anchor="middle" font-size="10.5" font-family="var(--font-mono)" fill="var(--accent-3)">Tabla OpenType name: AI-SYNTHETIC / PERSONAL-STYLUS</text>
      </g>

      <!-- Flechas -->
      <line x1="184" y1="141" x2="229" y2="141" stroke="var(--accent)" stroke-width="2.2" marker-end="url(#arrow-accent)"/>
      <path d="M 184 263 L 208 263 L 208 192 L 229 192" fill="none" stroke="var(--accent-2)" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#arrow-teal)"/>
      <line x1="532" y1="141" x2="563" y2="141" stroke="var(--accent)" stroke-width="2.2" marker-end="url(#arrow-accent)"/>
      <!-- y_hat -> Jueces -->
      <path d="M 674 126 L 692 126 L 692 115 L 707 115" fill="none" stroke="var(--svg-arrow)" stroke-width="2" marker-end="url(#arrow)"/>
      <!-- Jueces -> GradCAM/OOD -->
      <line x1="944" y1="115" x2="961" y2="115" stroke="var(--svg-arrow)" stroke-width="1.8" marker-end="url(#arrow)"/>
      <!-- y_hat -> Demo -->
      <path d="M 674 150 L 692 150 L 692 233 L 707 233" fill="none" stroke="var(--svg-arrow)" stroke-width="1.8" marker-end="url(#arrow)"/>
      <!-- y_hat -> TTF -->
      <path d="M 620 184 L 620 327 L 707 327" fill="none" stroke="var(--svg-arrow)" stroke-width="1.8" marker-end="url(#arrow)"/>
      <!-- x & y_hat -> PatchGAN (train only) -->
      <path d="M 101 190 L 101 336 L 251 336" fill="none" stroke="var(--text-muted)" stroke-width="1.4" stroke-dasharray="4 4" marker-end="url(#arrow-muted)"/>
      <path d="M 594 184 L 594 316 L 513 316" fill="none" stroke="var(--text-muted)" stroke-width="1.4" stroke-dasharray="4 4" marker-end="url(#arrow-muted)"/>
    </svg>
    """


def _svg_model_params_chart(model_rows: list[dict[str, Any]]) -> str:
    """Gráfica SVG inline de barras horizontales con conteos de parámetros reales."""
    max_val = max(r["millions"] for r in model_rows)
    row_h = 42
    top_pad = 18
    height = top_pad + len(model_rows) * row_h + 18
    bars_svg: list[str] = [
        f'<rect x="1" y="1" width="778" height="{height - 2}" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'
    ]
    for i, r in enumerate(model_rows):
        y = top_pad + i * row_h
        w = max(8.0, (r["millions"] / max_val) * 355.0)
        color = "var(--accent)" if r["highlight"] else ("var(--accent-2)" if "Judge" in r["name"] else "var(--bar-neutral)")
        bars_svg.append(
            f'<g transform="translate(0, {y})">'
            f'<text x="218" y="18" text-anchor="end" font-size="12" font-weight="600" fill="var(--text)">{_esc(r["name"])}</text>'
            f'<rect x="230" y="5" width="355" height="20" rx="5" fill="var(--bar-track)"/>'
            f'<rect x="230" y="5" width="{w:.1f}" height="20" rx="5" fill="{color}"/>'
            f'<text x="{230 + w + 10:.1f}" y="19" font-size="11.5" font-family="var(--font-mono)" font-weight="700" fill="var(--text)">{r["millions"]:.2f}M ({r["total"]:,})</text>'
            f"</g>"
        )
    return (
        f'<svg class="chart-svg" viewBox="0 0 780 {height}" role="img" '
        f'aria-label="Parámetros por arquitectura en millones" xmlns="http://www.w3.org/2000/svg">'
        + "".join(bars_svg)
        + "</svg>"
    )


def _svg_subgroups_chart(subgroups_data: dict[str, Any]) -> str:
    """Gráfica SVG agrupada de Macro-F1 por subgrupo en `val` (ResNet-18: pretrained vs reference_y)."""
    order = ["Mayúsculas (A-Z)", "Minúsculas (a-z)", "Dígitos (0-9)", "Signos"]
    lookup: dict[tuple[str, str], float] = {}
    for row in subgroups_data["subgroup_table"]:
        if row["split"] == "val" and row["judge"] == "resnet18":
            lookup[(row["source"], row["subgroup"])] = float(row["macro_f1"])

    chart_w = 700
    chart_h = 282
    left = 56
    bottom = 232
    top = 38
    usable_h = bottom - top
    group_w = (chart_w - left - 26) / len(order)

    bg = f'<rect x="1" y="1" width="{chart_w - 2}" height="{chart_h - 2}" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'

    grid_lines: list[str] = []
    for tick in (0, 25, 50, 75, 100):
        y = bottom - (tick / 100.0) * usable_h
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{chart_w - 20}" y2="{y:.1f}" stroke="var(--svg-grid)" stroke-dasharray="3 3"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="var(--font-mono)" fill="var(--text-muted)">{tick}%</text>'
        )

    bars: list[str] = []
    for idx, sg in enumerate(order):
        gx = left + idx * group_w + group_w * 0.16
        bar_w = group_w * 0.28
        ref_val = lookup.get(("reference_y", sg), 0.0)
        pre_val = lookup.get(("pretrained", sg), 0.0)

        h_ref = (ref_val / 100.0) * usable_h
        h_pre = (pre_val / 100.0) * usable_h
        y_ref = bottom - h_ref
        y_pre = bottom - h_pre
        pre_color = "var(--danger)" if sg == "Signos" else "var(--accent)"

        bars.append(
            f'<rect x="{gx:.1f}" y="{y_ref:.1f}" width="{bar_w:.1f}" height="{h_ref:.1f}" rx="5" fill="var(--bar-track-strong)"/>'
            f'<text x="{gx + bar_w / 2:.1f}" y="{y_ref - 6:.1f}" text-anchor="middle" font-size="11" font-family="var(--font-mono)" fill="var(--text-muted)">{ref_val:.1f}%</text>'
            f'<rect x="{gx + bar_w + 8:.1f}" y="{y_pre:.1f}" width="{bar_w:.1f}" height="{h_pre:.1f}" rx="5" fill="{pre_color}"/>'
            f'<text x="{gx + bar_w + 8 + bar_w / 2:.1f}" y="{y_pre - 6:.1f}" text-anchor="middle" font-size="11.5" font-family="var(--font-mono)" font-weight="700" fill="{pre_color}">{pre_val:.1f}%</text>'
            f'<text x="{gx + bar_w + 4:.1f}" y="{bottom + 22}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)">{_esc(sg)}</text>'
        )

    legend = (
        f'<g transform="translate({left}, 14)">'
        f'<rect x="0" y="0" width="12" height="12" rx="3" fill="var(--bar-track-strong)"/>'
        f'<text x="18" y="10" font-size="11.5" fill="var(--text-muted)">Referencia y (techo)</text>'
        f'<rect x="172" y="0" width="12" height="12" rx="3" fill="var(--accent)"/>'
        f'<text x="190" y="10" font-size="11.5" fill="var(--text-muted)">pix2pix_pretrained (Macro-F1)</text>'
        f'<rect x="425" y="0" width="12" height="12" rx="3" fill="var(--danger)"/>'
        f'<text x="443" y="10" font-size="11.5" font-weight="700" fill="var(--danger)">Brecha máx: {subgroups_data["val_gap_macro_f1_pp"]:.2f} pp</text>'
        f"</g>"
    )

    return (
        f'<svg class="chart-svg" viewBox="0 0 {chart_w} {chart_h}" role="img" '
        f'aria-label="Macro-F1 por subgrupo en validación" xmlns="http://www.w3.org/2000/svg">'
        + bg
        + legend
        + "".join(grid_lines)
        + "".join(bars)
        + "</svg>"
    )


def _svg_error_taxonomy_chart(metrics_pretrain: dict[str, Any]) -> str:
    """Gráfica horizontal SVG del desglose de los 468 errores."""
    ea = metrics_pretrain["error_analysis"]["pretrained"]
    total = int(ea["total_errors"])
    counts = sorted(ea["counts_by_category"].items(), key=lambda kv: kv[1], reverse=True)
    max_c = max(v for _, v in counts)
    row_h = 39
    height = 24 + len(counts) * row_h + 12
    rows_svg: list[str] = [
        f'<rect x="1" y="1" width="698" height="{height - 2}" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'
    ]
    for idx, (cat, count) in enumerate(counts):
        y = 18 + idx * row_h
        pct = (count / total) * 100.0
        w = max(8.0, (count / max_c) * 295.0)
        color = "var(--danger)" if idx == 0 else ("var(--warning)" if idx <= 2 else "var(--accent)")
        rows_svg.append(
            f'<g transform="translate(0, {y})">'
            f'<text x="248" y="17" text-anchor="end" font-size="11.5" font-weight="600" fill="var(--text)">{_esc(cat)}</text>'
            f'<rect x="260" y="4" width="295" height="19" rx="5" fill="var(--bar-track)"/>'
            f'<rect x="260" y="4" width="{w:.1f}" height="19" rx="5" fill="{color}"/>'
            f'<text x="{260 + w + 10:.1f}" y="17.5" font-size="11.5" font-family="var(--font-mono)" font-weight="700" fill="var(--text)">{count} ({pct:.1f}%)</text>'
            f"</g>"
        )
    return (
        f'<svg class="chart-svg" viewBox="0 0 700 {height}" role="img" '
        f'aria-label="Taxonomía de {total} errores en evaluación pretrain" xmlns="http://www.w3.org/2000/svg">'
        + "".join(rows_svg)
        + "</svg>"
    )


def _svg_gradcam_alignment_chart(gradcam_data: dict[str, Any]) -> str:
    """Gráfica comparativa SVG de similitud coseno Grad-CAM (aciertos vs errores)."""
    av = gradcam_data["alignment_val"]
    au = gradcam_data["alignment_test_unseen"]
    items = [
        ("val — Aciertos (ŷ correcto)", av["mean_cos_correct"], "var(--accent-2)"),
        ("val — Errores (ŷ incorrecto)", av["mean_cos_wrong"], "var(--warning)"),
        ("test_unseen — Aciertos Zero-Shot", au["mean_cos_correct"], "var(--accent-2)"),
        ("test_unseen — Errores Zero-Shot", au["mean_cos_wrong"], "var(--danger)"),
    ]
    # Escala zoom 0.85 -> 1.00 para resaltar diferencias con eje etiquetado honestamente
    min_x, max_x = 0.85, 1.00
    rows_svg: list[str] = [
        '<rect x="1" y="1" width="698" height="224" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'
    ]
    for tick_x in (254, 367, 481, 594):
        rows_svg.append(
            f'<line x1="{tick_x}" y1="18" x2="{tick_x}" y2="194" stroke="var(--svg-grid)" stroke-dasharray="3 3"/>'
        )
    for i, (label, val, color) in enumerate(items):
        y = 24 + i * 42
        frac = max(0.0, min(1.0, (val - min_x) / (max_x - min_x)))
        w = frac * 340.0
        rows_svg.append(
            f'<g transform="translate(0, {y})">'
            f'<text x="242" y="18" text-anchor="end" font-size="12" font-weight="600" fill="var(--text)">{_esc(label)}</text>'
            f'<rect x="254" y="5" width="340" height="20" rx="5" fill="var(--bar-track)"/>'
            f'<rect x="254" y="5" width="{w:.1f}" height="20" rx="5" fill="{color}"/>'
            f'<text x="{254 + w + 10:.1f}" y="19" font-size="12" font-family="var(--font-mono)" font-weight="700" fill="{color}">{val:.4f}</text>'
            f"</g>"
        )
    axis = (
        '<text x="254" y="212" font-size="10.5" font-family="var(--font-mono)" fill="var(--text-muted)">0.85</text>'
        '<text x="367" y="212" text-anchor="middle" font-size="10.5" font-family="var(--font-mono)" fill="var(--text-muted)">0.90</text>'
        '<text x="481" y="212" text-anchor="middle" font-size="10.5" font-family="var(--font-mono)" fill="var(--text-muted)">0.95</text>'
        '<text x="594" y="212" text-anchor="end" font-size="10.5" font-family="var(--font-mono)" fill="var(--text-muted)">1.00 (eje truncado 0.85–1.00)</text>'
    )
    return (
        '<svg class="chart-svg" viewBox="0 0 700 226" role="img" '
        'aria-label="Similitud coseno espacial Grad-CAM aciertos vs errores" xmlns="http://www.w3.org/2000/svg">'
        + "".join(rows_svg)
        + axis
        + "</svg>"
    )


def _svg_ood_categories_chart(ood_data: dict[str, Any]) -> str:
    """Gráfica SVG de rechazo OOD por categoría a τ=0.70 (Pipeline vs ResNet-18 directo)."""
    cats = sorted(
        ood_data["by_category"],
        key=lambda c: c["Rechazo R18 pipeline @0.70 (%)"],
        reverse=True,
    )
    row_h = 46
    height = 46 + len(cats) * row_h + 14
    bg = f'<rect x="1" y="1" width="668" height="{height - 2}" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'
    legend = (
        '<g transform="translate(196, 14)">'
        '<rect x="0" y="0" width="12" height="12" rx="3" fill="var(--accent)"/>'
        '<text x="18" y="10" font-size="11.5" fill="var(--text-muted)">Pipeline Demo (G + ResNet-18) @τ=0.70</text>'
        '<rect x="282" y="0" width="12" height="12" rx="3" fill="var(--accent-2)"/>'
        '<text x="300" y="10" font-size="11.5" fill="var(--text-muted)">ResNet-18 directo</text>'
        "</g>"
    )
    rows_svg: list[str] = []
    for i, c in enumerate(cats):
        y = 42 + i * row_h
        name = c["Categoría OOD"]
        p_val = float(c["Rechazo R18 pipeline @0.70 (%)"])
        d_val = float(c["Rechazo R18 directo @0.70 (%)"])
        pw = (p_val / 100.0) * 340.0
        dw = (d_val / 100.0) * 340.0
        rows_svg.append(
            f'<g transform="translate(0, {y})">'
            f'<text x="184" y="19" text-anchor="end" font-size="11.5" font-family="var(--font-mono)" font-weight="600" fill="var(--text)">{_esc(name)}</text>'
            f'<rect x="196" y="2" width="340" height="13" rx="4" fill="var(--bar-track)"/>'
            f'<rect x="196" y="2" width="{pw:.1f}" height="13" rx="4" fill="var(--accent)"/>'
            f'<text x="{196 + pw + 8:.1f}" y="12.5" font-size="10.5" font-family="var(--font-mono)" font-weight="700" fill="var(--accent)">{p_val:.1f}%</text>'
            f'<rect x="196" y="19" width="340" height="11" rx="4" fill="var(--bar-track)"/>'
            f'<rect x="196" y="19" width="{dw:.1f}" height="11" rx="4" fill="var(--accent-2)"/>'
            f'<text x="{196 + dw + 8:.1f}" y="28.5" font-size="10.5" font-family="var(--font-mono)" fill="var(--accent-2)">{d_val:.1f}%</text>'
            f"</g>"
        )
    return (
        f'<svg class="chart-svg" viewBox="0 0 670 {height}" role="img" '
        f'aria-label="Tasa de rechazo OOD por categoría a umbral 0.70" xmlns="http://www.w3.org/2000/svg">'
        + bg
        + legend
        + "".join(rows_svg)
        + "</svg>"
    )


def _svg_onnx_benchmark_chart(bench: dict[str, Any]) -> str:
    """Gráfica comparativa SVG de latencia (ms) PyTorch CPU vs ONNX Runtime CPU."""
    stages = [
        ("Generador G (UNet)", "generator"),
        ("Juez (ResNet-18)", "judge"),
        ("Pipeline (G + Juez)", "pipeline"),
    ]
    max_ms = max(bench["pytorch_cpu"][k]["mean_ms"] for _, k in stages)
    row_h = 62
    height = 48 + len(stages) * row_h + 16
    bg = f'<rect x="1" y="1" width="708" height="{height - 2}" rx="10" fill="var(--svg-zone)" stroke="var(--border)" stroke-width="1"/>'
    legend = (
        '<g transform="translate(178, 14)">'
        '<rect x="0" y="0" width="12" height="12" rx="3" fill="var(--bar-track-strong)"/>'
        '<text x="18" y="10" font-size="11.5" fill="var(--text-muted)">PyTorch CPU (ms)</text>'
        '<rect x="155" y="0" width="12" height="12" rx="3" fill="var(--accent-2)"/>'
        '<text x="173" y="10" font-size="11.5" fill="var(--text-muted)">ONNX Runtime CPU (ms · menor es mejor)</text>'
        "</g>"
    )
    rows_svg: list[str] = []
    for i, (label, key) in enumerate(stages):
        y = 44 + i * row_h
        pt = bench["pytorch_cpu"][key]
        ox = bench["onnxruntime_cpu"][key]
        sp = bench["speedup"][key]
        pt_w = (pt["mean_ms"] / max_ms) * 320.0
        ox_w = (ox["mean_ms"] / max_ms) * 320.0
        rows_svg.append(
            f'<g transform="translate(0, {y})">'
            f'<text x="166" y="23" text-anchor="end" font-size="12" font-weight="700" fill="var(--text)">{_esc(label)}</text>'
            f'<rect x="178" y="2" width="320" height="16" rx="4" fill="var(--bar-track)"/>'
            f'<rect x="178" y="2" width="{pt_w:.1f}" height="16" rx="4" fill="var(--bar-track-strong)"/>'
            f'<text x="{178 + pt_w + 10:.1f}" y="14" font-size="11" font-family="var(--font-mono)" fill="var(--text-muted)">{pt["mean_ms"]:.2f} ms ({pt["fps"]:.2f} FPS)</text>'
            f'<rect x="178" y="23" width="320" height="16" rx="4" fill="var(--bar-track)"/>'
            f'<rect x="178" y="23" width="{ox_w:.1f}" height="16" rx="4" fill="var(--accent-2)"/>'
            f'<text x="{178 + ox_w + 10:.1f}" y="35" font-size="11.5" font-family="var(--font-mono)" font-weight="700" fill="var(--accent-2)">{ox["mean_ms"]:.2f} ms ({ox["fps"]:.2f} FPS · {sp:.2f}×)</text>'
            f"</g>"
        )
    return (
        f'<svg class="chart-svg" viewBox="0 0 710 {height}" role="img" '
        f'aria-label="Comparativa de latencia PyTorch CPU versus ONNX Runtime CPU" xmlns="http://www.w3.org/2000/svg">'
        + bg
        + legend
        + "".join(rows_svg)
        + "</svg>"
    )


def _render_figure_card(src: str, title: str, caption: str) -> str:
    return (
        '<figure class="fig-card">'
        f'<button type="button" class="fig-trigger" data-lightbox-src="{_esc(src)}" '
        f'data-lightbox-title="{_esc(title)}" data-lightbox-caption="{_esc(caption)}" '
        f'aria-label="Ampliar figura: {_esc(title)}">'
        f'<img src="{_esc(src)}" alt="{_esc(title)}" loading="lazy"/>'
        '<span class="fig-zoom-hint">Clic para ampliar</span>'
        "</button>"
        "<figcaption>"
        f"<strong>{_esc(title)}</strong>"
        f"<span>{_esc(caption)}</span>"
        "</figcaption>"
        "</figure>"
    )


def _render_tabbed_gallery(group_id: str, tabs: list[dict[str, str]]) -> str:
    """Renderiza una galería con pestañas accesibles (JS inline)."""
    buttons: list[str] = []
    panels: list[str] = []
    for idx, tab in enumerate(tabs):
        active = " is-active" if idx == 0 else ""
        selected = "true" if idx == 0 else "false"
        hidden = "" if idx == 0 else " hidden"
        tid = f"{group_id}-tab-{idx}"
        pid = f"{group_id}-panel-{idx}"
        buttons.append(
            f'<button type="button" class="tab-btn{active}" role="tab" id="{tid}" '
            f'aria-controls="{pid}" aria-selected="{selected}" data-tab-group="{group_id}" data-tab-target="{pid}">'
            f'{_esc(tab["label"])}</button>'
        )
        panels.append(
            f'<div class="tab-panel{active}" role="tabpanel" id="{pid}" aria-labelledby="{tid}" data-tab-group="{group_id}"{hidden}>'
            f'{_render_figure_card(tab["src"], tab["title"], tab["caption"])}'
            "</div>"
        )
    return (
        f'<div class="tab-shell" data-tabs="{group_id}">'
        f'<div class="tab-bar" role="tablist">{"".join(buttons)}</div>'
        f'<div class="tab-panels">{"".join(panels)}</div>'
        "</div>"
    )


def build_report_html() -> str:
    """Carga todos los artefactos en vivo y ensambla el HTML autocontenido."""
    metrics_pretrain = _load_json("results/eval/metrics_pretrain.json")
    subgroups_pretrain = _load_json("results/eval/auditoria_subgrupos_pretrain.json")
    gradcam_pretrain = _load_json("results/eval/gradcam_pretrain.json")
    ood_data = _load_json("results/eval/robustez_ood.json")
    bench_onnx = _load_json("results/benchmark_onnx.json")

    model_rows = _count_model_params()
    ttf_entries = _extract_ttf_metadata()
    synthetic_exp = _build_synthetic_experiment_rows(metrics_pretrain)

    # Datos para selector interactivo de la Sección 4 (JS inline)
    experiments_payload = _clean_nan(
        {
            "real_val": {
                "title": "Dominio Stylus Real — Validación (64 clases vistas)",
                "subtitle": "data/real_pairs/val (64 muestras reales con stylus · 0 % de traslape con el entrenamiento de los jueces).",
                "note": "style_unet_finetuned (mejor época 4, StyleEncoder congelado) iguala exactamente el techo humano de las muestras reales en ResNet-18 (89.06 % Acc / 85.94 % Macro-F1), mientras style_unet_fewshot alcanza 84.38 % sin ningún fine-tuning.",
                "rows": REAL_STYLUS_RESULTS["val"],
            },
            "real_unseen": {
                "title": "Dominio Stylus Real — Zero-Shot (11 clases inéditas)",
                "subtitle": "data/real_pairs/test_unseen (44 muestras · 11 símbolos jamás vistos por el generador: G, Q, k, z, 4, 8, ?, &, +, ;, )).",
                "note": "Cada clase en test_unseen pesa ~9.09 pp en Accuracy. StyleUNet dispara Zero-Shot de 9.09 % (pix2pix_finetuned) a 72.73 % (+63.64 pp) y reduce el FID de dominio de 499.83 a 302.42.",
                "rows": REAL_STYLUS_RESULTS["test_unseen"],
            },
            "syn_val": {
                "title": "Dominio Sintético Multi-Fuente — Validación (64 clases vistas)",
                "subtitle": "data/synthetic_pairs/val (576 pares de 3 fuentes Google Fonts OFL completamente inéditas para generador y jueces).",
                "note": "Datos leídos directamente desde results/eval/metrics_pretrain.json. El generador preentrenado alcanza 71.88 % Acc (ResNet-18) / 68.75 % (EfficientNet-B0) con FID de dominio 133.24.",
                "rows": synthetic_exp["val"],
            },
            "syn_unseen": {
                "title": "Dominio Sintético Multi-Fuente — Zero-Shot (11 clases inéditas)",
                "subtitle": "data/synthetic_pairs/test_unseen (561 pares · 11 símbolos reservados fuera del entrenamiento del generador).",
                "note": "Transparencia sobre judge_split: 462/561 referencias sintéticas pertenecen a fuentes de entrenamiento de los jueces (100.00 % Acc), mientras las 99 de fuentes inéditas (judge_split=val) alcanzan 89.90 % Acc. El generador nunca vio ninguno de los 11 símbolos.",
                "rows": synthetic_exp["test_unseen"],
            },
        }
    )

    # KPIs exactos del Hero
    kpi_symbols = len(CHARSET)
    kpi_style_val = REAL_STYLUS_RESULTS["val"][-1]["r18_acc"]  # 89.06
    kpi_style_zs = REAL_STYLUS_RESULTS["test_unseen"][-1]["r18_acc"]  # 72.73
    kpi_p2p_ft_zs = next(
        r["r18_acc"] for r in REAL_STYLUS_RESULTS["test_unseen"] if r["key"] == "pix2pix_finetuned"
    )  # 9.09
    kpi_zs_delta = round(kpi_style_zs - kpi_p2p_ft_zs, 2)  # +63.64
    kpi_auroc = ood_data["scenarios"]["resnet18_direct"]["auroc"]  # 0.9435
    kpi_fps = bench_onnx["onnxruntime_cpu"]["pipeline"]["fps"]  # 47.45
    kpi_speedup = bench_onnx["speedup"]["pipeline"]  # 1.69
    kpi_ms = bench_onnx["onnxruntime_cpu"]["pipeline"]["mean_ms"]  # 21.07

    # Chips de los 11 símbolos Zero-Shot en el orden exacto documentado
    unseen_order = ["G", "Q", "k", "z", "4", "8", "?", "&", "+", ";", ")"]
    assert set(unseen_order) == TEST_UNSEEN_CHARS, "Desajuste con src.config.TEST_UNSEEN_CHARS"
    unseen_chips_html = "".join(f'<span class="char-chip">{_esc(ch)}</span>' for ch in unseen_order)

    # Tabla de parámetros de arquitecturas (Sección 3)
    arch_rows_html = "".join(
        f'<tr class="{"row-highlight" if r["highlight"] else ""}">'
        f'<td><strong>{_esc(r["name"])}</strong><div class="sub-label">{_esc(r["role"])}</div></td>'
        f'<td class="mono num">{r["total"]:,}</td>'
        f'<td class="mono num">{r["millions"]:.2f} M</td>'
        f'<td class="mono">{_esc(r["in_shape"])}</td>'
        f'<td class="mono">{_esc(r["out_shape"])}</td>'
        f'<td>{_esc(r["notes"])}</td>'
        "</tr>"
        for r in model_rows
    )

    # Tabla de hiperparámetros centralizados desde src/config.py
    hparams_rows_html = "".join(
        [
            f"<tr><td><strong>Jueces CNN</strong> (ResNet-18 / EffNet-B0)</td>"
            f"<td class='mono'>ep={JUDGES_CONFIG['epochs']} (patience={JUDGES_CONFIG['patience']})</td>"
            f"<td class='mono'>bs={JUDGES_CONFIG['batch_size']}, lr={JUDGES_CONFIG['lr']:.0e}</td>"
            f"<td class='mono'>wd={JUDGES_CONFIG['weight_decay']:.0e}, label_smooth={JUDGES_CONFIG['label_smoothing']}</td></tr>",
            f"<tr><td><strong>Pix2Pix Pretrain</strong> (sintético)</td>"
            f"<td class='mono'>ep={PIX2PIX_CONFIG['modes']['pretrain']['epochs']}</td>"
            f"<td class='mono'>bs={PIX2PIX_CONFIG['modes']['pretrain']['batch_size']}, lr={PIX2PIX_CONFIG['modes']['pretrain']['lr']:.0e}</td>"
            f"<td class='mono'>λL1={PIX2PIX_CONFIG['lambda_l1']:.0f}, real_label={PIX2PIX_CONFIG['real_label']}, β={PIX2PIX_CONFIG['betas']}</td></tr>",
            f"<tr><td><strong>Pix2Pix Scratch / Fine-Tune</strong> (stylus)</td>"
            f"<td class='mono'>ep={PIX2PIX_CONFIG['modes']['scratch']['epochs']} / {PIX2PIX_CONFIG['modes']['finetune']['epochs']}</td>"
            f"<td class='mono'>bs={PIX2PIX_CONFIG['modes']['finetune']['batch_size']}, lr={PIX2PIX_CONFIG['modes']['scratch']['lr']:.0e} / {PIX2PIX_CONFIG['modes']['finetune']['lr']:.0e}</td>"
            f"<td class='mono'>λL1={PIX2PIX_CONFIG['lambda_l1']:.0f}, SEED={SEED}</td></tr>",
            f"<tr><td><strong>StyleUNet Pretrain / Fine-Tune</strong> (aditivo)</td>"
            f"<td class='mono'>ep={STYLE_UNET_CONFIG['modes']['pretrain']['epochs']} / {STYLE_UNET_CONFIG['modes']['finetune']['epochs']}</td>"
            f"<td class='mono'>bs={STYLE_UNET_CONFIG['batch_size']}, lr={STYLE_UNET_CONFIG['modes']['pretrain']['lr']:.0e} / {STYLE_UNET_CONFIG['modes']['finetune']['lr']:.0e}</td>"
            f"<td class='mono'>style_dim={STYLE_UNET_CONFIG['style_dim']}, K={STYLE_UNET_CONFIG['num_style_refs']}, λL1={STYLE_UNET_CONFIG['lambda_l1']:.0f}, λstyle={STYLE_UNET_CONFIG['lambda_style']:.0f}, λink={STYLE_UNET_CONFIG['lambda_ink']:.0f}</td></tr>",
        ]
    )

    # Tablas de Sección 5: peores caracteres y desglose Zero-Shot
    worst_rows_html = "".join(
        f"<tr>"
        f'<td class="mono"><span class="char-chip sm">{_esc(r["true_char"])}</span></td>'
        f'<td class="mono">{_esc(r["split"])}</td>'
        f'<td>{_esc(r["subgroup"])}</td>'
        f'<td class="mono num">{r["n"]}</td>'
        f'<td class="mono num">{r["acc_pct"]:.1f}%</td>'
        f'<td class="mono num">{r["mean_conf"]:.3f}</td>'
        f'<td class="mono num">{r["mean_l1"]:.4f}</td>'
        f'<td class="mono"><span class="char-chip sm warn">{_esc(r["top_pred"])}</span></td>'
        f"</tr>"
        for r in subgroups_pretrain["worst_characters"]
    )

    unseen_char_rows_html = "".join(
        f'<tr class="{"row-ok" if r["acc_pct"] >= 50 else "row-bad"}">'
        f'<td class="mono"><span class="char-chip sm">{_esc(r["true_char"])}</span></td>'
        f'<td>{_esc(r["subgroup"])}</td>'
        f'<td class="mono num">{r["acc_pct"]:.1f}%</td>'
        f'<td class="mono num">{r["mean_conf"]:.3f}</td>'
        f'<td class="mono num">{r["mean_l1"]:.4f}</td>'
        f'<td class="mono">{_esc(r["top_pred"])}</td>'
        f"</tr>"
        for r in subgroups_pretrain["unseen_characters"]
    )

    # Tabla de consistencia entre jueces Grad-CAM (Sección 6)
    cross_judge_rows_html = "".join(
        f'<tr class="{"row-bad" if s["cross_judge_cam_cos"] < 0.7 else ""}">'
        f'<td class="mono"><span class="char-chip sm">{_esc(s["true_char"])}</span></td>'
        f'<td class="mono">{_esc(s["split"])}</td>'
        f'<td class="mono">{_esc(s["resnet18_pred"])} ({s["resnet18_conf"]:.2f})</td>'
        f'<td class="mono">{_esc(s["effnet_pred"])} ({s["effnet_conf"]:.2f})</td>'
        f'<td class="mono num"><strong>{s["cross_judge_cam_cos"]:.4f}</strong></td>'
        f"</tr>"
        for s in gradcam_pretrain["panels"]["resnet_vs_effnet"]["samples"]
    )

    # Tabla escenarios OOD (Sección 7)
    ood_scenario_meta = [
        ("resnet18_direct", "ResNet-18 directo (y vs. OOD)", True),
        ("effnet_direct", "EfficientNet-B0 directo (y vs. OOD)", False),
        ("resnet18_pipeline", "Pipeline Demo (G + ResNet-18)", True),
        ("effnet_pipeline", "Pipeline Demo (G + EfficientNet-B0)", False),
    ]
    ood_scenarios_html = "".join(
        f'<tr class="{"row-highlight" if hl else ""}">'
        f"<td><strong>{_esc(label)}</strong></td>"
        f'<td class="mono num"><strong>{ood_data["scenarios"][k]["auroc"]:.4f}</strong></td>'
        f'<td class="mono num">{ood_data["scenarios"][k]["ood_rejection_at_070"] * 100:.1f}%</td>'
        f'<td class="mono num">{ood_data["scenarios"][k]["id_false_rejection_at_070"] * 100:.1f}%</td>'
        f'<td class="mono num">{ood_data["scenarios"][k]["best_youden_tau"]:.2f}</td>'
        f'<td class="mono num">{ood_data["scenarios"][k]["best_youden_ood_rejection"] * 100:.1f}%</td>'
        f"</tr>"
        for k, label, hl in ood_scenario_meta
    )

    # Tabla Benchmark ONNX (Sección 7)
    bench_stage_meta = [
        ("Generador G (UNetGenerator)", "generator", f'{bench_onnx["sizes_mb"]["generator_onnx"]:.2f} MB', f'{bench_onnx["max_abs_diff"]["generator"]:.2e}', False),
        ("Juez (ResNet-18)", "judge", f'{bench_onnx["sizes_mb"]["judge_onnx"]:.2f} MB', f'{bench_onnx["max_abs_diff"]["judge"]:.2e}', False),
        ("Pipeline completo (G + Juez)", "pipeline", f'{bench_onnx["sizes_mb"]["generator_onnx"] + bench_onnx["sizes_mb"]["judge_onnx"]:.2f} MB', "&lt; 1e-5", True),
    ]
    bench_rows_html = "".join(
        f'<tr class="{"row-highlight" if hl else ""}">'
        f"<td><strong>{_esc(label)}</strong></td>"
        f'<td class="mono num">{bench_onnx["pytorch_cpu"][k]["mean_ms"]:.2f} ± {bench_onnx["pytorch_cpu"][k]["std_ms"]:.2f}</td>'
        f'<td class="mono num"><strong>{bench_onnx["onnxruntime_cpu"][k]["mean_ms"]:.2f} ± {bench_onnx["onnxruntime_cpu"][k]["std_ms"]:.2f}</strong></td>'
        f'<td class="mono num">{bench_onnx["onnxruntime_cpu"][k]["p95_ms"]:.2f}</td>'
        f'<td class="mono num"><strong>{bench_onnx["onnxruntime_cpu"][k]["fps"]:.2f}</strong></td>'
        f'<td class="mono num"><strong>{bench_onnx["speedup"][k]:.2f}×</strong></td>'
        f'<td class="mono num">{size_s}</td>'
        f'<td class="mono num">{diff_s}</td>'
        f"</tr>"
        for label, k, size_s, diff_s, hl in bench_stage_meta
    )

    # Tabla OpenType extraída en vivo de results/*.ttf
    ttf_cards_html_parts: list[str] = []
    for entry in ttf_entries:
        field_rows = "".join(
            f"<tr><td class='mono'>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
            for k, v in entry["fields"].items()
        )
        badge_cls = "badge-ok" if entry["provenance"] in {"AI-SYNTHETIC", "PERSONAL-STYLUS"} else "badge-warn"
        ttf_cards_html_parts.append(
            '<div class="card ttf-card">'
            '<div class="ttf-header">'
            f'<div><h4 class="mono">{_esc(entry["filename"])}</h4>'
            f'<p class="muted">{entry["char_count"]} caracteres ({entry["glyph_count"]} glifos con .notdef/space) · {entry["size_kb"]:.2f} KB</p></div>'
            f'<span class="badge {badge_cls}">{_esc(entry["provenance"])}</span>'
            "</div>"
            '<div class="table-wrap"><table class="compact-table">'
            "<thead><tr><th>Campo OpenType (name)</th><th>Valor extraído del binario .ttf</th></tr></thead>"
            f"<tbody>{field_rows}</tbody></table></div>"
            "</div>"
        )
    ttf_cards_html = "".join(ttf_cards_html_parts)

    # Tabla de costos (Sección 7)
    costs_rows_html = "".join(
        f"<tr><td><strong>{_esc(r['stage'])}</strong></td><td>{_esc(r['dataset'])}</td>"
        f"<td class='mono'>{_esc(r['epochs'])}</td><td class='mono num'>{_esc(r['time'])}</td>"
        f"<td class='mono num'>{_esc(r['speed'])}</td><td class='mono num'>{_esc(r['size'])}</td>"
        f"<td class='mono num'>{_esc(r['real_cost'])}</td><td class='mono num'>{_esc(r['cloud_cost'])}</td></tr>"
        for r in GPU_COSTS_ROWS
    ) + (
        "<tr class='row-highlight'><td><strong>Total ciclo completo GPU (medido)</strong></td><td>—</td>"
        "<td class='mono'><strong>462 épocas</strong></td><td class='mono num'><strong>4,533.8 s (75.56 min)</strong></td>"
        "<td class='mono num'>—</td><td class='mono num'><strong>~929 MB</strong></td>"
        "<td class='mono num'><strong>$0.00</strong></td><td class='mono num'><strong>$0.441 USD</strong></td></tr>"
    )

    # Galerías con pestañas
    eda_gallery_html = _render_tabbed_gallery(
        "eda",
        [
            {
                "label": "Distribución de clases",
                "src": "figuras/eda_distribucion_clases.png",
                "title": "Distribución balanceada por subgrupo y densidad de tinta",
                "caption": "75 clases balanceadas por carácter en train/val/test_unseen; menor densidad de tinta en signos básicos.",
            },
            {
                "label": "Pares sintéticos (x, y)",
                "src": "figuras/eda_sinteticos.png",
                "title": "Muestras del dataset sintético (3,825 pares · 17 fuentes OFL)",
                "caption": "Emparejamiento exacto Lato-Regular (x) → fuentes manuscritas Google Fonts (y) a 1×128×128.",
            },
            {
                "label": "Auditoría real (30 muestras)",
                "src": "figuras/auditoria_calidad_real_30.png",
                "title": "Auditoría manual sobre 30 trazos reales con stylus (SEED=42)",
                "caption": "0/30 errores de etiquetado (0.0 %); 2.57 % densidad media de tinta; 300/300 pares válidos.",
            },
            {
                "label": "Extractor OpenCV",
                "src": "figuras/opencv_extractor_simulado.png",
                "title": "Verificación del extractor OpenCV con homografía",
                "caption": "Detección de 4 fiduciales, corrección de perspectiva a 2550×3300 y recorte con inner_margin=14 px.",
            },
            {
                "label": "Muestras OOD (72)",
                "src": "figuras/eda_ood.png",
                "title": "Conjunto Fuera de Distribución (6 categorías × 12 imágenes)",
                "caption": "Ruido puro, garabatos, símbolos inéditos, rotación ±45°, desenfoque/corte y sombra/bajo contraste.",
            },
        ],
    )

    curves_gallery_html = _render_tabbed_gallery(
        "curves",
        [
            {
                "label": "StyleUNet Pretrain",
                "src": "figuras/train_style_unet_pretrain.png",
                "title": "Entrenamiento StyleUNet Pretrain (60 épocas · mejor ep. 59)",
                "caption": "Convergencia con aumento morfológico de grosor, pérdida de estilo y penalización InkDiff.",
            },
            {
                "label": "StyleUNet Fine-Tune",
                "src": "figuras/train_style_unet_finetune.png",
                "title": "Entrenamiento StyleUNet Fine-Tune sobre stylus real (mejor ep. 4)",
                "caption": "StyleEncoder congelado + lr=5e-5: evita sobreajuste y alcanza 89.06 % val / 72.73 % Zero-Shot.",
            },
            {
                "label": "Pix2Pix Pretrain",
                "src": "figuras/train_pix2pix_pretrain.png",
                "title": "Entrenamiento Pix2Pix Pretrain (60 épocas · mejor ep. 11)",
                "caption": "U-Net + PatchGAN sobre 2,688 pares de 14 Google Fonts manuscritas.",
            },
            {
                "label": "Pix2Pix Scratch",
                "src": "figuras/train_pix2pix_scratch.png",
                "title": "Entrenamiento Pix2Pix Scratch sobre stylus real (150 épocas)",
                "caption": "Ajuste desde cero sobre 192 pares reales: alcanza 75.00 % Acc y FID 94.87 en val, pero cae en Zero-Shot.",
            },
            {
                "label": "Pix2Pix Fine-Tune",
                "src": "figuras/train_pix2pix_finetune.png",
                "title": "Entrenamiento Pix2Pix Fine-Tune (100 épocas · lr=1e-4)",
                "caption": "Choque de grosor de trazo entre fuentes sintéticas y stylus fino (2.57 % tinta).",
            },
            {
                "label": "Juez ResNet-18",
                "src": "figuras/train_judge_resnet18.png",
                "title": "Entrenamiento Juez ResNet-18 (30 épocas · mejor ep. 23)",
                "caption": "87.67 % Acc y 87.58 % Macro-F1 sobre 3 fuentes completamente inéditas en val.",
            },
            {
                "label": "Juez EfficientNet-B0",
                "src": "figuras/train_judge_efficientnet_b0.png",
                "title": "Entrenamiento Juez EfficientNet-B0 (early stop ep. 22 · mejor ep. 15)",
                "caption": "85.07 % Acc y 85.30 % Macro-F1; segundo juez independiente para validación cruzada.",
            },
        ],
    )

    grids_gallery_html = _render_tabbed_gallery(
        "grids",
        [
            {
                "label": "StyleUNet Val (Stylus)",
                "src": "figuras/style_eval_style_real_val.png",
                "title": "Grid visual StyleUNet sobre stylus real — Validación (89.06 % Acc)",
                "caption": "Comparación x | Few-Shot (0 ep.) | Fine-Tuned (ep. 4) | referencia real y: captura fiel del trazo fino.",
            },
            {
                "label": "StyleUNet Zero-Shot",
                "src": "figuras/style_eval_style_real_test_unseen.png",
                "title": "Grid visual StyleUNet sobre stylus real — Zero-Shot (72.73 % Acc)",
                "caption": "Síntesis de los 11 símbolos jamás vistos con estilo personal fino y alta legibilidad estructural.",
            },
            {
                "label": "Pix2Pix Pretrain Val",
                "src": "figuras/pix2pix_pretrain_val_grid.png",
                "title": "Grid visual Pix2Pix Pretrain — Validación",
                "caption": "Salida promedio manuscrita de las 17 fuentes sintéticas sobre caracteres vistos.",
            },
            {
                "label": "Pix2Pix Pretrain Unseen",
                "src": "figuras/pix2pix_pretrain_unseen_grid.png",
                "title": "Grid visual Pix2Pix Pretrain — Zero-Shot",
                "caption": "Preserva símbolos simples (k, ?, +, ;) vía skip connections, con degradación en G, Q, 4, 8, &.",
            },
            {
                "label": "Pix2Pix Scratch Val",
                "src": "figuras/pix2pix_scratch_val_grid.png",
                "title": "Grid visual Pix2Pix Scratch — Validación stylus real (75.00 % Acc)",
                "caption": "Aprende el grosor del stylus en caracteres vistos (FID 94.87), pero memoriza sin generalizar.",
            },
            {
                "label": "Pix2Pix Scratch Unseen",
                "src": "figuras/pix2pix_scratch_unseen_grid.png",
                "title": "Grid visual Pix2Pix Scratch — Zero-Shot stylus real (27.27 % Acc)",
                "caption": "Pérdida severa de estructura en caracteres no vistos al entrenarse solo con 192 muestras.",
            },
            {
                "label": "Pix2Pix Fine-Tune Val",
                "src": "figuras/pix2pix_finetune_val_grid.png",
                "title": "Grid visual Pix2Pix Fine-Tune — Validación stylus real (62.50 % Acc)",
                "caption": "Artefactos por conflicto entre pesos sintéticos gruesos y trazos finos del stylus.",
            },
            {
                "label": "Pix2Pix Fine-Tune Unseen",
                "src": "figuras/pix2pix_finetune_unseen_grid.png",
                "title": "Grid visual Pix2Pix Fine-Tune — Zero-Shot stylus real (9.09 % Acc)",
                "caption": "Colapso de generalización Zero-Shot que motivó la arquitectura aditiva StyleUNet + AdaIN.",
            },
            {
                "label": "Comparativa Sintética Val",
                "src": "figuras/eval_comparativa_pretrain_val.png",
                "title": "Comparativa cuantitativa sintética — Validación",
                "caption": "Accuracy, Macro-F1 y FID entre referencia y y pix2pix_pretrained (ambos jueces).",
            },
            {
                "label": "Comparativa Sintética Unseen",
                "src": "figuras/eval_comparativa_pretrain_test_unseen.png",
                "title": "Comparativa cuantitativa sintética — Zero-Shot",
                "caption": "Evaluación sobre los 11 símbolos inéditos en el dominio sintético multi-fuente.",
            },
        ],
    )

    subgroups_gallery_html = _render_tabbed_gallery(
        "subgroups",
        [
            {
                "label": "Auditoría por subgrupos",
                "src": "figuras/auditoria_subgrupos_pretrain.png",
                "title": "Desglose por subgrupo tipográfico (Mayúsculas 77.8 % vs. Signos 45.8 %)",
                "caption": "Brecha de 31.95 pp en Macro-F1 debido a baja densidad de tinta y puntos aislados en signos.",
            },
            {
                "label": "Matriz confusión (val)",
                "src": "figuras/eval_cm_pretrain_pretrained_val.png",
                "title": "Matriz de confusión — Validación (64 clases vistas)",
                "caption": "Diagonal dominante con confusiones concentradas en pares sans-serif (l/I, O/0, 6/C, !/l).",
            },
            {
                "label": "Matriz confusión (Zero-Shot)",
                "src": "figuras/eval_cm_pretrain_pretrained_test_unseen.png",
                "title": "Matriz de confusión — Zero-Shot (11 clases inéditas)",
                "caption": "Aciertos consistentes en ?, k, +, ;, z frente a confusiones de baja confianza en 4, Q, 8, ), G, &.",
            },
            {
                "label": "Galería de errores",
                "src": "figuras/eval_errores_pretrain_pretrained.png",
                "title": "Galería visual de casos representativos de error (x | ŷ | y)",
                "caption": "Inspección directa de fallos estructurales (6→C, S→C), pérdida de puntos (!→l, i→l) y Zero-Shot.",
            },
        ],
    )

    gradcam_gallery_html = _render_tabbed_gallery(
        "gradcam",
        [
            {
                "label": "Aciertos en val",
                "src": "figuras/gradcam_pretrain_val_correctos.png",
                "title": "Grad-CAM Aciertos en Validación (A, g, e, X, 7, Z)",
                "caption": "Similitud coseno media 0.9803 en val: atención focalizada en barras transversales, bucles y cruces.",
            },
            {
                "label": "Zero-Shot (test_unseen)",
                "src": "figuras/gradcam_pretrain_unseen_zeroshot.png",
                "title": "Grad-CAM Generalización Zero-Shot (k, ?, +, ;, z)",
                "caption": "Similitud coseno 0.9895 en aciertos Zero-Shot: las skip connections preservan regiones discriminativas.",
            },
            {
                "label": "Diagnóstico de errores",
                "src": "figuras/gradcam_pretrain_errores.png",
                "title": "Grad-CAM Diagnóstico Causal de Errores (6→C, :→;, !→l, i→l, S→C, l→I)",
                "caption": "Muestra causalmente la pérdida de bucles cerrados y puntos aislados bajo pérdida L1 multi-fuente.",
            },
            {
                "label": "ResNet-18 vs. EffNet-B0",
                "src": "figuras/gradcam_pretrain_resnet_vs_effnet.png",
                "title": "Consistencia entre evaluadores independientes (ResNet-18 vs. EfficientNet-B0)",
                "caption": "Acuerdo espacial 0.90–0.95 en glifos limpios (A, e, k, ?); divergencia 0.40 en 'g' revela espuela sutil.",
            },
        ],
    )

    ood_demo_gallery_html = _render_tabbed_gallery(
        "ood",
        [
            {
                "label": "Distribución de confianza",
                "src": "figuras/robustez_ood_distribucion.png",
                "title": "Distribución de confianza ID vs. OOD (72 muestras)",
                "caption": "Separación clara entre glifos válidos (alta confianza) y entradas degradadas o inéditas.",
            },
            {
                "label": "Curva ROC y umbral τ",
                "src": "figuras/robustez_ood_roc_umbral.png",
                "title": "Curva ROC (AUROC 0.9435) y barrido del umbral de rechazo τ",
                "caption": "Balance entre rechazo OOD (81.9 % en pipeline @τ=0.70, 94.4 % directo @τ=0.85) y falso rechazo ID.",
            },
            {
                "label": "Rechazo por categoría",
                "src": "figuras/robustez_ood_categorias.png",
                "title": "Desglose de rechazo en las 6 categorías OOD",
                "caption": "100 % en garabatos y símbolos inéditos; 91.7 % en ruido puro y rotación ±45°.",
            },
            {
                "label": "Ejemplos OOD evaluados",
                "src": "figuras/robustez_ood_ejemplos.png",
                "title": "Panel visual de entradas OOD y veredicto del juez",
                "caption": "Casos reales de cada categoría y cómo U-Net tiende a restaurar desenfoque leve.",
            },
            {
                "label": "Demo en vivo (OK)",
                "src": "../results/demo_headless_ok.png",
                "title": "Captura de la demo ONNX en tiempo real — Veredicto OK (CONFIABLE)",
                "caption": "Pipeline G + ResNet-18 en CPU mostrando glifo de entrada, salida ŷ, predicción y FPS en vivo.",
            },
            {
                "label": "Demo en vivo (OOD)",
                "src": "../results/demo_headless_ood.png",
                "title": "Captura de la demo ONNX en tiempo real — Alerta BAJA CONFIANZA (OOD)",
                "caption": "Detección automática de entrada fuera de distribución por debajo de τ=0.70.",
            },
        ],
    )

    font_previews_html = (
        '<div class="two-col">'
        + _render_figure_card(
            "../results/font_preview.png",
            "HandwrittenCGAN_Pretrain.ttf — Vista previa (84 caracteres)",
            "Fuente sintetizada glifo a glifo con generator.onnx (75 CHARSET + 9 inéditos @ # $ % ñ Ñ ¿ ¡ *) y vectorizada con vtracer + fontTools.",
        )
        + _render_figure_card(
            "../results/font_preview_stylus.png",
            "MiLetra_Stylus.ttf — Vista previa personal (75 caracteres)",
            "Fuente compilada seleccionando el medoide L1 más consistente entre las 4 repeticiones reales de cada símbolo capturado con stylus.",
        )
        + "</div>"
    )

    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    experiments_json = json.dumps(experiments_payload, ensure_ascii=False)

    # Grad-CAM valores
    av = gradcam_pretrain["alignment_val"]
    au = gradcam_pretrain["alignment_test_unseen"]
    total_errors = metrics_pretrain["error_analysis"]["pretrained"]["total_errors"]

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta name="color-scheme" content="dark"/>
<title>Handwritten Font cGAN — Reporte Técnico Interactivo</title>
<style>
:root {{
  --bg: #080d17;
  --surface: #101827;
  --surface-alt: #162135;
  --surface-hover: #1c2a42;
  --text: #f1f5f9;
  --text-muted: #9eb0c8;
  --border: #24334a;
  --border-strong: #334866;
  --accent: #60a5fa;
  --accent-solid: #2563eb;
  --accent-soft: rgba(96, 165, 250, 0.14);
  --accent-2: #2dd4bf;
  --accent-2-soft: rgba(45, 212, 191, 0.14);
  --accent-3: #c084fc;
  --warning: #fbbf24;
  --warning-soft: rgba(251, 191, 36, 0.13);
  --danger: #fb7185;
  --danger-soft: rgba(251, 113, 133, 0.14);
  --bar-track: #1e2c42;
  --bar-track-strong: #64748b;
  --bar-neutral: #94a3b8;
  --svg-zone: #0c1322;
  --svg-card: #131e31;
  --svg-arrow: #cbd5e1;
  --svg-grid: #24354f;
  --plate-bg: #f8fafc;
  --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.34);
  --shadow-md: 0 12px 32px rgba(0, 0, 0, 0.48);
  --radius: 14px;
  --sidebar-w: 264px;
  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "Cascadia Code", Consolas, "Liberation Mono", monospace;
}}

*, *::before, *::after {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; color-scheme: dark; }}
body {{
  margin: 0;
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.62;
  font-size: 15.5px;
  -webkit-font-smoothing: antialiased;
}}

/* Barra fija de progreso de lectura */
.reading-progress {{
  position: fixed;
  top: 0;
  left: 0;
  width: 0%;
  height: 3px;
  z-index: 90;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
  box-shadow: 0 0 10px rgba(96, 165, 250, 0.45);
  pointer-events: none;
  transition: width 0.08s linear;
}}

/* Botón discreto Volver Arriba */
.back-to-top {{
  position: fixed;
  right: 24px;
  bottom: 24px;
  z-index: 50;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 9px 14px;
  border-radius: 999px;
  border: 1px solid var(--border-strong);
  background: rgba(16, 24, 39, 0.92);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: var(--shadow-md);
  opacity: 0;
  transform: translateY(10px);
  pointer-events: none;
  transition: opacity 0.2s ease, transform 0.2s ease, border-color 0.15s, color 0.15s;
}}
.back-to-top.is-visible {{
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}}
.back-to-top:hover {{
  border-color: var(--accent);
  color: var(--accent);
}}

/* Layout principal */
.app-shell {{
  display: grid;
  grid-template-columns: var(--sidebar-w) minmax(0, 1fr);
  min-height: 100vh;
}}

/* Sidebar fija */
.sidebar {{
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 22px 16px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  z-index: 30;
}}

.brand {{
  padding: 4px 8px 14px;
  border-bottom: 1px solid var(--border);
}}
.brand-kicker {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
}}
.brand-title {{
  font-size: 17px;
  font-weight: 800;
  margin: 4px 0 2px;
  line-height: 1.25;
}}
.brand-sub {{
  font-size: 12px;
  color: var(--text-muted);
  margin: 0;
}}

.nav-list {{
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.nav-link {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 11px;
  border-radius: 10px;
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13.5px;
  font-weight: 600;
  transition: background 0.16s, color 0.16s;
}}
.nav-link:hover {{
  background: var(--surface-alt);
  color: var(--text);
}}
.nav-link.is-active {{
  background: var(--accent-soft);
  color: var(--accent);
}}
.nav-num {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  background: var(--surface-alt);
  color: var(--text-muted);
  flex-shrink: 0;
}}
.nav-link.is-active .nav-num {{
  background: var(--accent-solid);
  color: #ffffff;
}}

.sidebar-footer {{
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 8px;
}}

.meta-pill {{
  font-size: 11.5px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: 10px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
}}

/* Topbar móvil */
.mobile-bar {{
  display: none;
  position: sticky;
  top: 0;
  z-index: 40;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 10px 16px;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}}
.mobile-nav-strip {{
  display: none;
  overflow-x: auto;
  gap: 6px;
  padding: 8px 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}}
.mobile-nav-strip a {{
  white-space: nowrap;
  padding: 6px 10px;
  border-radius: 8px;
  font-size: 12.5px;
  font-weight: 600;
  text-decoration: none;
  color: var(--text-muted);
  background: var(--surface-alt);
}}
.mobile-nav-strip a.is-active {{
  background: var(--accent-soft);
  color: var(--accent);
}}

/* Contenido principal */
.main {{
  padding: 28px 40px 72px;
  max-width: 1220px;
  width: 100%;
  margin: 0 auto;
}}

/* Hero */
.hero {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 32px 34px 28px;
  box-shadow: var(--shadow-md);
  margin-bottom: 30px;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
}}
.hero-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}}
.pill {{
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.02em;
  background: var(--surface-alt);
  color: var(--text-muted);
  border: 1px solid var(--border);
}}
.pill.accent {{
  background: var(--accent-soft);
  color: var(--accent);
  border-color: transparent;
}}
.pill.teal {{
  background: var(--accent-2-soft);
  color: var(--accent-2);
  border-color: transparent;
}}
.hero h1 {{
  margin: 0 0 10px;
  font-size: clamp(24px, 3vw, 34px);
  line-height: 1.18;
  letter-spacing: -0.02em;
}}
.hero-lead {{
  margin: 0 0 22px;
  color: var(--text-muted);
  font-size: 15.5px;
  max-width: 88ch;
}}

/* KPI Grid */
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}}
.kpi-card {{
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 15px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.kpi-card.emphasis {{
  background: var(--accent-soft);
  border-color: rgba(96, 165, 250, 0.34);
}}
.kpi-label {{
  font-size: 11.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
}}
.kpi-value {{
  font-family: var(--font-mono);
  font-size: 23px;
  font-weight: 800;
  line-height: 1.15;
  color: var(--text);
}}
.kpi-card.emphasis .kpi-value {{ color: var(--accent); }}
.kpi-sub {{
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.35;
}}

/* Secciones */
.report-section {{
  scroll-margin-top: 24px;
  margin-bottom: 42px;
}}
.section-head {{
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--border);
}}
.section-tag {{
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 800;
  color: #ffffff;
  background: var(--accent-solid);
  padding: 3px 9px;
  border-radius: 7px;
  flex-shrink: 0;
}}
.section-head h2 {{
  margin: 0;
  font-size: 22px;
  letter-spacing: -0.015em;
}}

/* Cards & grids */
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 22px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 18px;
}}
.card h3 {{
  margin: 0 0 10px;
  font-size: 16.5px;
}}
.card h4 {{
  margin: 0 0 6px;
  font-size: 14.5px;
}}
.muted {{ color: var(--text-muted); font-size: 13.5px; }}
.sub-label {{ font-size: 12px; color: var(--text-muted); font-weight: 500; }}

.two-col {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}}
.three-col {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}}
.four-col {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}}

/* Callouts */
.callout {{
  border-radius: var(--radius);
  padding: 18px 20px;
  margin: 18px 0;
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  background: var(--accent-soft);
}}
.callout.teal {{
  border-left-color: var(--accent-2);
  background: var(--accent-2-soft);
}}
.callout.warn {{
  border-left-color: var(--warning);
  background: var(--warning-soft);
}}
.callout-title {{
  font-weight: 800;
  font-size: 14.5px;
  margin: 0 0 6px;
}}
.callout p {{
  margin: 0;
  font-size: 14px;
}}

/* Chips y badges */
.chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 8px;
}}
.char-chip {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 34px;
  height: 34px;
  padding: 0 8px;
  border-radius: 8px;
  background: var(--surface-alt);
  border: 1px solid var(--border);
  font-family: var(--font-mono);
  font-weight: 800;
  font-size: 15px;
  color: var(--accent);
}}
.char-chip.sm {{
  min-width: 26px;
  height: 26px;
  font-size: 13px;
  padding: 0 6px;
}}
.char-chip.warn {{
  color: var(--danger);
  background: var(--danger-soft);
  border-color: transparent;
}}

.badge {{
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}}
.badge-ok {{ background: var(--accent-2-soft); color: var(--accent-2); }}
.badge-warn {{ background: var(--warning-soft); color: var(--warning); }}

/* Tablas */
.table-wrap {{
  overflow-x: auto;
  border-radius: 10px;
  border: 1px solid var(--border);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
}}
th, td {{
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
thead th {{
  background: var(--surface-alt);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--text-muted);
  white-space: nowrap;
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--surface-alt); }}
.compact-table th, .compact-table td {{ padding: 7px 10px; font-size: 12.5px; }}
.mono {{ font-family: var(--font-mono); }}
.num {{ text-align: right; white-space: nowrap; }}
.row-highlight {{ background: var(--accent-soft); }}
.row-highlight:hover {{ background: var(--accent-soft); }}
.row-ok td:first-child {{ border-left: 3px solid var(--accent-2); }}
.row-bad td:first-child {{ border-left: 3px solid var(--danger); }}

/* SVGs */
.svg-wrap {{
  width: 100%;
  overflow-x: auto;
  padding: 6px 0;
}}
.arch-svg {{
  width: 100%;
  min-width: 720px;
  height: auto;
  display: block;
}}
.chart-svg {{
  width: 100%;
  height: auto;
  display: block;
}}

/* Selector interactivo de experimentos (Sección 4) */
.exp-toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}}
.exp-btn {{
  padding: 9px 14px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface-alt);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.15s;
}}
.exp-btn:hover {{ border-color: var(--accent); }}
.exp-btn.is-active {{
  background: var(--accent-solid);
  color: #ffffff;
  border-color: var(--accent);
}}

.exp-charts-grid {{
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 18px;
  margin: 16px 0 18px;
}}
.bar-card {{
  background: var(--surface-alt);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 15px 16px;
}}
.bar-card h4 {{
  margin: 0 0 12px;
  font-size: 13.5px;
  color: var(--text);
}}
.hbar-row {{
  margin-bottom: 11px;
}}
.hbar-row:last-child {{ margin-bottom: 0; }}
.hbar-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  font-size: 12.5px;
  margin-bottom: 4px;
}}
.hbar-name {{ font-weight: 600; }}
.hbar-val {{ font-family: var(--font-mono); font-weight: 700; font-size: 12px; }}
.hbar-track {{
  width: 100%;
  height: 11px;
  background: var(--bar-track);
  border-radius: 999px;
  overflow: hidden;
  margin-bottom: 3px;
}}
.hbar-fill {{
  height: 100%;
  border-radius: 999px;
  transition: width 0.25s ease;
}}
.hbar-fill.r18 {{ background: var(--accent); }}
.hbar-fill.eff {{ background: var(--accent-2); }}
.hbar-fill.fid {{ background: var(--accent-3); }}
.hbar-fill.winner {{ background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
.legend-inline {{
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  font-size: 11.5px;
  color: var(--text-muted);
  margin-bottom: 10px;
}}
.legend-dot {{
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  margin-right: 5px;
  vertical-align: -1px;
}}

/* Pestañas de galerías */
.tab-shell {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
  margin-bottom: 18px;
}}
.tab-bar {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}}
.tab-btn {{
  padding: 7px 12px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface-alt);
  color: var(--text-muted);
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}}
.tab-btn:hover {{ color: var(--text); border-color: var(--accent); }}
.tab-btn.is-active {{
  background: var(--accent-soft);
  color: var(--accent);
  border-color: var(--accent);
}}

/* Figuras + marco lámina para PNGs sobre fondo oscuro */
.fig-card {{
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}}
.fig-trigger {{
  position: relative;
  display: block;
  width: 100%;
  padding: 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: var(--plate-bg);
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06), 0 8px 22px rgba(0, 0, 0, 0.34);
  cursor: zoom-in;
  overflow: hidden;
  transition: border-color 0.16s, transform 0.16s, box-shadow 0.16s;
}}
.fig-trigger:hover {{
  border-color: var(--accent);
  transform: translateY(-1px);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.48), 0 0 0 1px var(--accent);
}}
.fig-trigger img {{
  display: block;
  max-width: 100%;
  max-height: 520px;
  width: auto;
  height: auto;
  margin: 0 auto;
  border-radius: 6px;
}}
.fig-zoom-hint {{
  position: absolute;
  right: 14px;
  bottom: 14px;
  background: rgba(8, 13, 23, 0.86);
  color: #f8fafc;
  border: 1px solid rgba(255, 255, 255, 0.16);
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 999px;
  opacity: 0;
  transition: opacity 0.16s;
}}
.fig-trigger:hover .fig-zoom-hint {{ opacity: 1; }}
.fig-card figcaption {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 13px;
}}
.fig-card figcaption span {{ color: var(--text-muted); }}

/* TTF cards */
.ttf-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}}
.ttf-header p {{ margin: 2px 0 0; }}

/* Modal Lightbox con navegación ← / → y marco lámina */
.lightbox {{
  position: fixed;
  inset: 0;
  z-index: 100;
  background: rgba(4, 7, 14, 0.90);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
}}
.lightbox.is-open {{ display: flex; }}
.lightbox-dialog {{
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: 16px;
  width: min(1160px, 95vw);
  max-height: 92vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 28px 72px rgba(0, 0, 0, 0.68);
}}
.lightbox-bar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-alt);
}}
.lightbox-head-left {{
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}}
.lightbox-counter {{
  font-family: var(--font-mono);
  font-size: 11.5px;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  white-space: nowrap;
}}
.lightbox-title {{
  font-weight: 700;
  font-size: 14px;
  color: var(--text);
}}
.lightbox-actions {{
  display: flex;
  align-items: center;
  gap: 7px;
  flex-shrink: 0;
}}
.lightbox-nav-btn,
.lightbox-close {{
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text);
  border-radius: 8px;
  padding: 6px 11px;
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}}
.lightbox-nav-btn:hover,
.lightbox-close:hover {{
  border-color: var(--accent);
  color: var(--accent);
  background: var(--surface-hover);
}}
.lightbox-body {{
  padding: 18px;
  background: var(--bg);
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.lightbox-plate {{
  background: var(--plate-bg);
  padding: 12px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
  max-width: 100%;
}}
.lightbox-plate img {{
  display: block;
  max-width: 100%;
  max-height: 70vh;
  width: auto;
  height: auto;
  margin: 0 auto;
  border-radius: 6px;
}}
.lightbox-caption {{
  padding: 11px 18px;
  font-size: 13px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  background: var(--surface-alt);
}}

/* Footer */
.report-footer {{
  margin-top: 40px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 10px;
  font-size: 12.5px;
  color: var(--text-muted);
}}

/* Responsivo */
@media (max-width: 1080px) {{
  .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  .exp-charts-grid {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 900px) {{
  .app-shell {{ grid-template-columns: 1fr; }}
  .sidebar {{ display: none; }}
  .mobile-bar {{ display: flex; }}
  .mobile-nav-strip {{ display: flex; }}
  .main {{ padding: 20px 16px 64px; }}
  .two-col, .three-col, .four-col {{ grid-template-columns: 1fr; }}
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .back-to-top {{ right: 14px; bottom: 14px; }}
}}
@media print {{
  .sidebar, .mobile-bar, .mobile-nav-strip, .fig-zoom-hint, .exp-toolbar, .tab-bar, .reading-progress, .back-to-top {{ display: none !important; }}
  .app-shell {{ display: block; }}
  .main {{ max-width: 100%; padding: 0; }}
  .tab-panel {{ display: block !important; margin-bottom: 14px; }}
}}
</style>
</head>
<body>

<div class="reading-progress" id="reading-progress" aria-hidden="true"></div>

<div class="mobile-bar">
  <div>
    <div class="brand-kicker">Handwritten Font cGAN</div>
    <strong style="font-size:14px;">Reporte Técnico Interactivo</strong>
  </div>
  <span class="pill accent">100 % Offline</span>
</div>
<nav class="mobile-nav-strip" aria-label="Secciones móvil">
  <a href="#sec-1">1. Resumen</a>
  <a href="#sec-2">2. Datos</a>
  <a href="#sec-3">3. Arquitecturas</a>
  <a href="#sec-4">4. Experimentos</a>
  <a href="#sec-5">5. Subgrupos</a>
  <a href="#sec-6">6. Grad-CAM</a>
  <a href="#sec-7">7. OOD &amp; .ttf</a>
</nav>

<div class="app-shell">
  <aside class="sidebar" aria-label="Navegación principal">
    <div class="brand">
      <div class="brand-kicker">Deep Learning · Visión CV</div>
      <div class="brand-title">Handwritten Font cGAN</div>
      <p class="brand-sub">CUGDL · Carlo A. Amezcua L.</p>
    </div>

    <ul class="nav-list" id="sidebar-nav">
      <li><a class="nav-link is-active" href="#sec-1"><span class="nav-num">01</span>Resumen y Flujo End-to-End</a></li>
      <li><a class="nav-link" href="#sec-2"><span class="nav-num">02</span>Datos y Protocolo Anti-Fuga</a></li>
      <li><a class="nav-link" href="#sec-3"><span class="nav-num">03</span>Arquitecturas y Decisiones</a></li>
      <li><a class="nav-link" href="#sec-4"><span class="nav-num">04</span>Experimentos y Evolución</a></li>
      <li><a class="nav-link" href="#sec-5"><span class="nav-num">05</span>Subgrupos y Errores</a></li>
      <li><a class="nav-link" href="#sec-6"><span class="nav-num">06</span>Interpretabilidad Grad-CAM</a></li>
      <li><a class="nav-link" href="#sec-7"><span class="nav-num">07</span>OOD, ONNX, .ttf y Ética</a></li>
    </ul>

    <div class="sidebar-footer">
      <div class="meta-pill">
        SEED={SEED} · {IMG_SIZE}×{IMG_SIZE}<br/>
        100 % Offline · 0 CDNs<br/>
        Build: {_esc(generated_ts)}
      </div>
    </div>
  </aside>

  <main class="main">
    <!-- HERO -->
    <header class="hero">
      <div class="hero-meta">
        <span class="pill accent">Proyecto Final · Visión por Computadora</span>
        <span class="pill teal">Pix2Pix cGAN + StyleUNet (AdaIN)</span>
        <span class="pill">ONNX Runtime CPU · vtracer + fontTools</span>
        <span class="pill">Autocontenido Offline</span>
      </div>
      <h1>Handwritten Font cGAN — Generación de Tipografía Manuscrita</h1>
      <p class="hero-lead">
        Sistema <em>end-to-end</em> de traducción imagen-a-imagen condicional para sintetizar glifos manuscritos
        (<span class="mono">1×128×128</span>), auditar su legibilidad espacial con dos jueces CNN congelados
        (<span class="mono">ResNet-18</span> y <span class="mono">EfficientNet-B0</span>), filtrar entradas Fuera de Distribución (OOD),
        ejecutar inferencia en tiempo real sobre CPU con <strong>ONNX Runtime</strong> y compilar tipografías instalables <strong>.ttf</strong>
        con marcadores OpenType de trazabilidad ética.
      </p>

      <div class="kpi-grid">
        <div class="kpi-card">
          <span class="kpi-label">Vocabulario</span>
          <span class="kpi-value">{kpi_symbols} símbolos</span>
          <span class="kpi-sub">4 subgrupos · 64 vistas + 11 Zero-Shot</span>
        </div>
        <div class="kpi-card emphasis">
          <span class="kpi-label">StyleUNet Val</span>
          <span class="kpi-value">{kpi_style_val:.2f}% Acc</span>
          <span class="kpi-sub">Iguala techo humano stylus (85.94% F1)</span>
        </div>
        <div class="kpi-card emphasis">
          <span class="kpi-label">Zero-Shot Stylus</span>
          <span class="kpi-value">{kpi_style_zs:.2f}%</span>
          <span class="kpi-sub">+{kpi_zs_delta:.2f} pp vs Pix2Pix FT · FID 302.42</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Detección OOD</span>
          <span class="kpi-value">{kpi_auroc:.4f} AUROC</span>
          <span class="kpi-sub">ResNet-18 directo · 81.9% pipeline @τ=0.70</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Inferencia CPU</span>
          <span class="kpi-value">{kpi_fps:.2f} FPS</span>
          <span class="kpi-sub">{kpi_ms:.2f} ms ONNX ({kpi_speedup:.2f}× vs PyTorch)</span>
        </div>
      </div>
    </header>

    <!-- SECCIÓN 1 -->
    <section class="report-section" id="sec-1">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 01</span>
        <h2>Resumen Ejecutivo y Flujo End-to-End</h2>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Problema y Usuario Objetivo</h3>
          <p>
            Digitalizar caligrafía propia en una fuente instalable (<span class="mono">.ttf</span>) normalmente exige
            trazar manualmente decenas de glifos en editores vectoriales complejos (FontForge, Glyphs). Por otro lado,
            los modelos generativos basados en difusión requieren GPU dedicada y tardan segundos por carácter,
            impidiendo una experiencia interactiva en hardware cotidiano.
          </p>
          <p class="muted" style="margin-bottom:0;">
            <strong>Usuario objetivo:</strong> diseñadores gráficos, ilustradores, docentes y creadores que buscan
            capturar ~300 trazos propios con stylus o plantilla, previsualizar glifos en tiempo real por teclado,
            dibujo libre o cámara web, y exportar una tipografía <span class="mono">.ttf</span> lista para usar.
          </p>
        </div>

        <div class="card">
          <h3>Solución Técnica Integrada</h3>
          <ul style="margin:0; padding-left:18px;">
            <li><strong>Generación condicional:</strong> baseline <span class="mono">Pix2Pix</span> (U-Net 7 niveles + PatchGAN 70×70) y evolución aditiva <span class="mono">StyleUNet + AdaIN</span> con desacoplamiento de contenido y estilo.</li>
            <li><strong>Auditoría objetiva:</strong> dos clasificadores congelados (<span class="mono">ResNet-18</span> y <span class="mono">EfficientNet-B0</span>) para Accuracy, Macro-F1, FID de dominio, Grad-CAM y detección OOD.</li>
            <li><strong>Despliegue ligero:</strong> <span class="mono">ONNX Runtime</span> en CPU a <strong>{kpi_fps:.2f} FPS</strong> ({kpi_ms:.2f} ms, speedup <strong>{kpi_speedup:.2f}×</strong>) + vectorización <span class="mono">vtracer</span> + <span class="mono">fontTools</span>.</li>
          </ul>
        </div>
      </div>

      <div class="card">
        <h3>Arquitectura End-to-End (Flujo Tensorial <span class="mono">1×128×128</span>)</h3>
        <div class="svg-wrap">
          {_svg_architecture_diagram()}
        </div>
      </div>
    </section>

    <!-- SECCIÓN 2 -->
    <section class="report-section" id="sec-2">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 02</span>
        <h2>Datos y Protocolo Anti-Fuga</h2>
      </div>

      <div class="four-col" style="margin-bottom:18px;">
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Vocabulario Unificado</span>
          <div class="kpi-value" style="font-size:21px; margin:4px 0;">{len(CHARSET)} clases</div>
          <div class="muted">Mayús. ({len(UPPERCASE)}) · Minús. ({len(LOWERCASE)}) · Dígitos ({len(DIGITS)}) · Signos ({len(PUNCTUATION)})</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Dataset Sintético</span>
          <div class="kpi-value" style="font-size:21px; margin:4px 0;">3,825 pares</div>
          <div class="muted">17 Google Fonts (SIL OFL 1.1) + Lato-Regular</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Stylus Real Propio</span>
          <div class="kpi-value" style="font-size:21px; margin:4px 0;">300 muestras</div>
          <div class="muted">75 × 4 repeticiones · 300/300 válidas · 2.57% tinta</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Evaluación OOD</span>
          <div class="kpi-value" style="font-size:21px; margin:4px 0;">{ood_data["n_ood"]} imágenes</div>
          <div class="muted">6 categorías adversarias × 12 muestras c/u</div>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Particiones Estrictas Sin Fuga de Información</h3>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Conjunto</th>
                  <th class="num">Train</th>
                  <th class="num">Val</th>
                  <th class="num">Test Unseen</th>
                  <th class="num">Total</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Sintético</strong> (17 fuentes OFL)</td>
                  <td class="mono num">2,688</td>
                  <td class="mono num">576</td>
                  <td class="mono num">561</td>
                  <td class="mono num"><strong>3,825</strong></td>
                </tr>
                <tr>
                  <td><strong>Stylus Real</strong> (4 reps/símbolo)</td>
                  <td class="mono num">192</td>
                  <td class="mono num">64</td>
                  <td class="mono num">44</td>
                  <td class="mono num"><strong>300</strong></td>
                </tr>
                <tr>
                  <td><strong>Clases por split</strong></td>
                  <td class="mono num">64</td>
                  <td class="mono num">64</td>
                  <td class="mono num">11</td>
                  <td class="mono num"><strong>75</strong></td>
                </tr>
              </tbody>
            </table>
          </div>
          <ul style="margin:12px 0 0; padding-left:18px; font-size:13.5px;">
            <li><strong>Aislamiento Zero-Shot:</strong> <span class="mono">set(train) ∩ set(test_unseen) == ∅</span>. Nunca usados en entrenamiento ni selección de checkpoints.</li>
            <li><strong>Separación por fuente / repetición:</strong> 14 fuentes en <span class="mono">train</span> vs. 3 fuentes inéditas en <span class="mono">val</span> (sintético); 3 repeticiones en <span class="mono">train</span> vs. 1 en <span class="mono">val</span> (real).</li>
            <li><strong>Aumentos invariantes a semántica:</strong> afines suaves solo en <span class="mono">train</span> (±4°, ±4% traslación, 0.94–1.06 escala). <strong>Sin flips</strong> horizontales/verticales (<span class="mono">b↔d, p↔q, 6↔9</span>).</li>
          </ul>
        </div>

        <div class="card">
          <h3>11 Símbolos Reservados para Zero-Shot (<span class="mono">test_unseen</span>)</h3>
          <p class="muted" style="margin-top:0;">
            2 Mayúsculas, 2 Minúsculas, 2 Dígitos y 5 Signos básicos aislados en <span class="mono">src/config.py</span>:
          </p>
          <div class="chip-row">{unseen_chips_html}</div>

          <div class="callout teal" style="margin-top:16px; margin-bottom:0;">
            <div class="callout-title">Auditoría Manual de Calidad y Transparencia <span class="mono">judge_split</span></div>
            <p>
              <strong>Sintético:</strong> 0/32 errores (0.0 %) tras ajustar <span class="mono">inner_margin=14 px</span>.
              <strong>Stylus real:</strong> 0/30 errores de etiqueta (0.0 %); 1/30 contacto leve con borde superior; densidad media de tinta <strong>2.57 %</strong> (min 0.07 % en <span class="mono">'.'</span>, max 6.15 %). Los 26/300 (8.7 %) desacuerdos del juez son convenciones personales (<span class="mono">0</span> tachado, <span class="mono">&amp;</span> cursivo). En <span class="mono">synthetic_pairs/test_unseen</span>, las 99 muestras sin traslape de fuente (<span class="mono">judge_split=val</span>) obtienen <strong>89.90 % Acc</strong> y sobre <span class="mono">real_pairs</span> el traslape con los jueces es <strong>0 %</strong>.
            </p>
          </div>
        </div>
      </div>

      {eda_gallery_html}
    </section>

    <!-- SECCIÓN 3 -->
    <section class="report-section" id="sec-3">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 03</span>
        <h2>Arquitecturas y Decisiones Técnicas</h2>
      </div>

      <div class="card">
        <h3>Inventario de Modelos (Parámetros Calculados en Vivo desde <span class="mono">src/models/</span>)</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Arquitectura / Rol</th>
                <th class="num">Parámetros exactos</th>
                <th class="num">Escala</th>
                <th>Entrada</th>
                <th>Salida</th>
                <th>Detalles clave</th>
              </tr>
            </thead>
            <tbody>{arch_rows_html}</tbody>
          </table>
        </div>
        <div class="svg-wrap" style="margin-top:14px;">
          {_svg_model_params_chart(model_rows)}
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Hiperparámetros Centralizados (<span class="mono">src/config.py</span>)</h3>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Bloque</th>
                  <th>Épocas</th>
                  <th>Batch / LR</th>
                  <th>Regularización / Pérdidas</th>
                </tr>
              </thead>
              <tbody>{hparams_rows_html}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <h3>Matriz de Decisiones con Evidencia Propia</h3>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Comparativa</th>
                  <th>Evidencia y Decisión Adoptada</th>
                </tr>
              </thead>
              <tbody>
                <tr class="row-highlight">
                  <td><strong>Pix2Pix vs. StyleUNet + AdaIN</strong></td>
                  <td>Experimento propio (<span class="mono">03</span> vs <span class="mono">04</span>): desacoplar contenido y estilo (4 refs sin <span class="mono">InstanceNorm</span> + <span class="mono">AdaIN</span> + aumento morfológico) igualó el techo humano en <span class="mono">val</span> (<strong>89.06 % Acc</strong>) y elevó Zero-Shot a <strong>72.73 %</strong> (+63.64 pp vs Pix2Pix FT).</td>
                </tr>
                <tr>
                  <td><strong>ResNet-18 vs. EfficientNet-B0</strong></td>
                  <td>Experimento propio: <span class="mono">ResNet-18</span> superó a <span class="mono">EffNet-B0</span> en val (87.67 % vs 85.07 %), OOD (<strong>AUROC 0.9435</strong> vs 0.8856) y latencia CPU (4.69 ms). Elegido como juez principal; <span class="mono">EffNet-B0</span> actúa como validación cruzada.</td>
                </tr>
                <tr>
                  <td><strong>ONNX vs. PyTorch CPU</strong></td>
                  <td>Benchmark propio (<span class="mono">export_onnx.py</span>): <span class="mono">intra_op_num_threads=4</span> bajó el pipeline de 35.61 ms (28.08 FPS) a <strong>21.07 ms (47.45 FPS, 1.69×)</strong> con error numérico &lt;1e-5.</td>
                </tr>
                <tr>
                  <td><strong>Descarte Diffusion y CycleGAN</strong></td>
                  <td>Restricciones de latencia y pares: Diffusion exige GPU &gt;6 GB VRAM (&lt;1 FPS en CPU). CycleGAN sin pares ni pérdida L1 deforma topologías Zero-Shot; las skip connections + L1 (λ=100) preservan la geometría base.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- SECCIÓN 4 -->
    <section class="report-section" id="sec-4">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 04</span>
        <h2>Experimentos y Evolución de Resultados</h2>
      </div>

      <div class="card">
        <h3>Explorador Interactivo de Resultados (Dominio × Split)</h3>
        <p class="muted" style="margin-top:0;">
          Alterna entre <strong>Dominio Stylus Real</strong> y <strong>Dominio Sintético</strong> para inspeccionar
          Accuracy, Macro-F1, L1 y FID de dominio en tiempo real sobre ambos jueces congelados.
        </p>

        <div class="exp-toolbar" role="tablist" aria-label="Selector de dominio y split">
          <button type="button" class="exp-btn is-active" data-exp-key="real_val">Stylus Real · Val (64 vistas)</button>
          <button type="button" class="exp-btn" data-exp-key="real_unseen">Stylus Real · Zero-Shot (11 unseen)</button>
          <button type="button" class="exp-btn" data-exp-key="syn_val">Sintético · Val (64 vistas)</button>
          <button type="button" class="exp-btn" data-exp-key="syn_unseen">Sintético · Zero-Shot (11 unseen)</button>
        </div>

        <div id="exp-header">
          <h4 id="exp-title" style="margin:0 0 4px; font-size:15.5px;"></h4>
          <p id="exp-subtitle" class="muted" style="margin:0;"></p>
        </div>

        <div class="exp-charts-grid">
          <div class="bar-card">
            <h4>Legibilidad: Accuracy y Macro-F1 (ResNet-18 vs. EfficientNet-B0)</h4>
            <div class="legend-inline">
              <span><i class="legend-dot" style="background:var(--accent);"></i>Accuracy ResNet-18 (%)</span>
              <span><i class="legend-dot" style="background:var(--accent-2);"></i>Macro-F1 ResNet-18 (%)</span>
            </div>
            <div id="exp-acc-bars"></div>
          </div>
          <div class="bar-card">
            <h4>Fidelidad de Dominio: FID ResNet-18 (↓ menor es mejor)</h4>
            <div class="legend-inline">
              <span><i class="legend-dot" style="background:var(--accent-3);"></i>FID sobre embeddings 512-d (Referencia y = 0.00)</span>
            </div>
            <div id="exp-fid-bars"></div>
          </div>
        </div>

        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo / Fuente</th>
                <th class="num">Acc ResNet-18</th>
                <th class="num">Acc EffNet-B0</th>
                <th class="num">F1 ResNet-18</th>
                <th class="num">F1 EffNet-B0</th>
                <th class="num">L1 media</th>
                <th class="num">FID dominio</th>
              </tr>
            </thead>
            <tbody id="exp-table-body"></tbody>
          </table>
        </div>

        <div class="callout teal" style="margin-top:14px; margin-bottom:0;">
          <p id="exp-note"></p>
        </div>
      </div>

      <div class="callout warn">
        <div class="callout-title">Hallazgo Experimental Central: Choque de Grosor de Trazo → Resolución con StyleUNet + AdaIN</div>
        <p>
          El stylus real tiene una densidad media de tinta de apenas <strong>2.57 %</strong>, mucho más fina que el promedio de las 17 fuentes sintéticas.
          En <span class="mono">Pix2Pix</span> estándar (donde estilo y contenido se mezclan en los mismos pesos), <span class="mono">pix2pix_scratch</span> (<strong>75.00 % Acc</strong>, FID <strong>94.87</strong>)
          superó a <span class="mono">pix2pix_finetuned</span> (62.50 %) en <span class="mono">val</span>, pero ambos colapsaron en Zero-Shot (<strong>9.09–27.27 %</strong>); además, seleccionar únicamente por mínimo <span class="mono">val_l1</span> provocó colapso a lienzos casi en blanco (<span class="mono">val_l1 = 0.0455</span>).
          Al migrar a <strong>StyleUNet + AdaIN</strong> (<span class="mono">StyleEncoder</span> sin <span class="mono">InstanceNorm</span> + aumento morfológico emparejado de grosor + pérdida <span class="mono">InkDiff</span>), el modelo transfirió el trazo fino con <strong>0 épocas de fine-tuning</strong> (<span class="mono">style_unet_fewshot</span>: 84.38 % val / 54.55 % Zero-Shot) y con solo 4 épocas efectivas (<span class="mono">style_unet_finetuned</span>) alcanzó el techo humano en <span class="mono">val</span> (<strong>89.06 % Acc / 85.94 % Macro-F1</strong>) y disparó Zero-Shot a <strong>72.73 %</strong> (<strong>+63.64 pp</strong> vs <span class="mono">pix2pix_finetuned</span>, FID <strong>499.83 → 302.42</strong>).
        </p>
      </div>

      <h3 style="margin:22px 0 10px;">Curvas de Entrenamiento y Control de Sobreajuste</h3>
      {curves_gallery_html}

      <h3 style="margin:22px 0 10px;">Grids Visuales de Síntesis (<span class="mono">x | ŷ | y</span>) y Comparativas</h3>
      {grids_gallery_html}
    </section>

    <!-- SECCIÓN 5 -->
    <section class="report-section" id="sec-5">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 05</span>
        <h2>Auditoría por Subgrupos y Análisis de Errores</h2>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Desempeño por Subgrupo Tipográfico (<span class="mono">val</span> · ResNet-18)</h3>
          <p class="muted" style="margin-top:0;">
            Mayúsculas alcanza <strong>77.8 %</strong> Macro-F1 mientras Signos cae a <strong>45.8 %</strong> — una brecha
            exacta de <strong>{subgroups_pretrain["val_gap_macro_f1_pp"]:.2f} pp</strong> asociada a signos con muy baja área de tinta.
          </p>
          <div class="svg-wrap">
            {_svg_subgroups_chart(subgroups_pretrain)}
          </div>
        </div>

        <div class="card">
          <h3>Taxonomía de los {total_errors} Errores (<span class="mono">metrics_pretrain.json</span>)</h3>
          <p class="muted" style="margin-top:0;">
            El 65.4 % (306/{total_errors}) se concentra en artefactos Zero-Shot de símbolos complejos no vistos; en <span class="mono">val</span> dominan trazos ambiguos y similitudes morfológicas.
          </p>
          <div class="svg-wrap">
            {_svg_error_taxonomy_chart(metrics_pretrain)}
          </div>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Peores Caracteres en Evaluación Pretrain (12 casos)</h3>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Real</th>
                  <th>Split</th>
                  <th>Subgrupo</th>
                  <th class="num">N</th>
                  <th class="num">Acc</th>
                  <th class="num">Conf.</th>
                  <th class="num">L1</th>
                  <th>Top Pred</th>
                </tr>
              </thead>
              <tbody>{worst_rows_html}</tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <h3>Desglose de los 11 Símbolos Zero-Shot (<span class="mono">pix2pix_pretrained</span>)</h3>
          <p class="muted" style="margin-top:0;">
            5 símbolos (<span class="mono">?, k, +, ;, z</span>) logran 100 % de acierto en sintético; los 6 fallidos caen con confianza &lt;0.32 (interceptables por el filtro OOD a τ=0.70).
          </p>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Char</th>
                  <th>Subgrupo</th>
                  <th class="num">Acc</th>
                  <th class="num">Conf.</th>
                  <th class="num">L1</th>
                  <th>Pred</th>
                </tr>
              </thead>
              <tbody>{unseen_char_rows_html}</tbody>
            </table>
          </div>
        </div>
      </div>

      {subgroups_gallery_html}
    </section>

    <!-- SECCIÓN 6 -->
    <section class="report-section" id="sec-6">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 06</span>
        <h2>Interpretabilidad Espacial (Grad-CAM)</h2>
      </div>

      <div class="four-col" style="margin-bottom:18px;">
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">val · Aciertos (n=80)</span>
          <div class="kpi-value" style="color:var(--accent-2); margin:4px 0;">{av["mean_cos_correct"]:.4f}</div>
          <div class="muted">Global: {av["mean_cos_all"]:.4f} ± {av["std_cos_all"]:.4f}</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">val · Errores</span>
          <div class="kpi-value" style="color:var(--warning); margin:4px 0;">{av["mean_cos_wrong"]:.4f}</div>
          <div class="muted">Brecha: +{av["mean_cos_correct"] - av["mean_cos_wrong"]:.4f}</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Zero-Shot · Aciertos</span>
          <div class="kpi-value" style="color:var(--accent-2); margin:4px 0;">{au["mean_cos_correct"]:.4f}</div>
          <div class="muted">Global: {au["mean_cos_all"]:.4f} ± {au["std_cos_all"]:.4f}</div>
        </div>
        <div class="card" style="margin-bottom:0;">
          <span class="kpi-label">Zero-Shot · Errores</span>
          <div class="kpi-value" style="color:var(--danger); margin:4px 0;">{au["mean_cos_wrong"]:.4f}</div>
          <div class="muted">Brecha: <strong>+{au["mean_cos_correct"] - au["mean_cos_wrong"]:.4f}</strong></div>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Alineación Espacial <span class="mono">Grad-CAM(y)</span> vs. <span class="mono">Grad-CAM(ŷ)</span></h3>
          <p class="muted" style="margin-top:0;">
            Cuando el generador sintetiza un glifo legible —incluso en símbolos Zero-Shot— <span class="mono">ResNet-18</span>
            atiende exactamente los mismos rasgos estructurales que en la referencia real (<span class="mono">cos = 0.9895</span>).
          </p>
          <div class="svg-wrap">
            {_svg_gradcam_alignment_chart(gradcam_pretrain)}
          </div>
        </div>

        <div class="card">
          <h3>Consistencia entre Jueces Independientes</h3>
          <p class="muted" style="margin-top:0;">
            Validación cruzada <span class="mono">ResNet-18</span> (<span class="mono">layer4[-1]</span>) vs. <span class="mono">EfficientNet-B0</span> (<span class="mono">features[-1]</span>) sobre los mismos glifos ŷ:
          </p>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Char</th>
                  <th>Split</th>
                  <th>ResNet-18</th>
                  <th>EfficientNet-B0</th>
                  <th class="num">Cos. entre jueces</th>
                </tr>
              </thead>
              <tbody>{cross_judge_rows_html}</tbody>
            </table>
          </div>
          <p class="muted" style="margin:10px 0 0;">
            En glifos limpios (<span class="mono">A, e, k, ?</span>) la concordancia espacial es <strong>0.90–0.95</strong>; en <span class="mono">'g'</span> la caída a <strong>0.3952</strong> delata una espuela superior derecha que un solo juez ocultaría.
          </p>
        </div>
      </div>

      <div class="card">
        <h3>4 Causas Físicas de Error Diagnosticadas con Grad-CAM (<span class="mono">docs/interpretabilidad.md</span>)</h3>
        <div class="four-col">
          <div class="bar-card">
            <h4>1. Pérdida de bucles y curvas</h4>
            <p class="muted" style="margin:0;">
              <strong><span class="mono">6 → C</span> (0.92) y <span class="mono">S → C</span> (0.57):</strong> el promediado L1 sobre 17 estilos abre el bucle inferior del <span class="mono">6</span> o la curva baja de la <span class="mono">S</span>. Se mitiga al fijar un solo estilo.
            </p>
          </div>
          <div class="bar-card">
            <h4>2. Borrado / deformación de puntos</h4>
            <p class="muted" style="margin:0;">
              <strong><span class="mono">! → l</span> (0.70), <span class="mono">i → l</span> (0.62) y <span class="mono">: → ;</span> (0.76):</strong> componentes desconectados pequeños pesan poco en L1; Grad-CAM muestra foco ausente en el punto.
            </p>
          </div>
          <div class="bar-card">
            <h4>3. Ambigüedad sans-serif intrínseca</h4>
            <p class="muted" style="margin:0;">
              <strong><span class="mono">l → I</span> (0.57):</strong> incluso la referencia manuscrita <span class="mono">y</span> solo alcanza 0.56. En captura personal se desambigua marcando serifas sutiles en <span class="mono">I</span>.
            </p>
          </div>
          <div class="bar-card">
            <h4>4. Trazos complejos Zero-Shot</h4>
            <p class="muted" style="margin:0;">
              <strong><span class="mono">4 → &amp;</span>, <span class="mono">Q → U</span>, <span class="mono">G → O</span> (0.29–0.32):</strong> pérdida de colas/cruces inéditos con confianza &lt;0.35, <strong>interceptados al 100 %</strong> por el umbral OOD τ=0.70.
            </p>
          </div>
        </div>
      </div>

      {gradcam_gallery_html}
    </section>

    <!-- SECCIÓN 7 -->
    <section class="report-section" id="sec-7">
      <div class="section-head">
        <span class="section-tag">SECCIÓN 07</span>
        <h2>Robustez OOD, Despliegue ONNX y Producto Final (.ttf + Ética)</h2>
      </div>

      <div class="two-col">
        <div class="card">
          <h3>Separación ROC y Calibración OOD (72 imágenes)</h3>
          <div class="table-wrap">
            <table class="compact-table">
              <thead>
                <tr>
                  <th>Escenario</th>
                  <th class="num">AUROC</th>
                  <th class="num">Rech. @0.70</th>
                  <th class="num">Falso ID</th>
                  <th class="num">τ Youden</th>
                  <th class="num">Rech. ópt.</th>
                </tr>
              </thead>
              <tbody>{ood_scenarios_html}</tbody>
            </table>
          </div>
          <p class="muted" style="margin:10px 0 0;">
            En la demo (<span class="mono">app/demo_realtime.py</span>), el usuario ajusta τ en vivo con <span class="mono">+</span>/<span class="mono">-</span> (rango 0.10–0.95, default <strong>0.70</strong>).
          </p>
        </div>

        <div class="card">
          <h3>Rechazo OOD por Categoría a <span class="mono">τ = 0.70</span></h3>
          <div class="svg-wrap">
            {_svg_ood_categories_chart(ood_data)}
          </div>
          <p class="muted" style="margin:6px 0 0;">
            En <span class="mono">desenfoque_corte</span> el rechazo del pipeline baja a 33.3 % porque la U-Net restaura trazos ligeramente borrosos hacia letras legibles.
          </p>
        </div>
      </div>

      <div class="card">
        <h3>Benchmark CPU: PyTorch vs. ONNX Runtime (<span class="mono">results/benchmark_onnx.json</span>)</h3>
        <p class="muted" style="margin-top:0;">
          Entorno verificado: <strong>{_esc(bench_onnx["hardware"]["os"])}</strong> · {_esc(bench_onnx["hardware"]["processor"])} ({bench_onnx["hardware"]["cpu_count"]} núcleos lógicos, <span class="mono">intra_op_num_threads=4</span>) · Python <span class="mono">{_esc(bench_onnx["hardware"]["python"])}</span> · PyTorch <span class="mono">{_esc(bench_onnx["hardware"]["torch"])}</span> · ONNX Runtime <span class="mono">{_esc(bench_onnx["hardware"]["onnxruntime"])}</span> ({bench_onnx["runs"]} inferencias tras {bench_onnx["warmup"]} de calentamiento, <span class="mono">batch_size=1</span>).
        </p>

        <div class="svg-wrap">
          {_svg_onnx_benchmark_chart(bench_onnx)}
        </div>

        <div class="table-wrap" style="margin-top:12px;">
          <table class="compact-table">
            <thead>
              <tr>
                <th>Etapa</th>
                <th class="num">PyTorch CPU (ms)</th>
                <th class="num">ONNX CPU (ms)</th>
                <th class="num">ONNX p95 (ms)</th>
                <th class="num">ONNX FPS</th>
                <th class="num">Speedup</th>
                <th class="num">Tamaño ONNX</th>
                <th class="num">Error máx. abs.</th>
              </tr>
            </thead>
            <tbody>{bench_rows_html}</tbody>
          </table>
        </div>
      </div>

      {ood_demo_gallery_html}

      <div class="card">
        <h3>Producto Final: Tipografías Instalables (<span class="mono">.ttf</span>) y Trazabilidad OpenType</h3>
        <p class="muted" style="margin-top:0;">
          Compilación automática mediante <span class="mono">src/export/build_ttf.py</span> (<span class="mono">vtracer</span> → <span class="mono">Cu2QuPen</span> → <span class="mono">fontTools</span>).
          Las vistas previas y tablas OpenType <span class="mono">name</span> siguientes fueron leídas directamente de los binarios <span class="mono">results/*.ttf</span> en disco.
        </p>
        {font_previews_html}
        <div class="two-col" style="margin-top:18px;">
          {ttf_cards_html}
        </div>
        <div class="callout" style="margin-top:14px; margin-bottom:0;">
          <div class="callout-title">Salvaguarda Ética en Metadatos OpenType</div>
          <p>
            <span class="mono">HandwrittenCGAN_Pretrain.ttf</span> ya incluye en disco el bloque completo de procedencia <span class="mono">AI-SYNTHETIC</span> (<span class="mono">copyright</span>, <span class="mono">uniqueFontIdentifier</span>, <span class="mono">manufacturer</span>, <span class="mono">designer</span>, <span class="mono">description</span>, <span class="mono">licenseDescription</span>).
            El binario actual <span class="mono">MiLetra_Stylus.ttf</span> conserva los registros base (<span class="mono">nameID 1–6</span>) de una compilación previa; al recompilar con <span class="mono">python -m src.export.build_ttf --from-real-dir data/real_pairs</span> se incrusta automáticamente el marcador extendido <span class="mono">PERSONAL-STYLUS</span>.
          </p>
        </div>
      </div>

      <div class="card">
        <h3>Costos Medidos de Cómputo y Escalabilidad (<span class="mono">docs/costos.md</span>)</h3>
        <div class="table-wrap">
          <table class="compact-table">
            <thead>
              <tr>
                <th>Etapa / Modelo</th>
                <th>Dataset</th>
                <th>Épocas</th>
                <th class="num">Tiempo T4</th>
                <th class="num">Velocidad</th>
                <th class="num">Checkpoint</th>
                <th class="num">Costo real</th>
                <th class="num">Equiv. T4 ($0.35/h)</th>
              </tr>
            </thead>
            <tbody>{costs_rows_html}</tbody>
          </table>
        </div>
        <p class="muted" style="margin:10px 0 0;">
          <strong>Escalabilidad (1,000 usuarios/mes):</strong> con <span class="mono">StyleUNet</span>, la adaptación Few-Shot con 4 glifos cuesta <strong>$0.00 GPU</strong> por usuario; un fine-tuning ligero de ~2.6 min en T4 cuesta <strong>~$0.015 USD por fuente</strong> (~$15 USD/mes), mientras la inferencia interactiva corre en el cliente a 47.45 FPS CPU.
        </p>
      </div>

      <div class="card">
        <h3>Limitaciones Técnicas y Uso Responsable</h3>
        <div class="three-col">
          <div class="bar-card">
            <h4>Glifos aislados y alfabeto base</h4>
            <p class="muted" style="margin:0;">Opera carácter por carácter (<span class="mono">1×128×128</span>) sin ligaduras cursivas continuas ni presión dinámica. Entrenado sobre 75 símbolos ASCII; <span class="mono">ñ, Ñ, ¿, ¡</span> son experimentales en inferencia.</p>
          </div>
          <div class="bar-card">
            <h4>Varianza Zero-Shot y signos finos</h4>
            <p class="muted" style="margin:0;">Cada una de las 11 clases en <span class="mono">test_unseen</span> pesa ~9.09 pp en Accuracy. Puntos aislados (<span class="mono">!, i, :</span>) y pares sans-serif (<span class="mono">l/I/1, 0/O</span>) concentran los errores residuales.</p>
          </div>
          <div class="bar-card">
            <h4>Prohibición de suplantación</h4>
            <p class="muted" style="margin:0;">Estrictamente prohibido clonar escritura de terceros sin consentimiento o falsificar firmas/documentos. Datos restringidos a SIL OFL 1.1 y capturas propias con trazabilidad OpenType.</p>
          </div>
        </div>
      </div>
    </section>

    <footer class="report-footer">
      <span><strong>Handwritten Font cGAN</strong> · Reporte Técnico Autocontenido generado por <span class="mono">src/export/build_report.py</span></span>
      <span class="mono">Cero dependencias externas · 100 % Offline · {_esc(generated_ts)}</span>
    </footer>
  </main>
</div>

<button type="button" class="back-to-top" id="back-to-top" aria-label="Volver arriba">↑ Arriba</button>

<!-- Modal Lightbox -->
<div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-labelledby="lightbox-title" hidden>
  <div class="lightbox-dialog">
    <div class="lightbox-bar">
      <div class="lightbox-head-left">
        <span class="lightbox-counter" id="lightbox-counter">1 / 1</span>
        <span class="lightbox-title" id="lightbox-title"></span>
      </div>
      <div class="lightbox-actions">
        <button type="button" class="lightbox-nav-btn" id="lightbox-prev" aria-label="Imagen anterior">← Anterior</button>
        <button type="button" class="lightbox-nav-btn" id="lightbox-next" aria-label="Imagen siguiente">Siguiente →</button>
        <button type="button" class="lightbox-close" id="lightbox-close" aria-label="Cerrar vista ampliada">Cerrar (ESC)</button>
      </div>
    </div>
    <div class="lightbox-body">
      <div class="lightbox-plate">
        <img id="lightbox-img" src="" alt=""/>
      </div>
    </div>
    <div class="lightbox-caption" id="lightbox-caption"></div>
  </div>
</div>

<script>
(function () {{
  "use strict";

  var EXPERIMENTS = {experiments_json};

  /* 1. Barra de progreso de lectura + botón Volver Arriba */
  var progressBar = document.getElementById("reading-progress");
  var backToTopBtn = document.getElementById("back-to-top");

  function updateScrollUi() {{
    var docEl = document.documentElement;
    var scrollTop = window.pageYOffset || docEl.scrollTop || 0;
    var maxScroll = Math.max(1, docEl.scrollHeight - window.innerHeight);
    var pct = Math.min(100, Math.max(0, (scrollTop / maxScroll) * 100));
    if (progressBar) {{
      progressBar.style.width = pct.toFixed(2) + "%";
    }}
    if (backToTopBtn) {{
      backToTopBtn.classList.toggle("is-visible", scrollTop > 320);
    }}
  }}

  window.addEventListener("scroll", updateScrollUi, {{ passive: true }});
  window.addEventListener("resize", updateScrollUi, {{ passive: true }});
  updateScrollUi();

  if (backToTopBtn) {{
    backToTopBtn.addEventListener("click", function () {{
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
  }}

  /* 2. Scrollspy para barra lateral y barra móvil */
  var sections = Array.prototype.slice.call(document.querySelectorAll(".report-section"));
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav-link, .mobile-nav-strip a"));

  function setActiveNav(id) {{
    navLinks.forEach(function (link) {{
      var match = link.getAttribute("href") === "#" + id;
      link.classList.toggle("is-active", match);
    }});
  }}

  if ("IntersectionObserver" in window) {{
    var observer = new IntersectionObserver(
      function (entries) {{
        entries.forEach(function (entry) {{
          if (entry.isIntersecting) {{
            setActiveNav(entry.target.id);
          }}
        }});
      }},
      {{ rootMargin: "-20% 0px -65% 0px", threshold: 0.01 }}
    );
    sections.forEach(function (sec) {{ observer.observe(sec); }});
  }}

  /* 3. Selector interactivo de Experimentos (Sección 4) */
  var expButtons = Array.prototype.slice.call(document.querySelectorAll("[data-exp-key]"));
  var expTitle = document.getElementById("exp-title");
  var expSubtitle = document.getElementById("exp-subtitle");
  var expAccBars = document.getElementById("exp-acc-bars");
  var expFidBars = document.getElementById("exp-fid-bars");
  var expTableBody = document.getElementById("exp-table-body");
  var expNote = document.getElementById("exp-note");

  function fmtPct(v) {{
    return v === null || v === undefined ? "—" : v.toFixed(2) + " %";
  }}
  function fmtNum(v, d) {{
    return v === null || v === undefined ? "—" : v.toFixed(d);
  }}
  function escapeHtml(s) {{
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }}

  function renderExperiment(key) {{
    var data = EXPERIMENTS[key];
    if (!data) {{ return; }}

    expButtons.forEach(function (b) {{
      b.classList.toggle("is-active", b.getAttribute("data-exp-key") === key);
    }});

    expTitle.textContent = data.title;
    expSubtitle.textContent = data.subtitle;
    expNote.textContent = data.note;

    var maxFid = 1;
    data.rows.forEach(function (r) {{
      if (r.fid !== null && r.fid > maxFid) {{ maxFid = r.fid; }}
    }});

    var accHtml = "";
    var fidHtml = "";
    var tblHtml = "";

    data.rows.forEach(function (r) {{
      var cls = r.highlight ? "winner" : "r18";
      accHtml +=
        '<div class="hbar-row">' +
          '<div class="hbar-head">' +
            '<span class="hbar-name">' + escapeHtml(r.label) + '</span>' +
            '<span class="hbar-val">Acc ' + r.r18_acc.toFixed(2) + '% · F1 ' + r.r18_f1.toFixed(2) + '%</span>' +
          '</div>' +
          '<div class="hbar-track"><div class="hbar-fill ' + cls + '" style="width:' + Math.min(100, r.r18_acc).toFixed(2) + '%"></div></div>' +
          '<div class="hbar-track" style="height:7px;"><div class="hbar-fill eff" style="width:' + Math.min(100, r.r18_f1).toFixed(2) + '%"></div></div>' +
        '</div>';

      var fidW = r.fid === 0 ? 2 : Math.max(4, (r.fid / maxFid) * 100);
      fidHtml +=
        '<div class="hbar-row">' +
          '<div class="hbar-head">' +
            '<span class="hbar-name">' + escapeHtml(r.label) + '</span>' +
            '<span class="hbar-val">' + fmtNum(r.fid, 2) + '</span>' +
          '</div>' +
          '<div class="hbar-track"><div class="hbar-fill ' + (r.highlight ? "eff" : "fid") + '" style="width:' + fidW.toFixed(2) + '%"></div></div>' +
        '</div>';

      tblHtml +=
        '<tr class="' + (r.highlight ? "row-highlight" : "") + '">' +
          '<td><strong>' + escapeHtml(r.label) + '</strong><div class="sub-label">' + escapeHtml(r.tag) + '</div></td>' +
          '<td class="mono num"><strong>' + fmtPct(r.r18_acc) + '</strong></td>' +
          '<td class="mono num">' + fmtPct(r.eff_acc) + '</td>' +
          '<td class="mono num"><strong>' + fmtPct(r.r18_f1) + '</strong></td>' +
          '<td class="mono num">' + fmtPct(r.eff_f1) + '</td>' +
          '<td class="mono num">' + fmtNum(r.l1, 4) + '</td>' +
          '<td class="mono num"><strong>' + fmtNum(r.fid, 2) + '</strong></td>' +
        '</tr>';
    }});

    expAccBars.innerHTML = accHtml;
    expFidBars.innerHTML = fidHtml;
    expTableBody.innerHTML = tblHtml;
  }}

  expButtons.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      renderExperiment(btn.getAttribute("data-exp-key"));
    }});
  }});
  renderExperiment("real_val");

  /* 4. Pestañas de galerías */
  var tabButtons = Array.prototype.slice.call(document.querySelectorAll("[data-tab-group][data-tab-target]"));
  tabButtons.forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var group = btn.getAttribute("data-tab-group");
      var targetId = btn.getAttribute("data-tab-target");

      document.querySelectorAll('[role="tab"][data-tab-group="' + group + '"]').forEach(function (t) {{
        var isActive = t === btn;
        t.classList.toggle("is-active", isActive);
        t.setAttribute("aria-selected", isActive ? "true" : "false");
      }});

      document.querySelectorAll('[role="tabpanel"][data-tab-group="' + group + '"]').forEach(function (panel) {{
        var show = panel.id === targetId;
        panel.classList.toggle("is-active", show);
        panel.hidden = !show;
      }});
    }});
  }});

  /* 5. Lightbox modal con navegación ← / → para todas las figuras */
  var lightbox = document.getElementById("lightbox");
  var lbImg = document.getElementById("lightbox-img");
  var lbTitle = document.getElementById("lightbox-title");
  var lbCaption = document.getElementById("lightbox-caption");
  var lbCounter = document.getElementById("lightbox-counter");
  var lbPrev = document.getElementById("lightbox-prev");
  var lbNext = document.getElementById("lightbox-next");
  var lbClose = document.getElementById("lightbox-close");
  var triggers = Array.prototype.slice.call(document.querySelectorAll("[data-lightbox-src]"));
  var lbItems = triggers.map(function (el) {{
    return {{
      src: el.getAttribute("data-lightbox-src") || "",
      title: el.getAttribute("data-lightbox-title") || "",
      caption: el.getAttribute("data-lightbox-caption") || ""
    }};
  }});
  var currentLbIndex = 0;
  var lastFocused = null;

  function showLightboxAt(idx) {{
    if (!lbItems.length) {{ return; }}
    currentLbIndex = ((idx % lbItems.length) + lbItems.length) % lbItems.length;
    var item = lbItems[currentLbIndex];
    lbImg.setAttribute("src", item.src);
    lbImg.setAttribute("alt", item.title);
    lbTitle.textContent = item.title;
    lbCaption.textContent = item.caption;
    if (lbCounter) {{
      lbCounter.textContent = (currentLbIndex + 1) + " / " + lbItems.length;
    }}
  }}

  function openLightbox(idx) {{
    lastFocused = document.activeElement;
    showLightboxAt(idx);
    lightbox.hidden = false;
    lightbox.classList.add("is-open");
    document.body.style.overflow = "hidden";
    lbClose.focus();
  }}

  function closeLightbox() {{
    lightbox.classList.remove("is-open");
    lightbox.hidden = true;
    document.body.style.overflow = "";
    lbImg.setAttribute("src", "");
    if (lastFocused && typeof lastFocused.focus === "function") {{
      lastFocused.focus();
    }}
  }}

  triggers.forEach(function (trigger, idx) {{
    trigger.addEventListener("click", function () {{
      openLightbox(idx);
    }});
  }});

  if (lbPrev) {{
    lbPrev.addEventListener("click", function () {{ showLightboxAt(currentLbIndex - 1); }});
  }}
  if (lbNext) {{
    lbNext.addEventListener("click", function () {{ showLightboxAt(currentLbIndex + 1); }});
  }}
  lbClose.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", function (ev) {{
    if (ev.target === lightbox) {{
      closeLightbox();
    }}
  }});
  document.addEventListener("keydown", function (ev) {{
    if (!lightbox.classList.contains("is-open")) {{ return; }}
    if (ev.key === "Escape") {{
      closeLightbox();
    }} else if (ev.key === "ArrowLeft") {{
      ev.preventDefault();
      showLightboxAt(currentLbIndex - 1);
    }} else if (ev.key === "ArrowRight") {{
      ev.preventDefault();
      showLightboxAt(currentLbIndex + 1);
    }}
  }});
}})();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el reporte HTML autocontenido (docs/reporte_completo.html) desde los artefactos reales."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta de salida del reporte HTML (default: docs/reporte_completo.html).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = build_report_html()
    output_path.write_text(html_content, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024.0
    print(f"[OK] Reporte HTML autocontenido generado: {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
