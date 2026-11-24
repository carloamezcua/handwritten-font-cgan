"""Captura rápida de >=50 imágenes reales OOD / en condiciones reales con cámara web.

Permite cubrir el criterio B6 con imágenes capturadas por el autor (papel, inclinación,
sombras, garabatos y símbolos inéditos) y evaluar la tasa de rechazo OOD del pipeline ONNX.

Controles en vivo:
- `1`, `2`, `3`, `4`: selecciona categoría activa (`papel_real`, `sombra_inclinacion`,
  `garabato_ruido`, `simbolo_inedito`).
- `SPACE`: captura 1 muestra del ROI preprocesado (`128x128`) y actualiza `metadata.csv`.
- `A`: alterna ráfaga automática (1 captura cada 0.5 s) para llegar a 50+ muestras rápido.
- `F`: voltear espejo horizontal.
- `E`: ejecutar evaluación rápida con `generator.onnx` + `judge_resnet18.onnx` sobre lo capturado.
- `ESC`: salir.

Uso:
    python -m src.data.capturar_ood_real --camera 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from app.demo_realtime import _create_ort_session, _preprocess_webcam_roi, _softmax
from src.config import IDX_TO_CHAR

CATEGORIES = [
    ("1", "papel_real"),
    ("2", "sombra_inclinacion"),
    ("3", "garabato_ruido"),
    ("4", "simbolo_inedito"),
]


def evaluate_captured_ood(
    ood_dir: Path,
    gen_onnx: Path,
    judge_onnx: Path,
    threshold: float = 0.70,
) -> dict:
    meta_path = ood_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe {meta_path}. Captura muestras primero.")

    gen_sess = _create_ort_session(gen_onnx)
    judge_sess = _create_ort_session(judge_onnx)

    rows = list(csv.DictReader(meta_path.open("r", encoding="utf-8")))
    by_cat: dict[str, list[float]] = {}
    all_confs: list[float] = []

    for row in rows:
        img_p = ood_dir / row["image_path"]
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x_norm = (img.astype(np.float32) / 127.5) - 1.0
        y_hat = gen_sess.run(["y_hat"], {"x": x_norm[None, None, :, :]})[0]
        logits = judge_sess.run(["logits"], {"glyph": y_hat})[0][0]
        probs = _softmax(logits)
        conf = float(np.max(probs))
        all_confs.append(conf)
        by_cat.setdefault(row["category"], []).append(conf)

    summary = {
        "n_samples": len(all_confs),
        "threshold": threshold,
        "global_mean_conf": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
        "global_reject_rate": round(float(np.mean([c < threshold for c in all_confs])), 4)
        if all_confs
        else 0.0,
        "by_category": {
            cat: {
                "n": len(vals),
                "mean_conf": round(float(np.mean(vals)), 4),
                "reject_rate": round(float(np.mean([c < threshold for c in vals])), 4),
            }
            for cat, vals in by_cat.items()
        },
    }
    out_json = Path("results/eval/robustez_ood_real.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Evaluación OOD real guardada en: {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura >=50 muestras OOD reales con cámara web.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ood_real"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if args.eval_only:
        evaluate_captured_ood(
            args.output_dir,
            Path("checkpoints/generator.onnx"),
            Path("checkpoints/judge_resnet18.onnx"),
            threshold=args.threshold,
        )
        return

    img_dir = args.output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.csv"

    existing_rows: list[dict[str, str]] = []
    if meta_path.exists():
        existing_rows = list(csv.DictReader(meta_path.open("r", encoding="utf-8")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}.")

    cat_idx = 0
    flip = False
    auto_mode = False
    last_snap = 0.0
    win = "Captura OOD Real (Webcam) — Meta >= 50 muestras"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            side = min(h, w) // 2
            x1, y1 = (w - side) // 2, (h - side) // 2
            x2, y2 = x1 + side, y1 + side
            roi = frame[y1:y2, x1:x2]
            x_u8, is_valid = _preprocess_webcam_roi(roi)

            cat_key, cat_name = CATEGORIES[cat_idx]
            vis = frame.copy()
            color = (0, 220, 90) if is_valid else (40, 80, 240)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            thumb = cv2.cvtColor(cv2.resize(x_u8, (128, 128)), cv2.COLOR_GRAY2BGR)
            vis[10:138, w - 138 : w - 10] = thumb

            cv2.putText(
                vis,
                f"Cat [{cat_key}]: {cat_name} | Total: {len(existing_rows)}/50 | Auto: {'ON' if auto_mode else 'OFF'}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "SPACE: Capturar | 1-4: Categoria | A: Auto (0.5s) | F: Espejo | E: Evaluar | ESC: Salir",
                (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            do_capture = key == 32 or (auto_mode and (time.time() - last_snap) >= 0.5)
            if do_capture:
                idx = len(existing_rows)
                fname = f"ood_real_{idx:04d}_{cat_name}.png"
                cv2.imwrite(str(img_dir / fname), x_u8)
                existing_rows.append(
                    {
                        "sample_id": str(idx),
                        "category": cat_name,
                        "is_valid_roi": str(is_valid),
                        "image_path": f"images/{fname}",
                    }
                )
                with meta_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["sample_id", "category", "is_valid_roi", "image_path"]
                    )
                    writer.writeheader()
                    writer.writerows(existing_rows)
                last_snap = time.time()
                print(f"[CAP] #{idx + 1} ({cat_name}) -> {fname}")

            if key == 27:
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                cat_idx = key - ord("1")
            elif key in (ord("a"), ord("A")):
                auto_mode = not auto_mode
            elif key in (ord("f"), ord("F")):
                flip = not flip
            elif key in (ord("e"), ord("E")):
                evaluate_captured_ood(
                    args.output_dir,
                    Path("checkpoints/generator.onnx"),
                    Path("checkpoints/judge_resnet18.onnx"),
                    threshold=args.threshold,
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
