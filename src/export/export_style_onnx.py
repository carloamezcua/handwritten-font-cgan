"""Exporta StyleUNetGenerator (`Style Encoder + AdaIN`) a ONNX con estilo personal precalculado.

Precalcula `z_style` una sola vez a partir de K=4 trazos reales propios (`data/real_pairs/train/y`)
para que `checkpoints/style_unet.onnx` tenga exactamente la misma interfaz ONNX que `generator.onnx`
(`x -> y_hat`, `1x128x128`) y corra en tiempo real en CPU sin recomputar el StyleEncoder en cada cuadro.

Uso:
    python -m src.export.export_style_onnx
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image

from src.config import IMG_SIZE, SEED, TEST_UNSEEN_CHARS
from src.models.style_unet import StyleUNetGenerator
from src.utils import seed_everything

PREFERRED_STYLE_CHARS = ("A", "R", "e", "s", "m", "B", "H", "d")


class FixedStyleUNetONNX(nn.Module):
    """Wrapper ONNX que fija `z_style` como buffer constante para inferencia rápida `x -> y_hat`."""

    def __init__(self, gen: StyleUNetGenerator, z_style: torch.Tensor) -> None:
        super().__init__()
        self.gen = gen
        # [1, style_dim]
        self.register_buffer("z_style", z_style.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.z_style.expand(x.size(0), -1)
        return self.gen(x, z_style=z)


def load_real_style_refs(
    data_dir: Path,
    num_refs: int = 4,
) -> tuple[torch.Tensor, list[str]]:
    """Carga K glifos reales de `train/y` (jamás de `TEST_UNSEEN_CHARS`) en `[-1, 1]`."""
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"No se encontró '{meta_path}'.")

    train_rows: list[dict[str, str]] = []
    with meta_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train" and row.get("char") not in TEST_UNSEEN_CHARS:
                train_rows.append(row)

    if len(train_rows) < num_refs:
        raise RuntimeError(
            f"Se requieren al menos {num_refs} muestras de train en '{data_dir}', "
            f"solo se encontraron {len(train_rows)}."
        )

    selected: list[dict[str, str]] = []
    used_chars: set[str] = set()

    # Elegir primero caracteres representativos (curvas + rectas)
    for target_ch in PREFERRED_STYLE_CHARS:
        if len(selected) >= num_refs:
            break
        for row in train_rows:
            if row["char"] == target_ch and target_ch not in used_chars:
                selected.append(row)
                used_chars.add(target_ch)
                break

    # Completar si faltó alguno
    for row in train_rows:
        if len(selected) >= num_refs:
            break
        if row["char"] not in used_chars:
            selected.append(row)
            used_chars.add(row["char"])

    tensors: list[torch.Tensor] = []
    chars: list[str] = []
    for row in selected:
        y_path = data_dir / row["y_path"]
        img = Image.open(y_path).convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).unsqueeze(0))  # [1, H, W]
        chars.append(row["char"])

    # [1, K, 1, H, W]
    refs = torch.stack(tensors, dim=0).unsqueeze(0)
    return refs, chars


def export_style_unet_onnx(
    ckpt_path: Path,
    data_dir: Path,
    output_onnx: Path,
    opset: int = 17,
    intra_threads: int = 4,
) -> dict[str, float | str | list[str]]:
    seed_everything(SEED)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: '{ckpt_path}'.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    style_dim = int(ckpt.get("style_dim", 128))
    num_refs = int(ckpt.get("num_style_refs", 4))

    gen = StyleUNetGenerator(style_dim=style_dim)
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()

    style_refs, ref_chars = load_real_style_refs(data_dir, num_refs=num_refs)
    with torch.no_grad():
        z_style = gen.encode_style(style_refs)  # [1, style_dim]

    wrapper = FixedStyleUNetONNX(gen, z_style).eval()
    dummy_x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y_hat"],
        dynamic_axes={"x": {0: "batch"}, "y_hat": {0: "batch"}},
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(wrapper, (dummy_x,), str(output_onnx), **kwargs)
    onnx_model = onnx.load(str(output_onnx))
    onnx.checker.check_model(onnx_model)

    # Verificar equivalencia numérica y latencia en CPU
    with torch.no_grad():
        pt_out = wrapper(dummy_x).numpy()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = min(intra_threads, os.cpu_count() or intra_threads)
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(output_onnx), sess_options=opts, providers=["CPUExecutionProvider"])

    dummy_np = dummy_x.numpy()
    ort_out = sess.run(["y_hat"], {"x": dummy_np})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))

    for _ in range(5):
        sess.run(["y_hat"], {"x": dummy_np})
    times_ms = []
    for _ in range(40):
        t0 = time.perf_counter()
        sess.run(["y_hat"], {"x": dummy_np})
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(times_ms))
    fps = 1000.0 / max(mean_ms, 1e-6)
    size_mb = output_onnx.stat().st_size / (1024 * 1024)

    print(f"[OK] Exportado: {output_onnx} ({size_mb:.2f} MB)")
    print(f"     Checkpoint: {ckpt_path} | Refs estilo ({num_refs}): {ref_chars}")
    print(f"     Error max abs PyTorch vs ONNX: {max_diff:.2e}")
    print(f"     Latencia CPU ONNX: {mean_ms:.2f} ms ({fps:.1f} FPS)")
    return {
        "onnx_path": str(output_onnx),
        "ref_chars": ref_chars,
        "max_abs_diff": max_diff,
        "mean_ms": round(mean_ms, 2),
        "fps": round(fps, 2),
        "size_mb": round(size_mb, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta StyleUNetGenerator a ONNX con estilo fijo.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/style_unet_finetuned.pth"),
        help="Checkpoint entrenado de StyleUNet.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_pairs"),
        help="Directorio con trazos reales para calcular z_style (usa solo split=train).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/style_unet.onnx"),
        help="Ruta de salida del modelo ONNX.",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_style_unet_onnx(
        ckpt_path=args.ckpt,
        data_dir=args.data_dir,
        output_onnx=args.output,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
