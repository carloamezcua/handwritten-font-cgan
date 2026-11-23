"""Exportación a ONNX (y TorchScript opcional) + benchmark de latencia/FPS en CPU.

Exporta:
1. Generador `UNetGenerator` (`checkpoints/generator.onnx`).
2. Juez principal (`ResNet-18` por defecto -> `checkpoints/judge_resnet18.onnx`).

Mide latencia promedio (ms), percentil 95 (ms) y FPS sobre 100 inferencias en CPU
antes (PyTorch) vs. después (ONNX Runtime), verifica equivalencia numérica y
genera la tabla comparativa en `docs/robustez_y_despliegue.md` (Rúbrica C5).

Uso:
    python -m src.export.export_onnx
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import CHARSET, IMG_SIZE, SEED
from src.models.judges import build_judge
from src.models.pix2pix import UNetGenerator
from src.utils import seed_everything


def _export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: Path,
    input_name: str = "input",
    output_name: str = "output",
    opset: int = 17,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False

    torch.onnx.export(model, (dummy_input,), str(onnx_path), **kwargs)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def _create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _benchmark_callable(fn, warmup: int = 10, runs: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(float(arr.std()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "fps": round(1000.0 / max(mean_ms, 1e-6), 2),
    }


def write_deployment_doc(
    doc_path: Path,
    results: dict,
    generator_ckpt: Path,
    judge_ckpt: Path,
    gen_onnx: Path,
    judge_onnx: Path,
) -> None:
    """Escribe/actualiza `docs/robustez_y_despliegue.md` con la tabla C5 de latencia y FPS."""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    hw = results["hardware"]
    pt = results["pytorch_cpu"]
    ort_res = results["onnxruntime_cpu"]
    sp = results["speedup"]

    meets_target = ort_res["pipeline"]["fps"] >= 10.0
    status_str = "CUMPLE (>= 10 FPS / <= 100 ms)" if meets_target else "POR DEBAJO DE 10 FPS"

    content = f"""# Robustez y Despliegue (ONNX, Latencia y Hardware)

Documento de evidencia para los indicadores **C1–C5** de la rúbrica.

## 1. Entorno de Hardware y Software (Benchmark CPU)
- **Sistema Operativo:** {hw["os"]}
- **Procesador:** {hw["processor"]}
- **Núcleos lógicos CPU:** {hw["cpu_count"]}
- **Python / PyTorch / ONNX Runtime:** `{hw["python"]}` / `{hw["torch"]}` / `{hw["onnxruntime"]}`
- **Protocolo de medición:** `{results["runs"]}` inferencias consecutivas en CPU (`batch_size = 1`, entrada `1×128×128`), tras `{results["warmup"]}` ejecuciones de calentamiento.

## 2. Modelos Exportados y Verificación Numérica
| Componente | Checkpoint PyTorch | Modelo ONNX | Tamaño ONNX (MB) | Error máx. abs. (PyTorch vs. ONNX) |
| :--- | :--- | :--- | :---: | :---: |
| Generador `G` (`UNetGenerator`) | `{generator_ckpt.name}` | `{gen_onnx.name}` | {results["sizes_mb"]["generator_onnx"]:.2f} | `{results["max_abs_diff"]["generator"]:.2e}` |
| Juez (`{results["judge_model"]}`) | `{judge_ckpt.name}` | `{judge_onnx.name}` | {results["sizes_mb"]["judge_onnx"]:.2f} | `{results["max_abs_diff"]["judge"]:.2e}` |

## 3. Comparativa de Latencia y FPS: Antes (PyTorch) vs. Después (ONNX Runtime)
| Etapa | PyTorch CPU (ms) | PyTorch FPS | ONNX Runtime CPU (ms) | ONNX p95 (ms) | ONNX FPS | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Generador `G` (`UNetGenerator`) | {pt["generator"]["mean_ms"]:.2f} ± {pt["generator"]["std_ms"]:.2f} | {pt["generator"]["fps"]:.2f} | {ort_res["generator"]["mean_ms"]:.2f} ± {ort_res["generator"]["std_ms"]:.2f} | {ort_res["generator"]["p95_ms"]:.2f} | {ort_res["generator"]["fps"]:.2f} | {sp["generator"]:.2f}× |
| Juez (`{results["judge_model"]}`) | {pt["judge"]["mean_ms"]:.2f} ± {pt["judge"]["std_ms"]:.2f} | {pt["judge"]["fps"]:.2f} | {ort_res["judge"]["mean_ms"]:.2f} ± {ort_res["judge"]["std_ms"]:.2f} | {ort_res["judge"]["p95_ms"]:.2f} | {ort_res["judge"]["fps"]:.2f} | {sp["judge"]:.2f}× |
| **Pipeline completo (`G + Juez`)** | **{pt["pipeline"]["mean_ms"]:.2f} ± {pt["pipeline"]["std_ms"]:.2f}** | **{pt["pipeline"]["fps"]:.2f}** | **{ort_res["pipeline"]["mean_ms"]:.2f} ± {ort_res["pipeline"]["std_ms"]:.2f}** | **{ort_res["pipeline"]["p95_ms"]:.2f}** | **{ort_res["pipeline"]["fps"]:.2f}** | **{sp["pipeline"]:.2f}×** |

> **Veredicto de tiempo real (Rúbrica C2 / C5):** **{status_str}** (`{ort_res["pipeline"]["mean_ms"]:.2f} ms` promedio / `{ort_res["pipeline"]["fps"]:.2f} FPS` en CPU).

## 4. Demo en Tiempo Real y Calibración OOD
- **Arranque en 1 comando:** `python app/demo_realtime.py`
- **Modos (`M`):** Canvas interactivo (teclado / lista de caracteres) y Webcam ROI (flujo continuo cuadro por cuadro).
- **Umbral de confianza (`+` / `-`, default `0.70`):** si la probabilidad máxima del juez sobre el glifo generado cae por debajo del umbral (o si la entrada está vacía / con ruido extremo), se muestra la alerta visual `BAJA CONFIANZA (OOD)`.
- *(En la Fase 6 se añadirá aquí el resumen de tasas de rechazo OOD obtenido desde `eval/robustez.ipynb`.)*
"""
    doc_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta Generador y Juez a ONNX y corre benchmark en CPU."
    )
    parser.add_argument(
        "--generator-ckpt",
        type=Path,
        default=Path("checkpoints/pix2pix_pretrained.pth"),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
    )
    parser.add_argument(
        "--judge-ckpt",
        type=Path,
        default=Path("checkpoints/judge_resnet18.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=Path("docs/robustez_y_despliegue.md"),
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("results/benchmark_onnx.json"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="También exporta TorchScript (.ts) además de ONNX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    device = torch.device("cpu")

    if not args.generator_ckpt.exists():
        raise FileNotFoundError(f"No se encontró el generador: '{args.generator_ckpt}'.")
    if not args.judge_ckpt.exists():
        raise FileNotFoundError(f"No se encontró el juez: '{args.judge_ckpt}'.")

    # 1. Cargar modelos en CPU eval()
    gen = UNetGenerator().to(device)
    gen_ckpt = torch.load(args.generator_ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(gen_ckpt["generator_state_dict"])
    gen.eval()

    judge = build_judge(args.judge_model, num_classes=len(CHARSET), pretrained=False).to(device)
    j_ckpt = torch.load(args.judge_ckpt, map_location=device, weights_only=False)
    judge.load_state_dict(j_ckpt["state_dict"])
    judge.eval()

    dummy_x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)

    # 2. Exportar a ONNX
    gen_onnx = args.output_dir / "generator.onnx"
    judge_onnx = args.output_dir / f"judge_{args.judge_model}.onnx"

    print(f"Exportando Generador -> {gen_onnx}")
    _export_to_onnx(gen, dummy_x, gen_onnx, input_name="x", output_name="y_hat")

    print(f"Exportando Juez ({args.judge_model}) -> {judge_onnx}")
    _export_to_onnx(judge, dummy_x, judge_onnx, input_name="glyph", output_name="logits")

    if args.export_torchscript:
        gen_ts = args.output_dir / "generator.ts"
        torch.jit.trace(gen, dummy_x).save(str(gen_ts))
        print(f"TorchScript guardado -> {gen_ts}")

    # 3. Verificación numérica PyTorch vs. ONNX Runtime
    gen_sess = _create_ort_session(gen_onnx)
    judge_sess = _create_ort_session(judge_onnx)
    dummy_np = dummy_x.numpy().astype(np.float32)

    with torch.no_grad():
        pt_yhat = gen(dummy_x)
        pt_logits = judge(pt_yhat)

    ort_yhat = gen_sess.run(["y_hat"], {"x": dummy_np})[0]
    ort_logits = judge_sess.run(["logits"], {"glyph": ort_yhat})[0]

    diff_gen = float(np.max(np.abs(pt_yhat.numpy() - ort_yhat)))
    diff_judge = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
    print(f"[OK] Verificación numérica | max_abs_diff G={diff_gen:.2e} | Judge={diff_judge:.2e}")

    # 4. Benchmark CPU (100 inferencias)
    print(f"\nEjecutando benchmark CPU ({args.runs} inferencias, warmup={args.warmup})...")

    def pt_gen_fn() -> None:
        with torch.no_grad():
            _ = gen(dummy_x)

    def pt_judge_fn() -> None:
        with torch.no_grad():
            _ = judge(dummy_x)

    def pt_pipe_fn() -> None:
        with torch.no_grad():
            yh = gen(dummy_x)
            _ = judge(yh)

    def ort_gen_fn() -> None:
        _ = gen_sess.run(["y_hat"], {"x": dummy_np})

    def ort_judge_fn() -> None:
        _ = judge_sess.run(["logits"], {"glyph": dummy_np})

    def ort_pipe_fn() -> None:
        yh = gen_sess.run(["y_hat"], {"x": dummy_np})[0]
        _ = judge_sess.run(["logits"], {"glyph": yh})

    pt_gen = _benchmark_callable(pt_gen_fn, args.warmup, args.runs)
    pt_judge = _benchmark_callable(pt_judge_fn, args.warmup, args.runs)
    pt_pipe = _benchmark_callable(pt_pipe_fn, args.warmup, args.runs)

    ort_gen = _benchmark_callable(ort_gen_fn, args.warmup, args.runs)
    ort_judge = _benchmark_callable(ort_judge_fn, args.warmup, args.runs)
    ort_pipe = _benchmark_callable(ort_pipe_fn, args.warmup, args.runs)

    results = {
        "judge_model": args.judge_model,
        "warmup": args.warmup,
        "runs": args.runs,
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count() or 1,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "sizes_mb": {
            "generator_onnx": round(gen_onnx.stat().st_size / (1024 ** 2), 2),
            "judge_onnx": round(judge_onnx.stat().st_size / (1024 ** 2), 2),
        },
        "max_abs_diff": {
            "generator": diff_gen,
            "judge": diff_judge,
        },
        "pytorch_cpu": {
            "generator": pt_gen,
            "judge": pt_judge,
            "pipeline": pt_pipe,
        },
        "onnxruntime_cpu": {
            "generator": ort_gen,
            "judge": ort_judge,
            "pipeline": ort_pipe,
        },
        "speedup": {
            "generator": round(pt_gen["mean_ms"] / max(ort_gen["mean_ms"], 1e-6), 2),
            "judge": round(pt_judge["mean_ms"] / max(ort_judge["mean_ms"], 1e-6), 2),
            "pipeline": round(pt_pipe["mean_ms"] / max(ort_pipe["mean_ms"], 1e-6), 2),
        },
    }

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_deployment_doc(
        doc_path=args.doc_path,
        results=results,
        generator_ckpt=args.generator_ckpt,
        judge_ckpt=args.judge_ckpt,
        gen_onnx=gen_onnx,
        judge_onnx=judge_onnx,
    )

    print("\n=== Resumen Benchmark CPU (100 inferencias) ===")
    print(
        f"  PyTorch Pipeline (G+Judge): {pt_pipe['mean_ms']:.2f} ms ({pt_pipe['fps']:.2f} FPS)"
    )
    print(
        f"  ONNX RT Pipeline (G+Judge): {ort_pipe['mean_ms']:.2f} ms ({ort_pipe['fps']:.2f} FPS) "
        f"| Speedup: {results['speedup']['pipeline']:.2f}x"
    )
    print(f"[OK] Reporte guardado en: {args.doc_path}")
    print(f"[OK] JSON guardado en:    {args.json_path}")


if __name__ == "__main__":
    main()
