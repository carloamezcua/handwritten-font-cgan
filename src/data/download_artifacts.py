"""Descarga pesos entrenados (.onnx / .pth) y dataset personal desde GitHub Releases.

Permite reproducir la demo en tiempo real (`python app/demo_realtime.py`) y la evaluación
completa desde un clon limpio del repositorio sin almacenar binarios pesados en Git.

Uso:
    # Descargar solo modelos ONNX para correr la demo en tiempo real:
    python -m src.data.download_artifacts --only-demo

    # Descargar todos los pesos (.onnx + .pth) y capturas reales (real_pairs.zip):
    python -m src.data.download_artifacts --all
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_REPO = "carloamezcua/handwritten-font-cgan"
DEFAULT_TAG = "v1.0-final"

DEMO_ASSETS = [
    ("generator.onnx", Path("checkpoints/generator.onnx")),
    ("judge_resnet18.onnx", Path("checkpoints/judge_resnet18.onnx")),
]

CHECKPOINT_ASSETS = [
    ("pix2pix_pretrained.pth", Path("checkpoints/pix2pix_pretrained.pth")),
    ("pix2pix_finetuned.pth", Path("checkpoints/pix2pix_finetuned.pth")),
    ("pix2pix_scratch.pth", Path("checkpoints/pix2pix_scratch.pth")),
    ("judge_resnet18.pth", Path("checkpoints/judge_resnet18.pth")),
    ("judge_efficientnet_b0.pth", Path("checkpoints/judge_efficientnet_b0.pth")),
]

DATA_ASSETS = [
    ("real_pairs.zip", Path("data/real_pairs.zip")),
]


def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[SKIP] Ya existe: {dest} ({dest.stat().st_size / (1024 ** 2):.2f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    print(f"[GET]  {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "handwritten-font-cgan-downloader"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp_path.replace(dest)
        print(f"[OK]   Guardado: {dest} ({dest.stat().st_size / (1024 ** 2):.2f} MB)")
        return True
    except urllib.error.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[WARN] HTTP {exc.code} al descargar {url} (revisa que el asset exista en el Release).")
        return False
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[WARN] Error descargando {url}: {exc}")
        return False


def _extract_real_pairs(zip_path: Path, target_dir: Path) -> None:
    if not zip_path.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ZIP]  Extrayendo {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            rel = info.filename.replace("\\", "/").lstrip("/")
            if not rel or rel.endswith("/"):
                continue
            # Evitar duplicar prefijo real_pairs/ si el zip ya lo contiene
            if rel.startswith("real_pairs/"):
                rel = rel[len("real_pairs/") :]
            out_file = target_dir / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, out_file.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"[OK]   Dataset real extraído en: {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga pesos (.onnx/.pth) y datos reales desde GitHub Releases."
    )
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="URL base alternativa (sobrescribe --repo y --tag).",
    )
    parser.add_argument(
        "--only-demo",
        action="store_true",
        help="Descarga únicamente los modelos ONNX necesarios para `python app/demo_realtime.py`.",
    )
    parser.add_argument(
        "--include-real-data",
        action="store_true",
        help="Incluye la descarga y extracción de `real_pairs.zip` en `data/real_pairs/`.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Descarga ONNX, checkpoints PyTorch y `real_pairs.zip`.",
    )
    parser.add_argument("--force", action="store_true", help="Sobrescribe archivos existentes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = (
        args.base_url.rstrip("/")
        if args.base_url
        else f"https://github.com/{args.repo}/releases/download/{args.tag}"
    )

    assets: list[tuple[str, Path]] = list(DEMO_ASSETS)
    if args.all:
        assets.extend(CHECKPOINT_ASSETS)
        assets.extend(DATA_ASSETS)
    elif not args.only_demo:
        assets.extend(CHECKPOINT_ASSETS)
        if args.include_real_data:
            assets.extend(DATA_ASSETS)
    elif args.include_real_data:
        assets.extend(DATA_ASSETS)

    ok_count = 0
    for asset_name, dest in assets:
        if _download_file(f"{base_url}/{asset_name}", dest, force=args.force):
            ok_count += 1

    zip_path = Path("data/real_pairs.zip")
    if (args.all or args.include_real_data) and zip_path.exists():
        _extract_real_pairs(zip_path, Path("data/real_pairs"))

    print(f"\nResumen: {ok_count}/{len(assets)} artefactos disponibles.")
    if ok_count < len(assets):
        print(
            "Nota: si el Release todavía no tiene publicados todos los archivos, "
            "puedes generar los pesos localmente siguiendo la sección de reproducción del README."
        )


if __name__ == "__main__":
    main()
