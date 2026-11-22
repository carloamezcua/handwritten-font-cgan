"""Descarga Google Fonts manuscritas abiertas y genera el dataset sintético (x, y) 1x128x128.

Cada par contiene:
- `x`: glifo renderizado con la fuente base limpia (sans-serif).
- `y`: mismo glifo renderizado con una fuente manuscrita abierta y ligeras
  variaciones controladas (escala ±5 %, traslación ±3 px).

Particiones sin fuga de información (registradas en `metadata.csv`):
- `split` (para Pix2Pix):
    * `train`: 64 símbolos vistos en fuentes de entrenamiento.
    * `val`: 64 símbolos vistos en fuentes de validación (estilos no vistos).
    * `test_unseen`: 11 símbolos reservados (`TEST_UNSEEN_CHARS`, Zero-Shot).
- `judge_split` (para clasificadores jueces ResNet-18 / EfficientNet-B0):
    * `train` / `val`: los 75 símbolos separados estrictamente por fuente
      (los jueces necesitan conocer las 75 clases para evaluar Zero-Shot).

Uso:
    python -m src.data.03_generar_sinteticos --download-fonts
"""

from __future__ import annotations

import argparse
import csv
import random
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from src.config import CHAR_TO_IDX, CHARSET, IMG_SIZE, SEED, TEST_UNSEEN_CHARS
from src.data.glyph_render import (
    font_supports_charset,
    get_default_base_font_path,
    render_glyph,
)
from src.utils import seed_everything

GOOGLE_FONTS_RAW = "https://raw.githubusercontent.com/google/fonts/main"

# Fuente base limpia (sans-serif con licencia abierta OFL)
BASE_FONT_SPEC = {
    "name": "Lato",
    "filename": "Lato-Regular.ttf",
    "license": "SIL Open Font License 1.1 (OFL)",
    "url": f"{GOOGLE_FONTS_RAW}/ofl/lato/Lato-Regular.ttf",
}

# Catálogo de 20 Google Fonts manuscritas con licencia abierta (OFL)
HANDWRITTEN_FONT_SPECS = [
    ("Patrick Hand", "PatrickHand-Regular.ttf", "ofl/patrickhand/PatrickHand-Regular.ttf"),
    ("Indie Flower", "IndieFlower-Regular.ttf", "ofl/indieflower/IndieFlower-Regular.ttf"),
    ("Shadows Into Light", "ShadowsIntoLight.ttf", "ofl/shadowsintolight/ShadowsIntoLight.ttf"),
    ("Kalam", "Kalam-Regular.ttf", "ofl/kalam/Kalam-Regular.ttf"),
    ("Architects Daughter", "ArchitectsDaughter-Regular.ttf", "ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf"),
    ("Handlee", "Handlee-Regular.ttf", "ofl/handlee/Handlee-Regular.ttf"),
    ("Reenie Beanie", "ReenieBeanie.ttf", "ofl/reeniebeanie/ReenieBeanie.ttf"),
    ("Gloria Hallelujah", "GloriaHallelujah.ttf", "ofl/gloriahallelujah/GloriaHallelujah.ttf"),
    ("Coming Soon", "ComingSoon-Regular.ttf", "ofl/comingsoon/ComingSoon-Regular.ttf"),
    ("Just Another Hand", "JustAnotherHand-Regular.ttf", "ofl/justanotherhand/JustAnotherHand-Regular.ttf"),
    ("Neucha", "Neucha.ttf", "ofl/neucha/Neucha.ttf"),
    ("Gochi Hand", "GochiHand-Regular.ttf", "ofl/gochihand/GochiHand-Regular.ttf"),
    ("Courgette", "Courgette-Regular.ttf", "ofl/courgette/Courgette-Regular.ttf"),
    ("Pangolin", "Pangolin-Regular.ttf", "ofl/pangolin/Pangolin-Regular.ttf"),
    ("Sriracha", "Sriracha-Regular.ttf", "ofl/sriracha/Sriracha-Regular.ttf"),
    ("Short Stack", "ShortStack-Regular.ttf", "ofl/shortstack/ShortStack-Regular.ttf"),
    ("Dekko", "Dekko-Regular.ttf", "ofl/dekko/Dekko-Regular.ttf"),
    ("Mali", "Mali-Regular.ttf", "ofl/mali/Mali-Regular.ttf"),
    ("Delius", "Delius-Regular.ttf", "ofl/delius/Delius-Regular.ttf"),
    ("Schoolbell", "Schoolbell-Regular.ttf", "ofl/schoolbell/Schoolbell-Regular.ttf"),
]


def download_file(url: str, dest: Path, timeout: int = 20) -> bool:
    """Descarga un archivo si no existe localmente."""
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[AVISO] No se pudo descargar {dest.name} desde {url}: {exc}")
        return False


def download_fonts(fonts_dir: Path) -> tuple[Path | None, list[dict]]:
    """Descarga la fuente base y las fuentes manuscritas abiertas, registrando licencias."""
    base_dir = fonts_dir / "base"
    hw_dir = fonts_dir / "handwritten"
    base_dir.mkdir(parents=True, exist_ok=True)
    hw_dir.mkdir(parents=True, exist_ok=True)

    license_rows: list[dict] = []

    base_path = base_dir / BASE_FONT_SPEC["filename"]
    if download_file(BASE_FONT_SPEC["url"], base_path):
        ok, missing = font_supports_charset(base_path, CHARSET)
        license_rows.append(
            {
                "font_name": BASE_FONT_SPEC["name"],
                "filename": BASE_FONT_SPEC["filename"],
                "role": "base",
                "license": BASE_FONT_SPEC["license"],
                "source_url": BASE_FONT_SPEC["url"],
                "supports_full_charset": ok,
                "missing_chars": "".join(missing),
            }
        )
    else:
        base_path = None

    for name, filename, rel_path in HANDWRITTEN_FONT_SPECS:
        url = f"{GOOGLE_FONTS_RAW}/{rel_path}"
        dest = hw_dir / filename
        if download_file(url, dest):
            ok, missing = font_supports_charset(dest, CHARSET)
            license_rows.append(
                {
                    "font_name": name,
                    "filename": filename,
                    "role": "handwritten",
                    "license": "SIL Open Font License 1.1 (OFL)",
                    "source_url": url,
                    "supports_full_charset": ok,
                    "missing_chars": "".join(missing),
                }
            )
            if not ok:
                print(f"[AVISO] {name} no soporta todos los símbolos (faltan: {''.join(missing)}).")

    licenses_csv = fonts_dir / "licenses.csv"
    with licenses_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "font_name",
                "filename",
                "role",
                "license",
                "source_url",
                "supports_full_charset",
                "missing_chars",
            ],
        )
        writer.writeheader()
        writer.writerows(license_rows)

    print(f"Registro de licencias guardado en: {licenses_csv} ({len(license_rows)} fuentes)")
    return base_path, license_rows


def collect_handwritten_fonts(fonts_dir: Path) -> list[Path]:
    """Recopila todas las fuentes .ttf/.otf válidas en `data/fonts/handwritten/`."""
    hw_dir = fonts_dir / "handwritten"
    if not hw_dir.exists():
        return []
    candidates = sorted(
        list(hw_dir.glob("*.ttf")) + list(hw_dir.glob("*.otf"))
    )
    valid: list[Path] = []
    for path in candidates:
        ok, missing = font_supports_charset(path, CHARSET)
        if ok:
            valid.append(path)
        else:
            print(f"[OMITIDA] {path.name}: faltan {len(missing)} símbolos ({''.join(missing)})")
    return valid


def generate_synthetic_dataset(
    fonts_dir: Path,
    output_dir: Path,
    base_font: Path | None = None,
    variants_per_font: int = 3,
    val_font_ratio: float = 0.2,
    seed: int = SEED,
) -> Path:
    """Genera pares (x, y) 128x128 y escribe `metadata.csv`."""
    seed_everything(seed)
    rng = random.Random(seed)

    if base_font is None:
        base_font = get_default_base_font_path(fonts_dir)
    print(f"Fuente base limpia (x): {base_font}")

    hw_fonts = collect_handwritten_fonts(fonts_dir)
    if not hw_fonts:
        raise RuntimeError(
            f"No se encontraron fuentes manuscritas completas en '{fonts_dir / 'handwritten'}'. "
            "Ejecuta con --download-fonts."
        )

    # Dividir fuentes en train/val para evitar fuga de estilo en validación
    shuffled_fonts = list(hw_fonts)
    rng.shuffle(shuffled_fonts)
    n_val_fonts = max(1, int(round(len(shuffled_fonts) * val_font_ratio)))
    val_fonts = set(shuffled_fonts[:n_val_fonts])
    train_fonts = set(shuffled_fonts[n_val_fonts:])

    print(
        f"Fuentes manuscritas válidas: {len(hw_fonts)} "
        f"(train_fonts={len(train_fonts)}, val_fonts={len(val_fonts)})"
    )

    for split in ("train", "val", "test_unseen"):
        (output_dir / split / "x").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "y").mkdir(parents=True, exist_ok=True)

    # Pre-renderizar la fuente base limpia para los 75 símbolos (sin variaciones)
    base_glyphs = {ch: render_glyph(ch, base_font, img_size=IMG_SIZE) for ch in CHARSET}

    rows: list[dict] = []
    sample_id = 0

    for font_path in sorted(hw_fonts, key=lambda p: p.name):
        font_name = font_path.stem
        font_split = "val" if font_path in val_fonts else "train"

        for ch in CHARSET:
            class_id = CHAR_TO_IDX[ch]
            is_unseen = ch in TEST_UNSEEN_CHARS
            pix2pix_split = "test_unseen" if is_unseen else font_split
            judge_split = font_split

            for variant in range(variants_per_font):
                if variant == 0:
                    scale, shift_x, shift_y = 1.0, 0, 0
                else:
                    scale = rng.uniform(0.95, 1.05)
                    shift_x = rng.randint(-3, 3)
                    shift_y = rng.randint(-3, 3)

                x_img = base_glyphs[ch]
                y_img = render_glyph(
                    ch,
                    font_path,
                    img_size=IMG_SIZE,
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )

                fname = f"{sample_id:05d}_c{class_id:02d}_{font_name}_v{variant}.png"
                x_rel = Path(pix2pix_split) / "x" / fname
                y_rel = Path(pix2pix_split) / "y" / fname

                Image.fromarray(x_img, mode="L").save(output_dir / x_rel)
                Image.fromarray(y_img, mode="L").save(output_dir / y_rel)

                rows.append(
                    {
                        "sample_id": sample_id,
                        "char": ch,
                        "char_code": ord(ch),
                        "class_id": class_id,
                        "font_name": font_name,
                        "variant": variant,
                        "split": pix2pix_split,
                        "judge_split": judge_split,
                        "x_path": x_rel.as_posix(),
                        "y_path": y_rel.as_posix(),
                    }
                )
                sample_id += 1

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "char",
                "char_code",
                "class_id",
                "font_name",
                "variant",
                "split",
                "judge_split",
                "x_path",
                "y_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test_unseen")}
    print(
        f"Dataset sintético generado en '{output_dir}': {len(rows)} pares totales "
        f"(train={counts['train']}, val={counts['val']}, test_unseen={counts['test_unseen']})"
    )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga Google Fonts manuscritas y genera pares sintéticos (x, y) 128x128."
    )
    parser.add_argument("--fonts-dir", type=Path, default=Path("data/fonts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic_pairs"))
    parser.add_argument("--base-font", type=Path, default=None)
    parser.add_argument("--variants-per-font", type=int, default=3)
    parser.add_argument("--val-font-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--download-fonts",
        action="store_true",
        help="Descarga la fuente base y el catálogo de Google Fonts abiertas antes de generar.",
    )
    args = parser.parse_args()

    base_font = args.base_font
    if args.download_fonts:
        downloaded_base, _ = download_fonts(args.fonts_dir)
        if base_font is None and downloaded_base is not None:
            base_font = downloaded_base

    generate_synthetic_dataset(
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        base_font=base_font,
        variants_per_font=args.variants_per_font,
        val_font_ratio=args.val_font_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
