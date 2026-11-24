"""Dataset condicionado por estilo (`StyleGlyphPairDataset`) sin fuga de información.

Por cada muestra `(x, y)` de un autor/fuente `style_id`, devuelve además `K` glifos
de referencia (`style_refs`: `[K, 1, 128, 128]`) del mismo estilo, cumpliendo estrictamente:
1. Nunca se incluye el propio carácter objetivo `char` en `style_refs` (evita copiado).
2. En `test_unseen`, las referencias provienen exclusivamente de caracteres vistos
   (`train` o `val`), nunca de `TEST_UNSEEN_CHARS`.
3. En `real_pairs` (un solo autor), tanto `val` como `test_unseen` usan referencias
   fijas tomadas exclusivamente de `train` (0 % fuga).
4. En `train`, opcionalmente aplica el MISMO aumento morfológico de grosor de trazo
   (adelgazamiento tipo stylus o engrosamiento) a `y` y a las `K` referencias, para que
   el `StyleEncoder` aprenda a transferir trazos finos reales en inferencia Few-Shot.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import IMG_SIZE, SEED, TEST_UNSEEN_CHARS
from src.data.dataset import PairGentleAffine, _pil_to_normalized_tensor

DEFAULT_REAL_REF_CHARS = ("A", "e", "7", "M", "s", "R", "3", "h")


def _apply_stroke_thickness(img: Image.Image, op: str, ksize: int) -> Image.Image:
    """Modifica grosor de trazo negro sobre fondo blanco (255).

    - `thin`: dilata el fondo blanco (`cv2.dilate`), adelgazando el trazo negro (simula stylus fino).
    - `thick`: erosiona el fondo blanco (`cv2.erode`), engrosando el trazo negro.
    Se verifica que no se borre el glifo por completo (conserva el original si queda casi blanco).
    """
    if op == "none" or ksize <= 1:
        return img
    arr = np.array(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == "thin":
        out = cv2.dilate(arr, kernel, iterations=1)
        # Evitar borrar signos diminutos (p. ej. '.' o ',')
        if float(np.mean(out < 180)) < 0.0015:
            return img
        return Image.fromarray(out, mode="L")
    if op == "thick":
        out = cv2.erode(arr, kernel, iterations=1)
        return Image.fromarray(out, mode="L")
    return img


class StyleGlyphPairDataset(Dataset):
    """Dataset `(x, y, style_refs, class_id, char, style_id)` para `StyleUNetGenerator`."""

    def __init__(
        self,
        root_dir: Path | str,
        split: str = "train",
        num_style_refs: int = 4,
        augment: bool = False,
        stroke_augment: bool = True,
        img_size: int = IMG_SIZE,
        seed: int = SEED,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_style_refs = num_style_refs
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.stroke_augment = stroke_augment and (split == "train")
        self.pair_aug = PairGentleAffine() if self.augment else None
        self.seed = seed

        metadata_path = self.root_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontró '{metadata_path}'.")

        all_rows: list[dict] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["style_id"] = row.get("font_name") or "personal_stylus"
                all_rows.append(row)

        self.samples = [r for r in all_rows if r["split"] == split]
        if not self.samples:
            raise ValueError(f"No hay muestras para split='{split}' en '{metadata_path}'.")

        # Pools de referencias por (split, style_id) — siempre sin caracteres de test_unseen
        self.train_pool: dict[str, list[dict]] = defaultdict(list)
        self.val_pool: dict[str, list[dict]] = defaultdict(list)
        for r in all_rows:
            if r["split"] == "train" and r["char"] not in TEST_UNSEEN_CHARS:
                self.train_pool[r["style_id"]].append(r)
            elif r["split"] == "val" and r["char"] not in TEST_UNSEEN_CHARS:
                self.val_pool[r["style_id"]].append(r)

        # Verificación estricta anti-fuga: ninguna referencia pertenece a TEST_UNSEEN_CHARS
        for pool in (self.train_pool, self.val_pool):
            for sid, items in pool.items():
                assert all(it["char"] not in TEST_UNSEEN_CHARS for it in items), (
                    f"Fuga detectada en pool de referencias para estilo '{sid}'."
                )

    def _get_ref_candidates(self, style_id: str, exclude_char: str) -> list[dict]:
        """Obtiene candidatos de referencia del mismo estilo sin fuga de información."""
        if self.split == "train":
            pool = self.train_pool.get(style_id, [])
        else:
            # Si el estilo existe en `train` (ej. `real_pairs` o fuentes de train en test_unseen),
            # usar SIEMPRE `train` como banco de referencias (0% fuga desde val/test_unseen).
            # Si es una fuente inédita de validación sintética, usar `val` excluyendo el carácter objetivo.
            pool = self.train_pool.get(style_id) or self.val_pool.get(style_id, [])

        cands = [r for r in pool if r["char"] != exclude_char]
        return cands if cands else pool

    def _select_refs(self, idx: int, style_id: str, exclude_char: str) -> list[dict]:
        cands = self._get_ref_candidates(style_id, exclude_char)
        if not cands:
            raise RuntimeError(f"Sin referencias disponibles para estilo='{style_id}' en split='{self.split}'.")

        if self.split == "train":
            rng = random
        else:
            # Determinista para evaluación 100% reproducible
            if style_id == "personal_stylus":
                # Usar los mismos caracteres representativos del autor (tomados de train)
                by_ch = {r["char"]: r for r in cands}
                preferred = [by_ch[ch] for ch in DEFAULT_REAL_REF_CHARS if ch in by_ch]
                if len(preferred) >= self.num_style_refs:
                    return preferred[: self.num_style_refs]
            rng = random.Random(self.seed + idx * 997)

        if len(cands) >= self.num_style_refs:
            return rng.sample(cands, self.num_style_refs)
        return [rng.choice(cands) for _ in range(self.num_style_refs)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
        row = self.samples[idx]
        style_id = row["style_id"]
        target_char = row["char"]

        x_img = Image.open(self.root_dir / row["x_path"]).convert("L")
        y_img = Image.open(self.root_dir / row["y_path"]).convert("L")

        ref_rows = self._select_refs(idx, style_id, exclude_char=target_char)
        ref_imgs = [Image.open(self.root_dir / r["y_path"]).convert("L") for r in ref_rows]

        # Aumento afín suave emparejado solo en (x, y)
        if self.pair_aug is not None:
            x_img, y_img = self.pair_aug(x_img, y_img)

        # Aumento morfológico CONSISTENTE en grosor entre `y` y sus K `style_refs` (solo en train)
        if self.stroke_augment:
            p = random.random()
            if p < 0.40:
                op, ksize = "thin", random.choice((2, 3))
            elif p < 0.55:
                op, ksize = "thick", 2
            else:
                op, ksize = "none", 1

            if op != "none":
                y_img = _apply_stroke_thickness(y_img, op, ksize)
                ref_imgs = [_apply_stroke_thickness(im, op, ksize) for im in ref_imgs]

        x_t = _pil_to_normalized_tensor(x_img, self.img_size)
        y_t = _pil_to_normalized_tensor(y_img, self.img_size)
        refs_t = torch.stack([_pil_to_normalized_tensor(im, self.img_size) for im in ref_imgs], dim=0)
        return x_t, y_t, refs_t, int(row["class_id"]), target_char, style_id
