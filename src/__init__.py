"""Paquete principal de Tipografía Neural."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
__version__ = (
    _VERSION_FILE.read_text(encoding="utf-8").strip()
    if _VERSION_FILE.exists()
    else "0.9.0-pretrain"
)

__all__ = ["__version__"]
