# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.
El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el proyecto sigue [Semantic Versioning](https://semver.org/lang/es/).

## [1.0.0] - 2026-11-23

### Added
- Captura completa de **300 trazos manuscritos reales** con stylus (`data/raw_scans/hoja_01..07.png` → `data/real_pairs/`, 300/300 celdas extraídas sin celdas vacías) y auditoría manual visual de 30 muestras reales (`docs/figuras/auditoria_calidad_real_30.png`, 0/30 errores de etiqueta).
- Notebook dedicado de personalización en Google Colab GPU ([`notebooks/03_colab_finetuning.ipynb`](notebooks/03_colab_finetuning.ipynb)) comparando `pretrained` vs. `scratch` vs. `finetune` sobre `data/real_pairs/`, con figuras extraídas a `docs/figuras/`.
- Modo `--from-real-dir` y metadatos reales de trazabilidad OpenType (`AI-SYNTHETIC` / `PERSONAL-STYLUS` en tabla `name`) en `src/export/build_ttf.py`.
- Scripts de reproducibilidad y validación real: `src/data/download_artifacts.py` (descarga desde GitHub Releases) y `src/data/capturar_ood_real.py` (captura rápida de ≥50 muestras OOD reales con cámara web).
- `.gitignore` actualizado para permitir el versionado de los entregables `docs/caso_sector.pdf`, `docs/informe_etico.pdf` y `docs/defensa.pdf` (manteniendo ignorado `docs/proyecto_CV.pdf`).

### Changed
- **Optimización ONNX Runtime CPU (`export_onnx.py` y `app/demo_realtime.py`):** fijación de `intra_op_num_threads=4` e `inter_op_num_threads=1`, reduciendo la latencia del pipeline completo (`G + ResNet-18`) de `50.50 ms (19.80 FPS)` a **`21.07 ms (47.45 FPS, 1.69× speedup vs. PyTorch)`**.
- `requirements.txt` fijado con versiones exactas (`==`) del entorno probado (Python 3.11.18).
- `README.md`, `MODEL_CARD.md`, `docs/datasheet.md` y `docs/costos.md` alineados al orden exacto de la rúbrica, corrigiendo controles de la demo, tiempos reales de GPU T4 y transparencia sobre `judge_split` en `test_unseen` sintético.

### Findings
- En `data/real_pairs` (`val`), **`scratch`** (entrenado solo con trazos reales) superó a `pretrained` y `finetune` (**75.00 %** Acc y **FID 94.87** vs. `71.88 %` / `145.25` de `pretrained` y `62.50 %` / `121.14` de `finetune`) debido al choque de dominio entre el grosor sintético y el trazo fino con stylus (`2.57 %` tinta media).
- En pruebas de ablación de `finetune` con `--reset-discriminator` y `--lambda-l1 150`, seleccionar el checkpoint únicamente por `val_l1` mínimo provocó colapso a salidas casi en blanco en épocas tempranas (`val_l1 = 0.0455`), confirmando la importancia de mantener el discriminador sincronizado o seleccionar por métricas de legibilidad.

## [0.9.0-pretrain] - 2026-11-23

### Added
- Demo en tiempo real con OpenCV (`Canvas` / `Webcam ROI`), centrado automático por bounding box, espejo (`F`) y alerta OOD.
- Captura guiada de glifos manuscritos con stylus e interfaz web fluida con soporte Unicode (`A–Z`, `a–z`, `0–9`, acentos, `ñ`, puntuación).
- Evaluación cuantitativa (Accuracy, Macro-F1, L1, FID de dominio), mapas Grad-CAM, evaluación de robustez OOD y auditoría por subgrupos.
- Documentación técnica completa: `MODEL_CARD.md`, `docs/datasheet.md`, `docs/interpretabilidad.md`, `docs/robustez_y_despliegue.md`, `docs/costos.md` y `README.md`.
- Versionado semántico mediante `VERSION`, `src/__version__` y `CHANGELOG.md`.

## [0.4.0] - 2026-11-22

### Added
- Exportación del generador Pix2Pix y juez a formato ONNX con benchmark de latencia/FPS en CPU.
- Vectorización de glifos generados con `vtracer` y constructor de fuentes `.ttf` (`src/export/build_ttf.py`).

## [0.3.0] - 2026-11-22

### Added
- Arquitecturas Pix2Pix (U-Net + PatchGAN) y jueces clasificadores (`ResNet-18`, `EfficientNet-B0`).
- Scripts de entrenamiento para jueces y Pix2Pix (`pretrain`, `scratch`, `finetune`).
- Notebook de entrenamiento en Google Colab GPU (`notebooks/02_colab_training.ipynb`) y curvas de entrenamiento.

## [0.2.0] - 2026-11-22

### Added
- Generador de dataset sintético, extractor OpenCV de plantillas, set OOD y clase `Dataset` de PyTorch.
- Análisis exploratorio de datos (`notebooks/01_eda_datos.ipynb`) y gráficas de distribución de clases.

## [0.1.0] - 2026-11-22

### Added
- Inicialización del repositorio, configuración base (`src/config.py`, `src/utils.py`) y licencia MIT.
- Protocolo de captura de datos y generador de plantillas imprimibles.
