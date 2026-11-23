# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.
El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el proyecto sigue [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Roadmap para `v1.0.0`
- Fine-tuning de Pix2Pix (`--mode finetune`) vs. entrenamiento desde cero (`--mode scratch`) sobre capturas manuscritas reales con stylus.
- Re-evaluación cuantitativa final, mapas Grad-CAM y auditoría de subgrupos sobre modelos personalizados.
- Fijación exacta de dependencias en `requirements.txt`.

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
