# Generación de Tipografía Manuscrita con Image-to-Image cGAN

Proyecto final de **Deep Learning para Visión de Computadoras (2026B — Orientación A-3 LACD)**.

## Descripción
Sistema de visión por computadora que aprende el estilo manuscrito personal del autor a partir de un conjunto reducido de muestras (~300 caracteres) usando una **cGAN condicional (Pix2Pix: U-Net + PatchGAN)** con **preentrenamiento sintético** (*Transfer Learning*) sobre fuentes manuscritas abiertas y *fine-tuning* sobre trazos reales.

Incluye:
- Pipeline automático de captura y normalización con OpenCV (`1×128×128`).
- Evaluación cuantitativa con **ResNet-18** y **EfficientNet-B0** congelados (Accuracy, Macro-F1, Matriz de Confusión y FID de dominio) + pruebas **Zero-Shot** y **OOD** (≥50 imágenes).
- Optimización a **ONNX / TorchScript** y **demo en tiempo real (≥10 FPS / ≤100 ms)** con Canvas/Webcam y alerta de baja confianza.
- Exportación de la tipografía generada a archivo instalable **`.ttf`**.

## Documentación
- [Propuesta del Proyecto](docs/propuesta_proyecto.md)

## Inicio Rápido
```bash
pip install -r requirements.txt
```
