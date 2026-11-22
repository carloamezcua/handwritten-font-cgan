# Propuesta de Proyecto: Generación de Tipografía Manuscrita con Image-to-Image cGAN

## 1. Objetivo e Hipótesis
- **Objetivo:** Entrenar una red generativa adversarial condicional (**Pix2Pix: U-Net + PatchGAN**) capaz de traducir glifos de una tipografía base limpia (`x`) a trazos manuscritos con el estilo personal del autor (`ŷ`) en resolución `1×128×128`, evaluarla con clasificadores CNN congelados y exportar una fuente instalable en formato `.ttf`.
- **Hipótesis:** Dado un dataset propio reducido (~300 muestras reales de 75 símbolos), el **preentrenamiento sintético** con fuentes manuscritas abiertas (*Transfer Learning*) seguido de *fine-tuning* sobre muestras propias supera significativamente al entrenamiento **desde cero (*From Scratch*)**, especialmente en la generalización **Zero-Shot** sobre símbolos reservados nunca vistos durante el entrenamiento (`test_unseen`).

## 2. Alcance de Datos y Partición
- **Charset (75 símbolos × 4 repeticiones = 300 muestras):** `A-Z` (26), `a-z` (26), `0-9` (10) y signos básicos `.,!?()+-=/:;&` (13).
- **Partición estricta:**
  - `train`: ~64 símbolos × 3 repeticiones.
  - `val`: ~64 símbolos × 1 repetición (caracteres vistos).
  - `test_unseen`: ~11 símbolos reservados (`G, Q, k, z, 4, 8, ?, &, +, ;, )`) nunca vistos en entrenamiento (evaluación Zero-Shot).
  - **Símbolos inéditos (solo inferencia):** `@, #, $, %, ñ, Ñ, ¿, ¡, *`.
- **Dataset sintético:** 15–20 Google Fonts manuscritas emparejadas con la fuente base (3,000–5,000 pares `1×128×128`).
- **Dataset OOD (≥50 imágenes):** Muestras fuera de distribución (trazos extremos, ruido, inclinación o símbolos no válidos) para validar calibración y rechazo.

## 3. Arquitectura y Entrenamiento
- **Generador `G` (U-Net):** Entrada `[B, 1, 128, 128]` normalizada a `[-1, 1]`, 7 bloques encoder + 7 bloques decoder con *skip connections* y salida `Tanh`.
- **Discriminador `D` (PatchGAN):** Entrada `[x, y]` o `[x, ŷ]` (`[B, 2, 128, 128]`), mapa de logits `14×14`.
- **Pérdidas:** `BCEWithLogitsLoss` con *one-sided label smoothing* (`0.9` real / `0.0` falso) + pérdida `L1` (`λ = 100`).
- **Comparativa experimental:** Experimento A (*From Scratch*) vs. Experimento B (*Pretrain sintético + Fine-Tuning*).

## 4. Evaluación, Interpretabilidad y Despliegue (Rúbrica)
- **Evaluadores fijos (jueces):** `ResNet-18` y `EfficientNet-B0` entrenados previamente y congelados para medir **Accuracy**, **Macro-F1**, **Matriz de Confusión** y **FID de dominio** (extracción de embeddings de la penúltima capa).
- **Interpretabilidad:** Mapas de activación / Grad-CAM sobre evaluadores y análisis visual `x | ŷ | y` en `val` y `test_unseen`.
- **Optimización:** Exportación a **ONNX / TorchScript** con tabla comparativa de latencia y FPS antes/después.
- **Demo en tiempo real (≥10 FPS / ≤100 ms):** Ventana OpenCV triple (`Entrada x | Glifo ŷ | Confianza`), modo dual Canvas interactivo / Webcam ROI (`M`), umbral ajustable (`+`/`-`, default `0.70`) con alerta `OK` / `BAJA CONFIANZA (OOD)`.
- **Exportación `.ttf`:** Vectorización con `vtracer` y empaquetado Unicode con `fontTools`.
