# Robustez y Despliegue (ONNX Runtime, Latencia y OOD)

Benchmarks de exportación a ONNX, latencia en CPU y evaluación de robustez ante entradas Fuera de Distribución (OOD).

## 1. Entorno de Hardware y Software (Benchmark CPU)
- **Sistema Operativo:** Windows 10
- **Procesador:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- **Núcleos lógicos CPU:** 12
- **Python / PyTorch / ONNX Runtime:** `3.11.18` / `2.15.0+cpu` / `1.32.0`
- **Protocolo de medición:** `100` inferencias consecutivas en CPU (`batch_size = 1`, entrada `1×128×128`), tras `10` ejecuciones de calentamiento.

## 2. Modelos Exportados y Verificación Numérica
| Componente | Checkpoint PyTorch | Modelo ONNX | Tamaño ONNX (MB) | Error máx. abs. (PyTorch vs. ONNX) |
| :--- | :--- | :--- | :---: | :---: |
| Generador `G` (`UNetGenerator`) | `pix2pix_pretrained.pth` | `generator.onnx` | 159.56 | `8.97e-06` |
| Juez (`resnet18`) | `judge_resnet18.pth` | `judge_resnet18.onnx` | 42.75 | `1.91e-06` |

## 3. Comparativa de Latencia y FPS: PyTorch vs. ONNX Runtime
| Etapa | PyTorch CPU (ms) | PyTorch FPS | ONNX Runtime CPU (ms) | ONNX p95 (ms) | ONNX FPS | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Generador `G` (`UNetGenerator`) | 27.34 ± 5.73 | 36.58 | 11.45 ± 0.71 | 12.81 | 87.32 | 2.39× |
| Juez (`resnet18`) | 7.30 ± 0.74 | 137.05 | 6.82 ± 8.04 | 24.95 | 146.59 | 1.07× |
| **Pipeline completo (`G + Juez`)** | **34.56 ± 2.78** | **28.94** | **50.50 ± 25.68** | **102.25** | **19.80** | **0.68×** |

> **Rendimiento en tiempo real:** **19.80 FPS (`50.50 ms` promedio en CPU)**, cumpliendo el objetivo de operación fluida `>= 10 FPS / <= 100 ms`.

## 4. Demo en Tiempo Real y Calibración OOD
- **Arranque:** `python app/demo_realtime.py`
- **Modos (`M` / `Tab`):** Canvas interactivo (teclado / trazo libre con mouse) y Webcam ROI (flujo continuo cuadro por cuadro).
- **Umbral de confianza (`+` / `-`, default `0.70`):** si la probabilidad máxima del juez sobre el glifo generado cae por debajo del umbral (o si la entrada está vacía / con ruido extremo), se muestra la alerta visual `BAJA CONFIANZA (OOD)`.

### 4.1 Resumen de Evaluación OOD (72 imágenes — [`eval/robustez.ipynb`](../eval/robustez.ipynb))

| Escenario | AUROC (OOD vs. ID) | Rechazo OOD @ `τ=0.70` | Falso rechazo ID @ `τ=0.70` | Óptimo Youden (`τ`) | Rechazo OOD óptimo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 directo** (`y` vs. OOD) | **0.9435** | 77.8 % | **10.8 %** | 0.85 | **94.4 %** |
| **EfficientNet-B0 directo** (`y` vs. OOD) | **0.8856** | 70.8 % | **9.6 %** | 0.80 | 77.8 % |
| **Pipeline Demo (`G + ResNet-18`)** | 0.7574 | **81.9 %** | 39.1 % | 0.75 | **90.3 %** |
| **Pipeline Demo (`G + EfficientNet-B0`)** | 0.8162 | **81.9 %** | 39.1 % | 0.55 | 76.4 % |

**Tasa de rechazo por categoría OOD a `τ = 0.70` (12 imágenes por categoría):**

| Categoría OOD | ResNet-18 directo | EfficientNet-B0 directo | Pipeline Demo (`G + ResNet-18`) |
| :--- | :---: | :---: | :---: |
| `garabatos` | 83.3 % | 100.0 % | **100.0 %** |
| `simbolos_ineditos` (`@ # $ % ñ ¿ ¡ *`) | 75.0 % | 66.7 % | **100.0 %** |
| `ruido_puro` | 58.3 % | 66.7 % | **91.7 %** |
| `rotacion_extrema` (±45°) | 66.7 % | 83.3 % | **91.7 %** |
| `sombra_bajo_contraste` | **100.0 %** | 58.3 % | 75.0 % |
| `desenfoque_corte` | **83.3 %** | 50.0 % | 33.3 % *(U-Net restaura trazos borrosos)* |

> Evidencia visual completa en [`figuras/robustez_ood_distribucion.png`](figuras/robustez_ood_distribucion.png), [`figuras/robustez_ood_roc_umbral.png`](figuras/robustez_ood_roc_umbral.png), [`figuras/robustez_ood_categorias.png`](figuras/robustez_ood_categorias.png) y [`figuras/robustez_ood_ejemplos.png`](figuras/robustez_ood_ejemplos.png).
