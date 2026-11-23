# Robustez y Despliegue (ONNX, Latencia y Hardware)

Documento de evidencia para los indicadores **C1–C5** de la rúbrica.

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

## 3. Comparativa de Latencia y FPS: Antes (PyTorch) vs. Después (ONNX Runtime)
| Etapa | PyTorch CPU (ms) | PyTorch FPS | ONNX Runtime CPU (ms) | ONNX p95 (ms) | ONNX FPS | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Generador `G` (`UNetGenerator`) | 27.34 ± 5.73 | 36.58 | 11.45 ± 0.71 | 12.81 | 87.32 | 2.39× |
| Juez (`resnet18`) | 7.30 ± 0.74 | 137.05 | 6.82 ± 8.04 | 24.95 | 146.59 | 1.07× |
| **Pipeline completo (`G + Juez`)** | **34.56 ± 2.78** | **28.94** | **50.50 ± 25.68** | **102.25** | **19.80** | **0.68×** |

> **Veredicto de tiempo real (Rúbrica C2 / C5):** **CUMPLE (>= 10 FPS / <= 100 ms)** (`50.50 ms` promedio / `19.80 FPS` en CPU).

## 4. Demo en Tiempo Real y Calibración OOD
- **Arranque en 1 comando:** `python app/demo_realtime.py`
- **Modos (`M`):** Canvas interactivo (teclado / lista de caracteres) y Webcam ROI (flujo continuo cuadro por cuadro).
- **Umbral de confianza (`+` / `-`, default `0.70`):** si la probabilidad máxima del juez sobre el glifo generado cae por debajo del umbral (o si la entrada está vacía / con ruido extremo), se muestra la alerta visual `BAJA CONFIANZA (OOD)`.
- *(En la Fase 6 se añadirá aquí el resumen de tasas de rechazo OOD obtenido desde `eval/robustez.ipynb`.)*
