# Estimación de Costos de Cómputo, Almacenamiento y Operación

Análisis de costos técnicos, tiempos de entrenamiento y requerimientos operativos de **Handwritten Font cGAN**.

---

## 1. Infraestructura Utilizada

| Entorno | Hardware | Uso en el pipeline | Costo monetario real |
| :--- | :--- | :--- | :---: |
| **PC Local (Desarrollo e Inferencia)** | Windows 10, CPU Intel (12 núcleos lógicos), sin GPU dedicada | Generación de datasets, extractor OpenCV, captura con stylus, exportación ONNX, vectorización `.ttf`, evaluación/Grad-CAM y demo en tiempo real | \$0.00 (equipo local) |
| **Google Colab (Nivel Gratuito)** | GPU **NVIDIA Tesla T4 (16 GB VRAM)**, Python 3.12, PyTorch CUDA | Entrenamiento de clasificadores (`ResNet-18`, `EfficientNet-B0`), preentrenamiento Pix2Pix y experimentos de adaptación | \$0.00 (capa gratuita) |
| **GitHub** | Repositorio de código y documentación | Control de versiones del código, notebooks y gráficas de evaluación | \$0.00 (capa gratuita) |

---

## 2. Costo de Entrenamiento en GPU (Medido en `02_colab_training.ipynb`)

Tiempos reales medidos en **NVIDIA Tesla T4** (Google Colab) y su equivalente comercial de referencia tomando como tarifa estándar en la nube **\$0.35 USD/hora** (instancia tipo `g4dn.xlarge` / T4 *on-demand*):

| Etapa / Modelo | Dataset (pares) | Épocas ejecutadas | Tiempo real en T4 | Velocidad | Tamaño checkpoint | Costo real (Colab Free) | Equivalente comercial T4 (\$0.35/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Juez `ResNet-18`** | Sintético (3,825) | 30 *(mejor: ep. 23)* | 99.9 s (1.67 min) | 3.33 s/época | 42.84 MB | \$0.00 | \$0.010 USD |
| **Juez `EfficientNet-B0`** | Sintético (3,825) | 22 *(early stop, mejor: ep. 15)* | 117.3 s (1.96 min) | 5.33 s/época | 15.95 MB | \$0.00 | \$0.011 USD |
| **Pix2Pix `Pretrain`** | Sintético (2,688 train) | 60 *(mejor: ep. 11)* | 832.6 s (13.88 min) | 13.88 s/época | 170.16 MB | \$0.00 | \$0.081 USD |
| **Subtotal preentrenamiento (ejecutado)** | — | **112 épocas** | **1,049.8 s (17.50 min)** | — | **228.95 MB** | **\$0.00** | **\$0.102 USD** |
| *Adaptación personal — `scratch` (proyectado)* | *Real (192 train)* | *~100 épocas* | *~120 s (2.0 min)* | *~1.2 s/época* | *~170 MB* | *\$0.00* | *~\$0.012 USD* |
| *Adaptación personal — `finetune` (proyectado)* | *Real (192 train)* | *~80 épocas* | *~100 s (1.7 min)* | *~1.2 s/época* | *~170 MB* | *\$0.00* | *~\$0.010 USD* |
| **Total ciclo completo GPU** | — | **~292 épocas** | **~21.2 min** | — | **~569 MB** | **\$0.00** | **~\$0.124 USD** |

---

## 3. Costo de Despliegue e Inferencia en CPU Local

A diferencia de modelos de difusión (*Stable Diffusion* / *ControlNet*) que exigen GPU dedicada (>6 GB VRAM) y varios segundos por glifo, la arquitectura elegida (**Pix2Pix U-Net `1×128×128` + ResNet-18 exportados a ONNX**) opera completamente en **CPU estándar** sin costos recurrentes de servidores GPU:

| Operación en CPU (Intel 12 hilos) | Latencia medida | Rendimiento | Costo operativo (local) |
| :--- | :---: | :---: | :---: |
| Generador `G` (`generator.onnx`, 159.56 MB) | 11.45 ms | **87.32 FPS** | \$0.00 |
| Juez `ResNet-18` (`judge_resnet18.onnx`, 42.75 MB) | 6.82 ms | **146.59 FPS** | \$0.00 |
| **Pipeline demo en tiempo real (`G + Juez`)** | **50.50 ms** | **19.80 FPS** *(cumple ≥10 FPS)* | **\$0.00** |
| Generación + vectorización de fuente completa (84 glifos `.ttf`) | ~15–25 s totales | 1 fuente completa en <30 s | \$0.00 |

---

## 4. Almacenamiento del Proyecto

| Artefacto | Ubicación | Tamaño aproximado | En Git |
| :--- | :--- | :---: | :---: |
| Código fuente, notebooks ejecutados, docs y figuras | `src/`, `app/`, `eval/`, `notebooks/`, `docs/` | ~12 MB | Sí |
| Dataset sintético (3,825 pares `128×128`) + 18 fuentes `.ttf` | `data/synthetic_pairs/`, `data/fonts/` | ~45 MB | No (reproducible en <1 min) |
| Dataset OOD (72 imágenes) + plantillas PDF/PNG | `data/ood/`, `data/templates/` | ~4 MB | Parcial (scripts y layout sí) |
| Dataset real (7 hojas crudas + 300 pares `128×128`) | `data/raw_scans/`, `data/real_pairs/` | ~20 MB | No |
| Checkpoints PyTorch (`.pth`) + modelos exportados (`.onnx`) | `checkpoints/` | ~430 MB | No |
| Fuente tipográfica instalable exportada (`.ttf`) | `results/*.ttf` | ~0.1–0.3 MB | No |
| **Total en disco (entorno completo)** | — | **< 1 GB** | — |

---

## 5. Conclusión Económica y Escalabilidad
- **Viabilidad de bajo costo:** Todo el ciclo de vida (entrenamiento, evaluación, exportación ONNX y generación `.ttf`) cuesta **\$0.00 USD** usando recursos gratuitos o equipo local (y ~**\$0.12 USD** en cómputo comercial *on-demand*).
- **Escenario de producción (1,000 usuarios/mes):** Un servicio en la nube que realizara *fine-tuning* personalizado de ~2 minutos en GPU T4 por usuario tendría un costo marginal de cómputo de **~\$0.012 USD por fuente personalizada** (~12 USD/mes para 1,000 fuentes), mientras que la inferencia interactiva puede ejecutarse en cliente (CPU / ONNX Runtime Web) sin costo de servidor GPU.
