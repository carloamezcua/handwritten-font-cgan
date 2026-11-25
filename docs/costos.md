# Estimación de Costos de Cómputo, Almacenamiento y Operación

Análisis de costos técnicos, tiempos reales de entrenamiento medidos en GPU y requerimientos operativos de **Handwritten Font cGAN**.

---

## 1. Infraestructura Utilizada

| Entorno | Hardware | Uso en el pipeline | Costo monetario real |
| :--- | :--- | :--- | :---: |
| **PC Local (Desarrollo e Inferencia)** | Windows 10, CPU Intel (12 núcleos lógicos), sin GPU dedicada | Generación de datasets, extractor OpenCV, captura con stylus, exportación ONNX, vectorización `.ttf`, evaluación/Grad-CAM y demo en tiempo real | \$0.00 (equipo local) |
| **Google Colab (Nivel Gratuito)** | GPU **NVIDIA Tesla T4 (16 GB VRAM)**, PyTorch CUDA | Entrenamiento de clasificadores (`ResNet-18`, `EfficientNet-B0`), preentrenamiento Pix2Pix y experimentos de personalización (`scratch` y `finetune`) | \$0.00 (capa gratuita) |
| **GitHub + Releases** | Repositorio de código, documentación y distribución de pesos | Control de versiones y hospedaje de artefactos `.onnx` / `.pth` / `real_pairs.zip` | \$0.00 (capa gratuita) |

---

## 2. Costo de Entrenamiento en GPU (Medido en `02_colab_training.ipynb` y `03_colab_finetuning.ipynb`)

Tiempos reales medidos en **NVIDIA Tesla T4** (Google Colab) y su equivalente comercial de referencia tomando como tarifa estándar en la nube **\$0.35 USD/hora** (instancia tipo `g4dn.xlarge` / T4 *on-demand*):

| Etapa / Modelo | Dataset (pares) | Épocas ejecutadas | Tiempo real en T4 | Velocidad | Tamaño checkpoint | Costo real (Colab Free) | Equivalente comercial T4 (\$0.35/h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Juez `ResNet-18`** | Sintético (3,825) | 30 *(mejor: ep. 23)* | 99.9 s (1.67 min) | 3.33 s/época | 42.84 MB | \$0.00 | \$0.010 USD |
| **Juez `EfficientNet-B0`** | Sintético (3,825) | 22 *(early stop, mejor: ep. 15)* | 117.3 s (1.96 min) | 5.33 s/época | 15.95 MB | \$0.00 | \$0.011 USD |
| **Pix2Pix `Pretrain`** | Sintético (2,688 train) | 60 *(mejor: ep. 11)* | 832.6 s (13.88 min) | 13.88 s/época | 170.16 MB | \$0.00 | \$0.081 USD |
| **Pix2Pix `Fine-Tune`** | Real stylus (192 train) | 100 *(mejor: ep. 51)* | 235.0 s (3.92 min) | 2.35 s/época | 170.16 MB | \$0.00 | \$0.023 USD |
| **Pix2Pix `Scratch`** | Real stylus (192 train) | 150 *(mejor: ep. 120)* | 309.1 s (5.15 min) | 2.06 s/época | 170.16 MB | \$0.00 | \$0.030 USD |
| **StyleUNet `Pretrain`** *(aditivo)* | Sintético (2,688 train) | 60 *(mejor: ep. 59)* | 2,782.9 s (46.38 min) | 46.38 s/época | ~180 MB | \$0.00 | \$0.271 USD |
| **StyleUNet `Fine-Tune`** *(aditivo)* | Real stylus (192 train) | 40 *(mejor: ep. 4)* | 157.0 s (2.62 min) | 3.93 s/época | ~180 MB | \$0.00 | \$0.015 USD |
| **Total ciclo completo GPU (medido)** | — | **462 épocas** | **4,533.8 s (75.56 min)** | — | **~929 MB** | **\$0.00** | **\$0.441 USD** |

---

## 3. Costo de Despliegue e Inferencia en CPU Local

A diferencia de modelos de difusión (*Stable Diffusion* / *ControlNet*) que exigen GPU dedicada (>6 GB VRAM) y varios segundos por glifo, la arquitectura elegida (**Pix2Pix U-Net `1×128×128` + ResNet-18 exportados a ONNX Runtime** con `intra_op_num_threads=4`) opera completamente en **CPU estándar** sin costos recurrentes de servidores GPU:

| Operación en CPU (Intel 12 hilos) | Latencia medida | Rendimiento | Costo operativo (local) |
| :--- | :---: | :---: | :---: |
| Generador `G` (`generator.onnx`, 159.56 MB) | 16.86 ms | **59.30 FPS** | \$0.00 |
| Juez `ResNet-18` (`judge_resnet18.onnx`, 42.75 MB) | 4.69 ms | **213.22 FPS** | \$0.00 |
| **Pipeline demo en tiempo real (`G + Juez`)** | **21.07 ms (p95: 24.00 ms)** | **47.45 FPS** *(1.69× speedup vs. PyTorch; cumple ≥10 FPS)* | **\$0.00** |
| Generación + vectorización de fuente completa (84 glifos ONNX: 75 base + 9 inéditos, o 75 reales) | ~15–25 s totales | 1 fuente `.ttf` en <30 s | \$0.00 |

---

## 4. Almacenamiento del Proyecto

| Artefacto | Ubicación | Tamaño aproximado | En Git |
| :--- | :--- | :---: | :---: |
| Código fuente, notebooks ejecutados, docs, PDFs y figuras | `src/`, `app/`, `eval/`, `notebooks/`, `docs/` | ~15 MB | Sí |
| Dataset sintético (3,825 pares `128×128`) + 18 fuentes `.ttf` | `data/synthetic_pairs/`, `data/fonts/` | ~45 MB | No (reproducible en <1 min) |
| Dataset OOD (72 imágenes) + plantillas PDF/PNG | `data/ood/`, `data/templates/` | ~4 MB | Parcial (scripts y layout sí) |
| Dataset real (7 hojas crudas + 300 pares `128×128`) | `data/raw_scans/`, `data/real_pairs/` | ~20 MB | No (GitHub Release `real_pairs.zip`) |
| Checkpoints PyTorch (`.pth`) + modelos exportados (`.onnx`) | `checkpoints/` | ~600 MB | No (GitHub Release) |
| Fuentes tipográficas instalables exportadas (`.ttf`) | `results/*.ttf` | ~0.05 MB | No |
| **Total en disco (entorno completo)** | — | **< 1 GB** | — |

---

## 5. Conclusión Económica y Escalabilidad
- **Viabilidad de bajo costo:** Todo el ciclo de vida ejecutado (jueces, Pix2Pix `pretrain`/`scratch`/`finetune`, experimento aditivo `StyleUNet` `pretrain`/`finetune`, evaluación, exportación ONNX y generación `.ttf`) costó **\$0.00 USD** usando Google Colab gratuito y equipo local (**\$0.441 USD** en cómputo comercial *on-demand*).
- **Escenario de producción (1,000 usuarios/mes):** Con `StyleUNet`, la inferencia *Few-Shot* a partir de 4 glifos de referencia cuesta **\$0.00 de GPU** por usuario, y un *fine-tuning* ligero de ~2.6 min en GPU T4 tiene un costo marginal de **~\$0.015 USD por fuente personalizada** (~15 USD/mes para 1,000 fuentes), mientras la inferencia interactiva corre en el cliente (CPU / ONNX Runtime a 47 FPS).
