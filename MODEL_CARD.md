# Model Card — Generación de Tipografía Manuscrita (Pix2Pix cGAN + Jueces CNN)

Model Card basada en Mitchell et al. (2019) para cumplir con el indicador **B7** de la rúbrica (*usos previstos y no previstos, sesgos identificados y mitigaciones concretas*).

---

## 1. Información General del Modelo

| Campo | Detalle |
| :--- | :--- |
| **Nombre del sistema** | `HandwrittenFont-Pix2Pix-cGAN` |
| **Versión** | `v0.9-pretrain` *(pendiente actualización a `v1.0-final` tras fine-tuning real en Fase 5)* |
| **Tipo de tarea** | Traducción imagen-a-imagen condicional (*Image-to-Image Translation*) + clasificación/auditoría de legibilidad |
| **Arquitectura generativa** | **Pix2Pix cGAN:** Generador `UNetGenerator` (7 bloques encoder + 7 bloques decoder con *skip connections*, salida `Tanh`, `1×128×128`) + Discriminador `PatchGANDiscriminator` (70×70, salida `14×14`) |
| **Evaluadores auxiliares (jueces)** | `ResNet-18` (1 canal, 75 clases, embeddings 512-d) y `EfficientNet-B0` (1 canal, 75 clases, embeddings 1280-d), congelados tras entrenamiento |
| **Formato de despliegue** | PyTorch (`.pth`) y **ONNX** (`generator.onnx`, `judge_resnet18.onnx`) ejecutables en CPU a **19.80 FPS (50.50 ms)** |
| **Licencia** | MIT (código); SIL OFL 1.1 (fuentes sintéticas de preentrenamiento) |
| **Repositorio** | `https://github.com/carloamezcua/handwritten-font-cgan` |

---

## 2. Usos Previstos y Usos No Previstos (Rúbrica B7)

### 2.1 Usos previstos (*Intended Uses*)
- **Digitalización personal con consentimiento:** Generar una fuente tipográfica `.ttf` instalable basada únicamente en la caligrafía propia del usuario (o de un colaborador con consentimiento explícito y documentado) para apuntes digitales, cómics, diseño gráfico, material educativo o proyectos creativos.
- **Investigación y docencia en Visión por Computadora:** Demostración reproducible de *Transfer Learning* en cGAN con pocos datos (~300 muestras reales), evaluación objetiva mediante clasificadores congelados (Macro-F1, FID de dominio), interpretabilidad Grad-CAM y despliegue en tiempo real sobre CPU con ONNX.
- **Prototipado rápido de glifos *Zero-Shot*:** Estimación estilística de caracteres no escritos por el autor (símbolos adicionales como `?, &, +, ;, @, #, $, %, ñ, ¿, ¡`) sujetos a verificación de confianza por el juez CNN.

### 2.2 Usos estrictamente NO previstos (*Out-of-Scope / Misuse*)
- **Suplantación de identidad o falsificación de escritura/firmas:** Queda terminantemente fuera de alcance clonar la letra manuscrita de terceras personas sin su consentimiento, falsificar documentos legales, cheques, recetas médicas, pagarés o firmas autógrafas.
- **Falsificación académica o fraudulenta:** Generar tareas, cartas o declaraciones haciéndolas pasar por manuscritos originales de otra persona.
- **Escritura cursiva ligada de palabra completa:** El modelo opera **glifo por glifo aislado (`1×128×128`)** y no modela ligaduras continuas entre letras consecutivas ni presión dinámica de tinta sobre líneas completas de texto.
- **Sistemas de autenticación biométrica o peritaje caligráfico:** Los clasificadores jueces (`ResNet-18` / `EfficientNet-B0`) clasifican **identidad de carácter (75 clases)**, no identidad de escritor; nunca deben usarse para verificación forense ni grafoscopía.
- **Alfabetos no latinos:** El sistema no está entrenado ni validado para escrituras CJK, árabe, cirílico, devanagari ni emojis.

---

## 3. Datos de Entrenamiento y Evaluación

Ver documentación completa en [`docs/datasheet.md`](docs/datasheet.md):
- **Preentrenamiento sintético (`data/synthetic_pairs/`):** 3,825 pares `1×128×128` generados a partir de 17 fuentes manuscritas abiertas de Google Fonts (SIL OFL 1.1) emparejadas con `Lato-Regular`. Partición por fuente completa (`14` fuentes en `train`, `3` en `val`) y 11 símbolos estrictamente aislados en `test_unseen` (`G, Q, k, z, 4, 8, ?, &, +, ;, )`).
- **Fine-tuning real (`data/real_pairs/`, Fase 5):** 300 muestras manuscritas propias del autor (`train`: 192, `val`: 64, `test_unseen`: 44) con el mismo contrato de partición sin fuga de información.
- **Evaluación de robustez OOD (`data/ood/`):** 72 imágenes en 6 categorías (ruido puro, garabatos, símbolos inéditos, rotación extrema ±45°, desenfoque/corte y sombra/bajo contraste).

---

## 4. Desempeño Cuantitativo y Robustez

### 4.1 Desempeño actual del modelo preentrenado (`pix2pix_pretrained.pth`)

| Split | Modelo / Fuente | Accuracy (`ResNet-18` / `EffNet-B0`) | Macro-F1 (`ResNet-18` / `EffNet-B0`) | FID dominio (`ResNet-18`) |
| :--- | :--- | :---: | :---: | :---: |
| **`val`** (64 clases vistas) | Referencia `y` (techo) | 87.67 % / 85.07 % | 87.58 % / 85.30 % | 0.00 |
| **`val`** (64 clases vistas) | `pix2pix_pretrained` | 71.88 % / 68.75 % | 67.45 % / 64.58 % | 133.24 |
| **`test_unseen`** (11 Zero-Shot) | Referencia `y` (techo) | 98.22 % / 97.68 % | 98.99 % / 98.61 % | 0.00 |
| **`test_unseen`** (11 Zero-Shot) | `pix2pix_pretrained` | 45.45 % / 54.55 % | 45.45 % / 50.00 % | 358.22 |

> *(En la Fase 5 se añadirá la comparativa directa **Experimento A — From Scratch** vs. **Experimento B — Pretrain + Fine-Tuning** sobre `data/real_pairs/`).*

### 4.2 Robustez ante entradas Fuera de Distribución (OOD — 72 imágenes)
- **Separación intrínseca del juez (`ResNet-18` directo):** **AUROC = 0.9435** (rechaza 77.8 % OOD a `τ = 0.70` y 94.4 % a `τ = 0.85`).
- **Pipeline de demo (`G + ResNet-18`) a `τ = 0.70`:** **81.9 %** de rechazo global OOD (**100 %** en `garabatos` y `simbolos_ineditos`; **91.7 %** en `ruido_puro` y `rotacion_extrema`). Detalles completos en [`eval/robustez.ipynb`](eval/robustez.ipynb) y [`docs/robustez_y_despliegue.md`](docs/robustez_y_despliegue.md).

---

## 5. Sesgos Identificados, Limitaciones Técnicas y Mitigaciones Concretas (Rúbrica B7)

| Riesgo / Sesgo identificado | Evidencia empírica en el proyecto | Mitigación concreta implementada |
| :--- | :--- | :--- |
| **1. Brecha de desempeño entre subgrupos tipográficos** | En [`eval/auditoria_subgrupos.ipynb`](eval/auditoria_subgrupos.ipynb), el modelo preentrenado alcanza **77.8 % Macro-F1** en `Mayúsculas (A–Z)` pero cae a **45.8 %** en `Signos` (brecha de **31.95 pp**). Los signos con pocos píxeles activos (`:`, `!`, `(`) y letras con punto (`i`) sufren pérdida de puntos aislados bajo pérdida L1. | (a) Auditoría automatizada por subgrupos (`auditoria_subgrupos.ipynb`) y diagnóstico Grad-CAM (`docs/interpretabilidad.md`). (b) Captura balanceada con 4 repeticiones por símbolo incluyendo todos los signos para el *fine-tuning* sobre estilo único coherente. |
| **2. Confusión morfológica en caracteres ambiguos** | Pares *sans-serif* casi idénticos (`l`↔`I`↔`1`, `O`↔`0`↔`o`, `K`↔`k`, `:`↔`;`) concentran los principales errores en `eval_errores_pretrain_pretrained.png`. | (a) Etiquetado descriptivo explícito durante la captura guiada (`src/data/captura_wacom.py`, ej. `"Mayúscula 'I'"` vs. `"Minúscula 'l'"`) para alentar diferenciación de trazos. (b) Prohibición estricta de *flips* horizontales/verticales en `dataset.py`. |
| **3. Artefactos en símbolos *Zero-Shot* y entradas ruidosas** | En `test_unseen`, los símbolos complejos no vistos (`4`, `Q`, `G`, `8`, `&`) presentan artefactos con confianza promedio `< 0.35`. En cámara web, sombras o garabatos pueden generar salidas sin sentido. | **Filtro de confianza en tiempo real (`app/demo_realtime.py`):** umbral interactivo (`+`/`-`, default `τ = 0.70`) con alerta `BAJA CONFIANZA (OOD)`, más preprocesamiento automático por *bounding box*, limpieza de tinta vacía y validación con dos jueces independientes. |
| **4. Riesgo ético de suplantación de escritura** | Un modelo entrenado sobre la letra de un tercero podría emplearse para imitar notas manuscritas ajenas. | (a) **Control de datos:** entrenamiento restringido a fuentes abiertas (SIL OFL) y muestras propias del autor con consentimiento explícito; no se entrenan firmas autógrafas ni trazos cursivos continuos. (b) **Metadato de transparencia en `.ttf`:** el constructor [`src/export/build_ttf.py`](src/export/build_ttf.py) incrusta metadatos explícitos de origen sintético/IA en la tabla `name` de la fuente (marca `HandwrittenCGAN`) y no permite falsificar trazos vectoriales de presión dinámica. (c) Análisis extendido en `docs/informe_etico.pdf`. |
| **5. Sesgo lingüístico y de alfabeto** | El `CHARSET` cubre 75 símbolos ASCII (sin tildes `á, é, í, ó, ú` ni `ñ, ¿, ¡` en entrenamiento). | Los símbolos del español (`ñ, Ñ, ¿, ¡`) se documentan explícitamente como *símbolos inéditos experimentales* solo en inferencia y quedan sujetos a revisión manual antes de publicar el `.ttf`. |

---

## 6. Mantenimiento y Actualizaciones
- Los pesos del generador pueden actualizarse sin modificar la aplicación en tiempo real re-exportando `checkpoints/generator.onnx` con `python -m src.export.export_onnx`.
- Cualquier cambio en el conjunto de datos o modelo debe re-ejecutar `eval/evaluate_models.py`, `eval/gradcam_analysis.py`, `eval/robustez.ipynb` y `eval/auditoria_subgrupos.ipynb` antes de emitir una nueva etiqueta de versión.
