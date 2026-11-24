# Model Card — Handwritten Font cGAN (Pix2Pix + Evaluadores CNN)

Model Card estructurada conforme a Mitchell et al. (2019), documentando alcance, desempeño cuantitativo, sesgos identificados y mitigaciones técnicas.

---

## 1. Información General del Modelo

| Campo | Detalle |
| :--- | :--- |
| **Nombre del sistema** | `HandwrittenFont-Pix2Pix-cGAN` |
| **Versión** | `v1.0.0` *(modelo base preentrenado `pix2pix_pretrained.pth` + variantes personales `pix2pix_scratch.pth` y `pix2pix_finetuned.pth`)* |
| **Tipo de tarea** | Traducción imagen-a-imagen condicional (*Image-to-Image Translation*) + auditoría de legibilidad |
| **Arquitectura generativa** | **Pix2Pix cGAN:** Generador `UNetGenerator` (7 bloques encoder + 7 bloques decoder con *skip connections*, salida `Tanh`, `1×128×128`) + Discriminador `PatchGANDiscriminator` (70×70, salida `14×14`) |
| **Evaluadores auxiliares (jueces)** | `ResNet-18` (1 canal, 75 clases, embeddings 512-d) y `EfficientNet-B0` (1 canal, 75 clases, embeddings 1280-d), congelados tras entrenamiento |
| **Formato de despliegue** | PyTorch (`.pth`) y **ONNX** (`generator.onnx`, `judge_resnet18.onnx`) ejecutables en CPU a **47.45 FPS (21.07 ms, speedup 1.69×)** |
| **Licencia** | MIT (código); SIL OFL 1.1 (fuentes sintéticas de preentrenamiento) |
| **Repositorio** | `https://github.com/carloamezcua/handwritten-font-cgan` |

---

## 2. Usos Previstos y Usos No Previstos

### 2.1 Usos previstos (*Intended Uses*)
- **Digitalización personal con consentimiento:** Generar una fuente tipográfica `.ttf` instalable basada en la caligrafía propia del usuario (o de un colaborador con consentimiento explícito) para apuntes digitales, diseño gráfico, ilustración o proyectos creativos.
- **Demostración técnica de *Transfer Learning* en visión:** Ejemplo reproducible de cGAN con datos limitados (300 muestras reales para adaptación), evaluación objetiva con clasificadores congelados (Macro-F1, FID de dominio), explicabilidad Grad-CAM y optimización CPU vía ONNX.
- **Prototipado de glifos *Zero-Shot*:** Estimación estilística de caracteres no dibujados explícitamente (`?, &, +, ;, @, #, $, %, ñ, ¿, ¡`) sujeta al filtro de confianza del evaluador CNN.

### 2.2 Usos estrictamente NO previstos (*Out-of-Scope / Misuse*)
- **Suplantación de identidad o falsificación:** Clonar la escritura de terceros sin consentimiento ni falsificar documentos legales, cheques, recetas médicas o firmas autógrafas.
- **Falsificación de documentos manuscritos:** Generar textos o declaraciones haciéndolos pasar por manuscritos originales de otra persona.
- **Escritura cursiva ligada de palabra completa:** El modelo opera **glifo por glifo aislado (`1×128×128`)** y no modela ligaduras continuas entre letras consecutivas ni presión dinámica de tinta sobre líneas completas.
- **Autenticación biométrica o peritaje caligráfico:** Los clasificadores jueces (`ResNet-18` / `EfficientNet-B0`) clasifican **identidad de carácter (75 clases)**, no identidad de escritor; nunca deben emplearse en verificación forense.
- **Alfabetos no latinos:** Sin soporte ni validación para escrituras CJK, árabe, cirílico, devanagari o emojis.

---

## 3. Datos de Entrenamiento y Evaluación

Ver documentación completa en [`docs/datasheet.md`](docs/datasheet.md):
- **Preentrenamiento sintético (`data/synthetic_pairs/`):** 3,825 pares `1×128×128` generados a partir de 17 fuentes manuscritas abiertas de Google Fonts (SIL OFL 1.1) emparejadas con `Lato-Regular`. Partición por fuente completa (`14` fuentes en `train`, `3` en `val`) y 11 símbolos estrictamente aislados en `test_unseen` (`G, Q, k, z, 4, 8, ?, &, +, ;, )`).
- **Adaptación personal (`data/real_pairs/`):** 300 muestras manuscritas reales del autor capturadas con stylus (`train`: 192, `val`: 64, `test_unseen`: 44) bajo el mismo contrato de partición sin fuga de información (`--mode scratch` / `--mode finetune`).
- **Evaluación de robustez OOD (`data/ood/`):** 72 imágenes en 6 categorías (ruido puro, garabatos, símbolos inéditos, rotación extrema ±45°, desenfoque/corte y sombra/bajo contraste), complementadas por `src/data/capturar_ood_real.py` para captura de muestras reales con cámara web.

---

## 4. Desempeño Cuantitativo y Robustez

### 4.1 Desempeño del modelo preentrenado (`pix2pix_pretrained.pth`, dominio sintético)

| Split | Modelo / Fuente | Accuracy (`ResNet-18` / `EffNet-B0`) | Macro-F1 (`ResNet-18` / `EffNet-B0`) | FID dominio (`ResNet-18`) |
| :--- | :--- | :---: | :---: | :---: |
| **`val`** (64 clases vistas) | Referencia `y` (techo) | 87.67 % / 85.07 % | 87.58 % / 85.30 % | 0.00 |
| **`val`** (64 clases vistas) | `pix2pix_pretrained` | 71.88 % / 68.75 % | 67.45 % / 64.58 % | 133.24 |
| **`test_unseen`** (11 Zero-Shot)* | Referencia `y` (techo global)* | 98.22 % / 97.68 % | 98.99 % / 98.61 % | 0.00 |
| **`test_unseen`** (11 Zero-Shot) | `pix2pix_pretrained` | 45.45 % / 54.55 % | 45.45 % / 50.00 % | 358.22 |

> \* **Aclaración sobre `judge_split`:** 462/561 referencias sintéticas de `test_unseen` pertenecen a fuentes usadas para entrenar los jueces (`judge_split=train`, 100.00 % Acc), mientras que las 99 de fuentes inéditas (`judge_split=val`) alcanzan **89.90 % Acc** en `ResNet-18`. Sobre `data/real_pairs/`, ningún trazo real fue visto por los jueces (0 % de traslape).

### 4.2 Comparativa sobre capturas reales con stylus (`data/real_pairs/`, 300 muestras)

Evaluación sobre las muestras personales reales (`val`: 64 clases vistas; `test_unseen`: 11 clases Zero-Shot, `n=44`) usando los dos jueces congelados (`notebooks/03_colab_finetuning.ipynb`, `--tag final`):

| Split | Modelo / Fuente | Accuracy (`ResNet-18` / `EffNet-B0`) | Macro-F1 (`ResNet-18` / `EffNet-B0`) | L1 media | FID dominio (`ResNet-18`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`val`** (64 vistas) | Referencia `y` (trazos reales) | 89.06 % / 84.38 % | 86.20 % / 81.25 % | — | 0.00 |
| **`val`** (64 vistas) | `pix2pix_pretrained` | 71.88 % / 71.88 % | 67.45 % / 67.97 % | 0.0947 | 145.25 |
| **`val`** (64 vistas) | **`pix2pix_scratch`** | **75.00 %** / 67.19 % | **69.27 %** / 59.64 % | 0.0570 | **94.87** |
| **`val`** (64 vistas) | `pix2pix_finetuned` (`100 ep, lr=1e-4`) | 62.50 % / 56.25 % | 56.38 % / 47.50 % | 0.0550 | 121.14 |
| **`test_unseen`** (11 Zero-Shot) | Referencia `y` (trazos reales) | 86.36 % / 81.82 % | 85.28 % / 78.79 % | — | 0.00 |
| **`test_unseen`** (11 Zero-Shot) | **`pix2pix_pretrained`** | 27.27 % / **45.45 %** | 27.27 % / **42.42 %** | 0.0981 | **407.37** |
| **`test_unseen`** (11 Zero-Shot) | `pix2pix_scratch` | 27.27 % / 9.09 % | 27.27 % / 3.03 % | 0.0691 | 469.96 |
| **`test_unseen`** (11 Zero-Shot) | `pix2pix_finetuned` (`100 ep, lr=1e-4`) | 9.09 % / 27.27 % | 9.09 % / 21.82 % | 0.0667 | 499.83 |

**Hallazgos experimentales clave:**
- **Choque de dominio en *fine-tuning*:** `scratch` superó a `finetune` y `pretrained` en dominio propio (`val`: **75.00 %** Acc, **FID 94.87**). Las 17 fuentes sintéticas tienen trazos más gruesos que la captura fina con stylus (densidad media de tinta `2.57 %`), provocando degradación al adaptar el modelo preentrenado sobre 192 muestras.
- **Riesgo de selección por `val_l1` con trazos finos:** Al probar `finetune` reiniciando el discriminador (`--reset-discriminator`, `--lambda-l1 150`), el generador minimizó `L1` generando lienzos casi en blanco en la época 5 (`val_l1 = 0.0455`, ya que ~97 % de los píxeles son fondo blanco).
- **Límite arquitectónico Zero-Shot y varianza muestral:** Con solo 11 clases en `test_unseen` (~9.09 pp por clase), adaptarse a un único estilo sin *encoder* de estilo explícito hace que Pix2Pix proyecte caracteres inéditos hacia letras vistas cercanas (`G, Q → O`). Por ello la demo en tiempo real usa `pix2pix_pretrained` como modelo por defecto.
- **Compilación `.ttf` personal fiel:** Para obtener una fuente instalable fiel a partir de capturas reales completas (75 símbolos × 4 repeticiones), `src/export/build_ttf.py --from-real-dir data/real_pairs` selecciona automáticamente el **medoide L1** entre las 4 repeticiones propias de cada carácter.

### 4.3 Robustez ante entradas Fuera de Distribución (OOD — 72 imágenes)
- **Separación intrínseca del juez (`ResNet-18` directo):** **AUROC = 0.9435** (rechaza 77.8 % OOD a `τ = 0.70` y 94.4 % a `τ = 0.85`).
- **Pipeline de demo (`G + ResNet-18`) a `τ = 0.70`:** **81.9 %** de rechazo global OOD (**100 %** en `garabatos` y `simbolos_ineditos`; **91.7 %** en `ruido_puro` y `rotacion_extrema`). Detalles completos en [`eval/robustez.ipynb`](eval/robustez.ipynb) y [`docs/robustez_y_despliegue.md`](docs/robustez_y_despliegue.md).

---

## 5. Sesgos Identificados, Limitaciones Técnicas y Mitigaciones Concretas

| Riesgo / Sesgo identificado | Evidencia empírica | Mitigación implementada |
| :--- | :--- | :--- |
| **1. Brecha de desempeño entre subgrupos tipográficos** | En [`eval/auditoria_subgrupos.ipynb`](eval/auditoria_subgrupos.ipynb), el modelo preentrenado alcanza **77.8 % Macro-F1** en `Mayúsculas (A–Z)` pero cae a **45.8 %** en `Signos` (brecha de **31.95 pp**). Los signos con pocos píxeles activos (`:`, `!`, `(`) y letras con punto (`i`) sufren pérdida de puntos aislados bajo pérdida L1. | (a) Auditoría automatizada por subgrupos (`auditoria_subgrupos.ipynb`) y diagnóstico Grad-CAM (`docs/interpretabilidad.md`). (b) Captura balanceada con 4 repeticiones por símbolo incluyendo todos los signos y selección por medoide L1 en `build_ttf.py`. |
| **2. Confusión morfológica y convenciones estilísticas personales** | Pares *sans-serif* (`l`↔`I`↔`1`, `O`↔`0`↔`o`, `x`↔`X`) y convenciones personales como el cero tachado (`0`→`Ø`) concentran los errores (`8.7 %` de desacuerdos del juez sobre `data/real_pairs` corresponden a variaciones estilísticas, no a etiquetas erróneas). | (a) Etiquetado descriptivo explícito durante la captura guiada (`src/data/captura_stylus.py`). (b) Prohibición estricta de *flips* horizontales/verticales en `dataset.py`. |
| **3. Artefactos en símbolos *Zero-Shot* y entradas ruidosas** | En `test_unseen`, los símbolos complejos no vistos (`4`, `Q`, `G`, `8`, `&`) presentan artefactos con confianza promedio `< 0.35`. En cámara web, sombras o garabatos pueden generar salidas sin sentido. | **Filtro de confianza en tiempo real (`app/demo_realtime.py`):** umbral interactivo (`+`/`-`, default `τ = 0.70`) con alerta `BAJA CONFIANZA (OOD)`, preprocesamiento automático por *bounding box*, limpieza de tinta vacía y validación con dos jueces independientes. |
| **4. Riesgo ético de suplantación de escritura** | Un modelo entrenado sobre la letra de un tercero podría emplearse para imitar notas manuscritas ajenas. Ver análisis completo en [`docs/informe_etico.pdf`](docs/informe_etico.pdf). | (a) **Control de datos:** entrenamiento restringido a fuentes abiertas (SIL OFL 1.1) y muestras propias del autor; no se entrenan firmas autógrafas ni trazos cursivos continuos. (b) **Metadatos de trazabilidad incrustados en `.ttf`:** [`src/export/build_ttf.py`](src/export/build_ttf.py) escribe los campos `copyright`, `uniqueFontIdentifier`, `manufacturer`, `designer`, `description` y `licenseDescription` en la tabla OpenType `name` con marcadores explícitos (`AI-SYNTHETIC` / `PERSONAL-STYLUS`). |
| **5. Sesgo lingüístico y de alfabeto** | El `CHARSET` cubre 75 símbolos ASCII (sin tildes `á, é, í, ó, ú` ni `ñ, ¿, ¡` en entrenamiento). | Los símbolos del español (`ñ, Ñ, ¿, ¡`) se documentan explícitamente como *símbolos inéditos experimentales* solo en inferencia y quedan sujetos a revisión manual antes de distribuir el `.ttf`. |

---

## 6. Mantenimiento y Actualizaciones
- Los pesos del generador pueden actualizarse sin modificar la aplicación en tiempo real re-exportando `checkpoints/generator.onnx` con `python -m src.export.export_onnx --generator-ckpt <ruta.pth>`.
- Cualquier cambio en el conjunto de datos o modelo debe re-ejecutar `eval/evaluate_models.py`, `eval/gradcam_analysis.py`, `eval/robustez.ipynb` y `eval/auditoria_subgrupos.ipynb` antes de emitir una nueva etiqueta de versión.
