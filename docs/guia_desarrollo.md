# Guía de Desarrollo Autocontenida — Generación de Tipografía Manuscrita (Pix2Pix cGAN)

Esta guía organiza el proyecto en **fases secuenciales con checkpoints de verificación** para avanzar sin acumular deuda técnica y cubrir el 100 % de la rúbrica ([proyecto_CV.pdf](proyecto_CV.pdf)) y la propuesta ([propuesta_proyecto.md](propuesta_proyecto.md)).

> **Convención de ejecución:**
> - `[Local CPU]` — Se programa y prueba en tu PC (entorno `handwritten-font-cgan`).
> - `[Colab GPU]` — Se ejecuta en Google Colab con GPU (T4 o superior) montando Drive o clonando el repo.
> - **Bloqueo de hojas impresas:** Las **Fases 0 a 4** no requieren tus hojas impresas. Puedes completar casi todo el sistema y dejar el *fine-tuning* real para la **Fase 5**.

---

## Reglas Anti-Deuda Técnica (leer antes de empezar)

1. **Una sola fuente de verdad (`src/config.py`):** Define una sola vez `CHARSET` (75 símbolos), `TEST_UNSEEN_CHARS` (`G, Q, k, z, 4, 8, ?, &, +, ;, )`), `IMG_SIZE = 128`, `SEED = 42` y rutas base. Todos los scripts importan de ahí; nunca copies/pegues el charset.
2. **Contrato de datos único (sintético = real):** Tanto `data/synthetic_pairs/` como `data/real_pairs/` deben tener exactamente la misma estructura de carpetas (`train/`, `val/`, `test_unseen/`) y el mismo formato de imagen (`1×128×128`, escala de grises, fondo blanco `255` y trazo negro `0`, normalizado a `[-1, 1]` al entrar a la red). Así el mismo `Dataset` sirve para preentrenamiento y *fine-tuning*.
3. **Cero fuga de datos (Rúbrica B3):** `test_unseen/` nunca se usa para entrenar, ajustar hiperparámetros ni *early stopping*. Solo se evalúa al final.
4. **Semillas fijadas desde el día 1 (Rúbrica A5):** Crea una función `seed_everything(42)` en `src/utils.py` y llámala al inicio de cada script de datos/entrenamiento.
5. **No avances sin pasar el Checkpoint:** Al terminar cada fase, ejecuta su prueba de humo (*smoke test*) y haz un `git commit` limpio con mensaje descriptivo.

---

## Estructura Objetivo del Repositorio (Rúbrica A6)

Asegúrate de ir poblando exactamente esta estructura:

```text
final-project_CV/
├── README.md                  # Completo en el orden exacto de la sección 4 (A4)
├── MODEL_CARD.md              # Usos previstos/no previstos, sesgos, mitigaciones (B7)
├── requirements.txt           # Fijar versiones (==) antes del tag final (A5)
├── LICENSE
├── .gitignore
├── data/                      # Solo README + scripts de descarga (nunca datos pesados)
├── src/
│   ├── config.py              # Constantes globales (charset, splits, seed, rutas)
│   ├── utils.py               # seed_everything, guardado de cuadrículas x|ŷ|y
│   ├── data/
│   │   ├── 01_generar_plantilla.py   # [HECHO]
│   │   ├── 02_extraer_recortes.py    # Segmentación OpenCV de hojas
│   │   ├── 03_generar_sinteticos.py  # Dataset sintético (Google Fonts)
│   │   ├── 04_generar_ood.py         # >=50 imágenes OOD para calibración/rechazo
│   │   └── dataset.py                # PyTorch Dataset/DataLoader unificado
│   ├── models/
│   │   ├── pix2pix.py         # Generador U-Net + Discriminador PatchGAN + pérdidas
│   │   └── judges.py          # ResNet-18 y EfficientNet-B0 (clasificadores + embeddings FID)
│   ├── train/
│   │   ├── train_judges.py    # Entrenamiento y congelado de jueces
│   │   └── train_pix2pix.py   # Pretrain sintético, From Scratch (Exp A) y Fine-Tuning (Exp B)
│   └── export/
│       ├── export_onnx.py     # Exportación ONNX/TorchScript + benchmark latencia/FPS
│       └── build_ttf.py       # Vectorización vtracer + empaquetado .ttf con fontTools
├── app/
│   └── demo_realtime.py       # Ventana OpenCV triple (Canvas/Webcam + FPS + umbral OOD)
├── eval/
│   ├── evaluate_models.py     # Accuracy, Macro-F1, Matriz Confusión, FID, análisis de errores
│   ├── gradcam_analysis.py    # Mapas de activación / Grad-CAM sobre jueces
│   ├── robustez.ipynb         # Pruebas OOD (>=50 imgs) y comparación vs test original (B6)
│   └── auditoria_subgrupos.ipynb  # Desempeño por subgrupo: mayús/minús/dígitos/signos
├── notebooks/
│   ├── 01_eda_datos.ipynb     # Distribución de clases, desbalance, inspección visual (B4)
│   └── 02_colab_training.ipynb # Notebook limpio para correr entrenamientos en Colab GPU
└── docs/
    ├── propuesta_proyecto.md  # [HECHO]
    ├── guia_desarrollo.md     # [ESTE ARCHIVO]
    ├── datasheet.md           # B1, B2, B5 (incluye revisión manual de >=20 muestras)
    ├── interpretabilidad.md   # Grad-CAM + análisis visual x|ŷ|y (Avance A)
    ├── robustez_y_despliegue.md # Tabla latencia antes/después ONNX, OOD, hardware (C5)
    ├── costos.md              # Costo de cómputo/almacenamiento/tiempo (A6)
    ├── caso_sector.pdf        # Contexto del problema/sector (A6)
    ├── informe_etico.pdf      # Riesgos de suplantación de firma/escritura y mitigaciones (B7)
    ├── defensa.pdf            # Presentación final (máx. 12 min)
    └── figuras/               # Curvas, matrices de confusión, Grad-CAM, ejemplos x|ŷ|y
```

---

## Fase 0 — Base Limpia y Configuración Común `[Local CPU]`

**Objetivo:** Evitar duplicación de constantes y dejar la estructura de carpetas lista.

- [ ] Crear carpetas base vacías (`app/`, `eval/`, `notebooks/`, `src/models/`, `src/train/`, `src/export/`, `docs/figuras/`).
- [ ] Crear `src/config.py` extrayendo `CHARSET`, `TEST_UNSEEN_CHARS`, `IMG_SIZE = 128`, `SEED = 42` y los mapeos `CHAR_TO_IDX` / `IDX_TO_CHAR`.
- [ ] Actualizar [src/data/01_generar_plantilla.py](../src/data/01_generar_plantilla.py) para importar `CHARSET` y `TEST_UNSEEN_CHARS` desde `src/config.py`.
- [ ] Crear `src/utils.py` con `seed_everything(seed=42)` (fija `random`, `numpy`, `torch`, `cudnn`).

### Checkpoint 0 (Prueba de humo)
```powershell
python -c "from src.config import CHARSET, TEST_UNSEEN_CHARS; assert len(CHARSET)==75 and len(TEST_UNSEEN_CHARS)==11; print('Config OK')"
```
> Hacer `git commit -m "chore: agregar configuración base, utilidades y estructura del proyecto"`.

---

## Fase 1 — Pipeline de Datos Independiente de Hojas Impresas `[Local CPU]`

**Objetivo:** Tener el dataset sintético listo, el dataset OOD listo y el extractor OpenCV probado antes de imprimir.

### 1.1 Dataset Sintético (`src/data/03_generar_sinteticos.py`)
- [ ] Descargar 15–20 Google Fonts manuscritas con licencia abierta (OFL/Apache, ej. *Caveat*, *Patrick Hand*, *Indie Flower*, *Shadows Into Light*, *Kalam*, *Architects Daughter*, *Handlee*, *Reenie Beanie*, etc.) y una fuente base limpia (ej. *Roboto* o *DejaVu Sans*) en `data/fonts/`.
- [ ] Documentar fuente + licencia de cada `.ttf` en `data/fonts/licenses.csv` (te servirá directo para `docs/datasheet.md` — Rúbrica B1).
- [ ] Renderizar pares `(x, y)` de `128×128` (fuente base vs. fuente manuscrita), aplicando centrados consistentes y ligeras variaciones controladas (escala ±5%, traslación ±3 px) para alcanzar **3,000–5,000 pares** en `data/synthetic_pairs/{train,val,test_unseen}/` + un archivo `metadata.csv`.
- [ ] Mantener los 11 símbolos de `TEST_UNSEEN_CHARS` estrictamente en `test_unseen/` (cero fuga).

### 1.2 Extractor OpenCV (`src/data/02_extraer_recortes.py`)
- [ ] Leer [data/templates/template_layout.json](../data/templates/template_layout.json).
- [ ] Detectar los 4 marcadores negros de esquina, calcular homografía (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`) para enderezar a `2550×3300`.
- [ ] Recortar cada celda (con margen interior para eliminar el borde gris del recuadro y la línea base tenue), binarizar/limpiar con umbral adaptativo, centrar el glifo por *bounding box* conservando proporción y redimensionar a `128×128`.
- [ ] Emparejar cada glifo `y` con su glifo base `x` renderizado con la misma fuente base de la Fase 1.1 y guardar en `data/real_pairs/{train,val,test_unseen}/`.

### 1.3 Dataset OOD (`src/data/04_generar_ood.py`)
- [ ] Generar al menos **60 imágenes fuera de distribución** en `data/ood/` (ruido gaussiano/sal-pimienta, garabatos aleatorios, símbolos inéditos `@ # $ % ñ ¿ ¡ *`, caracteres rotados ±45°, trazos cortados o borrosos, fondo con sombra fuerte) para cumplir Rúbrica B6 (≥50 imágenes).

### 1.4 `Dataset` y `DataLoader` unificados (`src/data/dataset.py`)
- [ ] Implementar `GlyphPairDataset` que lea `metadata.csv`, cargue `(x, y)` en escala de grises `1×128×128`, normalice a `[-1, 1]` y devuelva `(x, y, class_id, char)`.
- [ ] Implementar `DataLoader` con aumentos suaves solo en `train` (nunca flips horizontales/verticales porque cambian el carácter, ej. `b`↔`d`, `p`↔`q`, `6`↔`9`).

### Checkpoint 1 (Pruebas obligatorias antes de pasar a Fase 2)
1. **Inspección visual sintética:** generar una cuadrícula de 16 pares `(x | y)` aleatorios de `data/synthetic_pairs/train/` y verificar que ningún glifo esté cortado, invertido o vacío.
2. **Prueba de fuga de datos:** verificar por script que la intersección de caracteres entre `train` y `test_unseen` sea vacía (`set()`).
3. **Prueba simulada del extractor OpenCV:** tomar `plantilla_hoja_01.png`, dibujar letras sintéticas dentro de 6 celdas, rotarla 5° con perspectiva simulada y pasarla por `02_extraer_recortes.py`; confirmar que los 6 recortes salen alineados y limpios.
4. **Notebook EDA inicial (`notebooks/01_eda_datos.ipynb`):** graficar distribución de clases y verificar balance (Rúbrica B4).

> Hacer `git commit -m "feat(data): agregar generador sintético, extractor OpenCV, set OOD y Dataset"`.

---

## Fase 2 — Arquitecturas y Pruebas Unitarias de Tensores `[Local CPU]`

**Objetivo:** Confirmar que todos los modelos compilan, tienen los tamaños exactos y hacen *forward + backward* sin errores antes de gastar tiempo en Colab.

- [ ] `src/models/pix2pix.py`:
  - **Generador `UNetGenerator`:** entrada `[B, 1, 128, 128]`, 7 bloques encoder + 7 bloques decoder con *skip connections*, salida `Tanh` `[B, 1, 128, 128]`.
  - **Discriminador `PatchGANDiscriminator`:** entrada concatenada `[x, y]` (`[B, 2, 128, 128]`), salida mapa de logits `14×14`.
  - **Pérdidas:** `BCEWithLogitsLoss` con *one-sided label smoothing* (`0.9` real / `0.0` falso) + `L1Loss` con `λ = 100`.
- [ ] `src/models/judges.py`:
  - Adaptar `ResNet-18` y `EfficientNet-B0` a entrada de 1 canal (`1×128×128`) y salida de 75 clases (`len(CHARSET)`).
  - Exponer un método `extract_features(x)` (penúltima capa) para calcular **FID de dominio**.

### Checkpoint 2 (Smoke test en CPU)
- [ ] Correr un script de prueba que pase un tensor aleatorio `[4, 1, 128, 128]` por `UNetGenerator`, `PatchGANDiscriminator`, `ResNet-18` y `EfficientNet-B0`, verifique *shapes* exactos y ejecute 1 paso de `loss.backward()` sin NaNs.
- [ ] Correr **2 épocas de prueba** sobre un subconjunto de 32 imágenes sintéticas en local (CPU) para confirmar que el loop de entrenamiento guarda checkpoints y curvas de pérdida sin errores.

> Hacer `git commit -m "feat(models): implementar Pix2Pix U-Net/PatchGAN y jueces ResNet-18/EfficientNet-B0"`.

---

## Fase 3 — Entrenamiento Sintético (Jueces + Pretrain Pix2Pix) `[Colab GPU]`

**Objetivo:** Tener los evaluadores congelados y el modelo base preentrenado listos antes de que lleguen tus hojas reales.

- [ ] Crear `notebooks/02_colab_training.ipynb` que clone tu repo, instale dependencias faltantes y monte Google Drive para guardar `checkpoints/` y `results/`.
- [ ] **Entrenar Jueces (`src/train/train_judges.py`):**
  - Entrenar `ResNet-18` y `EfficientNet-B0` sobre los glifos manuscritos sintéticos (75 clases) con *early stopping* y curvas train/val guardadas en `docs/figuras/`.
  - Guardar pesos congelados (`checkpoints/judge_resnet18.pth`, `checkpoints/judge_efficientnet_b0.pth`).
- [ ] **Preentrenar Pix2Pix (`src/train/train_pix2pix.py --mode pretrain`):**
  - Entrenar sobre `data/synthetic_pairs/train/` (64 símbolos), monitoreando `val/` y guardando cuadrículas visuales `x | ŷ | y` cada N épocas.
  - Guardar `checkpoints/pix2pix_pretrained.pth`.

### Checkpoint 3 (Validación antes de seguir)
- [ ] Los jueces congelados alcanzan **alta exactitud (>90 % Macro-F1)** en validación sintética (si un juez no reconoce bien letras manuscritas, no sirve para evaluar a la GAN).
- [ ] Las imágenes `ŷ` del Pix2Pix preentrenado son visualmente legibles en `val/` y coherentes en `test_unseen/`.
- [ ] Guardar las curvas de entrenamiento/validación en `docs/figuras/` (Rúbrica D2).

> Descargar los pesos a `checkpoints/` (ignorado por Git) y hacer commit del código/figuras.

---

## Fase 4 — Optimización ONNX, Exportación `.ttf` y Demo en Tiempo Real `[Local CPU]`

**Objetivo:** Dejar funcionando toda la capa de despliegue **con el modelo preentrenado**, de modo que cuando tengas el modelo *fine-tuned* real solo cambies la ruta del archivo `.onnx`.

- [ ] **Exportación y Benchmark (`src/export/export_onnx.py`):**
  - Exportar el Generador (`G`) y el juez principal (`ResNet-18` o `EfficientNet-B0`) a **ONNX** (y opcionalmente TorchScript).
  - Medir latencia promedio (ms) y FPS sobre 100 inferencias en CPU antes (PyTorch) vs. después (ONNX Runtime) y guardar tabla en `docs/robustez_y_despliegue.md` (Rúbrica C5).
- [ ] **Demo en Tiempo Real (`app/demo_realtime.py`) (Rúbrica C1–C4):**
  - Ventana OpenCV triple: `Entrada x | Glifo generado ŷ | Predicción + Confianza + FPS`.
  - Modo dual con tecla `M`: **Canvas interactivo** (escribir/elegir carácter) y **Webcam ROI** (flujo continuo cuadro por cuadro, obligatorio por Sección 3 del PDF).
  - Umbral de confianza ajustable en vivo (`+` / `-`, default `0.70`) con alerta `OK` / `BAJA CONFIANZA (OOD)` y manejo seguro si no hay entrada o hay ruido.
  - Mostrar FPS y latencia (ms) en pantalla verificando **≥10 FPS (≤100 ms)**.
- [ ] **Exportación `.ttf` (`src/export/build_ttf.py`):**
  - Generar los 75 glifos (+ los inéditos `@ # $ % ñ Ñ ¿ ¡ *`) con `G`, vectorizar con `vtracer` y empaquetar en una fuente `.ttf` instalable con `fontTools`.

### Checkpoint 4 (Prueba de despliegue end-to-end)
- [ ] La demo arranca en **1 solo comando** (`python app/demo_realtime.py`), corre sin caerse al cambiar de modo (`M`), cambiar umbral (`+`/`-`) o tapar la cámara, y marca **≥10 FPS** en pantalla.
- [ ] Instalar el `.ttf` generado de prueba en Windows (o abrirlo en un visor de fuentes) y verificar que se puede escribir una frase legible.

> Hacer `git commit -m "feat(deploy): agregar exportación ONNX, demo OpenCV en tiempo real y constructor TTF"`.

---

## Fase 5 — Captura Real y Comparativa Experimental A vs. B `[Local + Colab GPU]`

> **Aquí es donde entran tus 7 hojas impresas** (`data/templates/plantilla_completa.pdf`).

- [ ] Imprimir y llenar las 7 hojas con pluma negra (0.5–0.7 mm), tomar fotos cenitales/escaneos y guardarlos en `data/raw_scans/`.
- [ ] Correr `python src/data/02_extraer_recortes.py` para generar `data/real_pairs/{train,val,test_unseen}/`.
- [ ] **Revisión manual obligatoria (Rúbrica B5):** Inspeccionar al menos **20–30 muestras reales**, anotar cuántas tuvieron error de recorte/etiqueta (%), corregirlas y registrar el hallazgo en `docs/datasheet.md`.
- [ ] **Experimento A — From Scratch (`[Colab GPU]`):** Entrenar Pix2Pix desde pesos aleatorios solo sobre `data/real_pairs/train/`.
- [ ] **Experimento B — Pretrain + Fine-Tuning (`[Colab GPU]`):** Cargar `checkpoints/pix2pix_pretrained.pth` y hacer *fine-tuning* sobre `data/real_pairs/train/`.
- [ ] Exportar el mejor generador final (Experimento B) a ONNX y generar tu `.ttf` personal definitivo.

### Checkpoint 5 (Validación de hipótesis)
- [ ] Comparar visualmente cuadrículas `x | ŷ_A (Scratch) | ŷ_B (Fine-Tune) | y_real` en `val` y `test_unseen` (Zero-Shot).
- [ ] Confirmar que `app/demo_realtime.py` y `build_ttf.py` funcionan con el modelo final.

---

## Fase 6 — Evaluación Cuantitativa, Interpretabilidad y Robustez `[Colab GPU / Local]`

**Objetivo:** Generar toda la evidencia cuantitativa y visual exigida por las secciones B, D y Avance A.

- [ ] **Métricas (`eval/evaluate_models.py`) (Rúbrica D1):**
  - Evaluar Exp A vs. Exp B con ambos jueces congelados (`ResNet-18` y `EfficientNet-B0`) en `val` y `test_unseen`: **Accuracy, Macro-F1, Matriz de Confusión y FID de dominio**.
- [ ] **Análisis de errores (Rúbrica D3):**
  - Seleccionar al menos **5 errores representativos**, agruparlos por tipo (ej. confusión de trazos similares `I/l/1`, signos delgados, artefactos en Zero-Shot) y documentar hipótesis de causa.
- [ ] **Interpretabilidad (`eval/gradcam_analysis.py` + `docs/interpretabilidad.md`):**
  - Generar mapas **Grad-CAM** sobre los jueces evaluando glifos reales vs. generados (`ŷ`) para explicar qué trazos determinan la legibilidad.
- [ ] **Notebooks de evaluación (`eval/robustez.ipynb` y `eval/auditoria_subgrupos.ipynb`):**
  - `robustez.ipynb` (Rúbrica B6): evaluar las ≥50 imágenes OOD (`data/ood/`), reportar tasas de rechazo según umbral y comparar contra `val`/`test_unseen`.
  - `auditoria_subgrupos.ipynb`: desglosar Macro-F1 por subgrupo (`A-Z`, `a-z`, `0-9`, signos) para detectar sesgos de clase.

### Checkpoint 6 (Cierre de experimentos)
- [ ] Todos los notebooks (`notebooks/` y `eval/`) ejecutados de principio a fin **sin errores** y con salidas guardadas (Rúbrica A6).
- [ ] Todas las gráficas exportadas en `docs/figuras/`.

---

## Fase 7 — Documentación Final, Video Plan B y Tag `v1.0-final` `[Local]`

**Objetivo:** Asegurar los 100 puntos de la lista de cotejo antes del **Jueves 26 de noviembre**.

- [ ] Congelar versiones exactas en `requirements.txt` (`pip freeze` limpio solo con las librerías del proyecto — Rúbrica A5).
- [ ] Subir datasets/pesos a Google Drive / GitHub Releases y dejar script o enlace de descarga en `data/README.md`.
- [ ] Completar documentos obligatorios:
  - [ ] `README.md` en el **orden exacto** de la Sección 4 del PDF: problema y usuario → integrantes (nombre + código) → dataset + enlace a datasheet → arquitectura con diagrama y justificación → tabla de resultados → instalación/reproducción en ≤5 comandos → demo en 1 comando → enlace a video de respaldo → limitaciones → uso de IA generativa y créditos.
  - [ ] `MODEL_CARD.md` (B7).
  - [ ] `docs/datasheet.md` (B1, B2, B5).
  - [ ] `docs/interpretabilidad.md`, `docs/robustez_y_despliegue.md` (C5), `docs/costos.md`, `docs/caso_sector.pdf`, `docs/informe_etico.pdf` (B7).
- [ ] Grabar **video de respaldo (1–2 min)** de la demo en vivo funcionando y enlazarlo en `README.md` (Plan B).
- [ ] Preparar presentación `docs/defensa.pdf` (12 min) y practicar la explicación línea por línea de bloques clave del código (Rúbrica E1–E3).
- [ ] Verificar que **todos los integrantes** tengan commits técnicos propios en GitHub (Rúbrica A3).
- [ ] Verificar que el repositorio sea **público**, crear y subir la etiqueta final:
  ```bash
  git tag -a v1.0-final -m "Entrega final proyecto CV"
  git push origin v1.0-final
  ```

---

## Resumen de Fechas Clave (según `proyecto_CV.pdf`)

| Fecha | Hito | Fases recomendadas completadas |
| :--- | :--- | :--- |
| **Martes 24 nov** | Avance A (interpretabilidad) + simulacro de demo | Fases 0–4 + primeros resultados de Fase 5/6 |
| **Jueves 26 nov** | Repositorio final público con tag `v1.0-final` | Fases 0–7 completas |
| **1–3 dic** | Defensa técnica con demo en vivo + `docs/defensa.pdf` | Ensayo + Plan B listo |
