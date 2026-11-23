# Datasheet del Conjunto de Datos — Generación de Tipografía Manuscrita (Pix2Pix cGAN)

Documento basado en el marco *Datasheets for Datasets* (Gebru et al., 2021) para cumplir los indicadores **B1, B2, B3, B4 y B5** de la rúbrica del proyecto.

---

## 1. Motivación y Selección Justificada (Rúbrica B1)

### 1.1 Problema y usuario real
- **Problema:** Crear una tipografía personal (`.ttf`) fiel a la caligrafía propia suele exigir dibujar manualmente cientos de glifos en editores vectoriales complejos (FontForge, Glyphs) o contratar digitalización artesanal.
- **Usuario real:** Estudiantes, diseñadores, ilustradores, docentes y creadores de contenido que desean digitalizar su propia letra manuscrita a partir de una captura rápida (~300 muestras) y generar tanto el alfabeto completo como símbolos no escritos (*Zero-Shot*).
- **Por qué el conjunto representa al problema y usuario real:** El sistema combina dos conjuntos de datos con contrato de imagen idéntico (`1×128×128`, escala de grises, fondo blanco `255` y trazo negro `0`):
  1. **Dataset sintético de preentrenamiento (`data/synthetic_pairs/`):** 3,825 pares `(x, y)` derivados de 17 fuentes manuscritas abiertas de Google Fonts emparejadas con una fuente base sans-serif (`Lato-Regular`). Permite que la red aprenda la correspondencia topológica entre glifos impresos y escritura humana antes de ver muestras del usuario.
  2. **Dataset propio del autor (`data/real_pairs/`):** 300 trazos manuscritos reales (75 símbolos × 4 repeticiones) capturados por el autor mediante stylus / tableta digitalizadora o plantilla impresa y normalizados automáticamente con OpenCV.
  3. **Dataset Fuera de Distribución (`data/ood/`):** 72 imágenes adversarias/degradadas en 6 categorías (ruido, garabatos, símbolos inéditos, rotación extrema, desenfoque/corte y sombra/bajo contraste) que reflejan las fallas reales de captura en cámara web y trazo libre (Rúbrica B6).

---

## 2. Composición, Fuentes y Licencias (Rúbrica B1 y B2)

### 2.1 Resumen cuantitativo de conjuntos de datos

| Conjunto | Origen | Licencia | Resolución y formato | Clases (`CHARSET`) | Total imágenes/pares | Splits |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Sintético (`synthetic_pairs`)** | Google Fonts (1 fuente base + 17 manuscritas) | SIL Open Font License 1.1 (OFL) | `1×128×128` PNG (8-bit gris) | 75 | **3,825 pares** | `train`: 2,688 / `val`: 576 / `test_unseen`: 561 |
| **Propio (`real_pairs`)** | Captura manuscrita propia del autor (stylus / plantilla) | Uso académico / CC0 (consentimiento explícito del autor) | `1×128×128` PNG (8-bit gris) | 75 | **300 pares** *(75 × 4 rep.)* | `train`: 192 / `val`: 64 / `test_unseen`: 44 |
| **Fuera de Distribución (`ood`)** | Generación programática (`04_generar_ood.py`) | MIT (repositorio propio) | `1×128×128` PNG (8-bit gris) | 6 categorías OOD | **72 imágenes** | Evaluación OOD (12 por categoría) |

### 2.2 Catálogo de fuentes tipográficas abiertas y licencias (`data/fonts/licenses.csv`)

Todas las fuentes utilizadas en el preentrenamiento provienen del repositorio oficial de Google Fonts (`github.com/google/fonts`) bajo licencia **SIL Open Font License 1.1 (OFL)** y verificaron soporte del 100 % de los 75 caracteres del `CHARSET` (`missing_chars = ""`):

| # | Nombre de la fuente | Archivo `.ttf` | Rol | Licencia |
| :-: | :--- | :--- | :--- | :--- |
| 0 | **Lato** | `Lato-Regular.ttf` | Base limpia (`x`) | SIL OFL 1.1 |
| 1 | Patrick Hand | `PatrickHand-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 2 | Indie Flower | `IndieFlower-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 3 | Shadows Into Light | `ShadowsIntoLight.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 4 | Kalam | `Kalam-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 5 | Architects Daughter | `ArchitectsDaughter-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 6 | Handlee | `Handlee-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 7 | Reenie Beanie | `ReenieBeanie.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 8 | Gloria Hallelujah | `GloriaHallelujah.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 9 | Neucha | `Neucha.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 10 | Gochi Hand | `GochiHand-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 11 | Courgette | `Courgette-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 12 | Pangolin | `Pangolin-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 13 | Sriracha | `Sriracha-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 14 | Short Stack | `ShortStack-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 15 | Dekko | `Dekko-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 16 | Mali | `Mali-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |
| 17 | Delius | `Delius-Regular.ttf` | Manuscrita (`y`) | SIL OFL 1.1 |

---

## 3. Distribución de Clases y Representación por Subgrupo (Rúbrica B2 y B4)

El vocabulario definido como fuente única de verdad en [`src/config.py`](../src/config.py) consta de **75 clases** repartidas en 4 subgrupos tipográficos, balanceadas con el mismo número de muestras por carácter dentro de cada split:

| Subgrupo | Símbolos incluidos | Clases totales | En `train` / `val` (64 vistas) | En `test_unseen` (11 Zero-Shot) | Pares sintéticos (`train` / `val` / `unseen`) | Pares reales objetivo (`train` / `val` / `unseen`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mayúsculas (`A–Z`)** | `A–Z` | 26 | 24 | 2 (`G, Q`) | 1,008 / 216 / 102 | 72 / 24 / 8 |
| **Minúsculas (`a–z`)** | `a–z` | 26 | 24 | 2 (`k, z`) | 1,008 / 216 / 102 | 72 / 24 / 8 |
| **Dígitos (`0–9`)** | `0–9` | 10 | 8 | 2 (`4, 8`) | 336 / 72 / 102 | 24 / 8 / 8 |
| **Signos básicos** | `. , ! ? ( ) + - = / : ; &` | 13 | 8 | 5 (`?, &, +, ;, )`) | 336 / 72 / 255 | 24 / 8 / 20 |
| **Total** | **75 símbolos** | **75** | **64** | **11** | **2,688 / 576 / 561 (3,825)** | **192 / 64 / 44 (300)** |

### Desbalance detectado y estrategia de atención (Rúbrica B4)
Ver análisis gráfico completo en [`notebooks/01_eda_datos.ipynb`](../notebooks/01_eda_datos.ipynb) y [`figuras/eda_distribucion_clases.png`](figuras/eda_distribucion_clases.png):
- **A nivel de clase individual:** El diseño es estrictamente balanceado (exactamente 42 muestras por carácter visto en `train` sintético, 9 en `val` sintético y 51 en `test_unseen` sintético; 3/1/4 repeticiones en datos reales).
- **A nivel de subgrupo y densidad de píxeles:** Existen menos clases de dígitos (10) y signos (13) que de letras (26+26), y los signos ocupan un área de tinta mucho menor (p. ej. `.`, `,`, `:`, `;`), lo que reduce su gradiente relativo bajo pérdida L1.
- **Estrategia de mitigación:**
  1. Aumentos afines suaves exclusivamente en `train` (`rotación ±4°`, `traslación ±4 %`, `escala 0.94–1.06`, relleno blanco `255`) en [`src/data/dataset.py`](../src/data/dataset.py). **Nunca se usan volteos horizontales ni verticales**, ya que alteran la identidad semántica de los glifos (`b ↔ d`, `p ↔ q`, `6 ↔ 9`).
  2. Evaluación desglosada obligatoria en [`eval/auditoria_subgrupos.ipynb`](../eval/auditoria_subgrupos.ipynb) usando **Macro-F1** para que cada subgrupo y clase pese por igual.

---

## 4. Proceso de Captura, Etiquetado y Prevención de Fuga de Datos (Rúbrica B2 y B3)

### 4.1 Proceso de generación y etiquetado
- **Sintético (`src/data/03_generar_sinteticos.py`):** Cada carácter del `CHARSET` se renderiza con `Lato-Regular` (`x`) y con cada una de las 17 fuentes manuscritas (`y`) aplicando 3 variaciones controladas de escala (±5 %) y traslación (±3 px), centrado por *bounding box* sobre lienzo `128×128`. Las etiquetas (`char`, `char_code`, `class_id`, `font_name`, `split`, `judge_split`) se registran automáticamente en `metadata.csv`.
- **Propio (`src/data/01_generar_plantilla.py`, `src/data/captura_stylus.py`, `src/data/02_extraer_recortes.py`):**
  - El archivo `data/templates/template_layout.json` fija las coordenadas exactas de las 300 celdas (7 hojas) y su etiqueta de carácter.
  - La captura puede realizarse con **stylus / tableta digitalizadora** (`python -m src.data.captura_stylus`, que guía símbolo por símbolo y compone `data/raw_scans/hoja_01..07.png`) o mediante **hojas impresas** llenadas a mano y escaneadas.
  - El extractor OpenCV detecta los 4 marcadores fiduciales negros de esquina, corrige la perspectiva por homografía a `2550×3300`, recorta cada celda descartando márgenes (`inner_margin = 14 px`), binariza con umbral adaptativo, centra el glifo por *bounding box* conservando relación de aspecto y lo empareja con su entrada base `x` de `Lato-Regular`.

### 4.2 Particiones sin fuga de información (Rúbrica B3)
Se aplican **dos niveles de aislamiento verificados por prueba automática** (`Checkpoint 1`):
1. **Aislamiento de caracteres Zero-Shot (Pix2Pix):** Los 11 símbolos de `TEST_UNSEEN_CHARS = {'G','Q','k','z','4','8','?','&','+',';',')'}` viven **exclusivamente en `test_unseen/`**. La intersección de caracteres `set(train) ∩ set(test_unseen) == ∅`. Nunca se usan para entrenar el generador, ajustar hiperparámetros ni *early stopping*.
2. **Aislamiento por fuente tipográfica (`train` vs. `val`):** En `data/synthetic_pairs/`, la partición entre `train` y `val` se realiza **por fuente completa** (`14 fuentes` exclusivas para `train` y `3 fuentes` nunca vistas para `val`, columna `split` y `judge_split`), evitando que la misma tipografía aparezca en entrenamiento y validación. En `data/real_pairs/`, las 4 repeticiones de cada símbolo visto se dividen en 3 para `train` y 1 para `val` (sin mezclar trazos repetidos).

---

## 5. Revisión Manual de Calidad de Etiquetas (Rúbrica B5)

### 5.1 Auditoría de calidad sobre el dataset sintético y extractor simulado (completada)
- **Muestras inspeccionadas manualmente:** 32 pares sintéticos (`16` de `train` en `figuras/checkpoint1_sinteticos.png` + `16` de `val`/`test_unseen`), 18 imágenes OOD (`figuras/checkpoint1_ood.png`) y 6 recortes del extractor OpenCV sobre plantilla rotada 5° con perspectiva simulada (`figuras/checkpoint1_opencv_simulado.png`).
- **Hallazgos y tasa de error:** **0 / 32 errores (0.0 %)** de etiqueta, inversión de color o recorte truncado tras corregir el margen interno de recorte (`inner_margin = 14 px`) y verificar soporte completo de los 75 símbolos en las 17 fuentes (3 fuentes candidatas con HTTP 404 —`ComingSoon`, `JustAnotherHand`, `Schoolbell`— fueron descartadas limpiamente durante la descarga).

### 5.2 Auditoría manual obligatoria sobre muestras reales del autor (pendiente de captura con stylus — Fase 5)
> **[PENDIENTE FASE 5]** Una vez capturadas las 300 muestras reales con `src/data/captura_stylus.py` y extraídas con `src/data/02_extraer_recortes.py`, inspeccionar al menos **30 muestras reales aleatorias** (≥20 exigidas por B5) y llenar la tabla siguiente antes del tag `v1.0-final`:

| Indicador de revisión manual (`data/real_pairs/`) | Valor registrado |
| :--- | :--- |
| Total de muestras reales inspeccionadas (`N ≥ 20`) | *Pendiente (Fase 5 — objetivo: 30 muestras)* |
| Muestras con error de etiqueta o confusión visual | *Pendiente* |
| Muestras con error de recorte / ruido residual / trazo cortado | *Pendiente* |
| **Porcentaje total de errores detectados (%)** | *Pendiente* |
| **Acciones de corrección aplicadas** | *Re-trazar celdas defectuosas en `captura_stylus.py` y re-ejecutar `02_extraer_recortes.py`* |

---

## 6. Privacidad, Consentimiento y Almacenamiento
- **Datos personales:** Los trazos manuscritos propios no contienen nombres, firmas legales ni rostros; aun así, la escritura manuscrita puede considerarse un rasgo biométrico estilístico. Las muestras reales pertenecen exclusivamente al autor del proyecto, quien otorga consentimiento expreso para su uso académico y entrenamiento del modelo.
- **Almacenamiento y reproducibilidad:** Los archivos pesados (`data/synthetic_pairs/`, `data/real_pairs/`, `data/raw_scans/`, `checkpoints/`) están excluidos del historial de Git en `.gitignore`. El dataset sintético y el set OOD se regeneran de forma 100 % determinista (`SEED = 42`) ejecutando `src/data/03_generar_sinteticos.py` y `src/data/04_generar_ood.py`; los pesos entrenados y el respaldo del dataset real se distribuyen mediante Google Drive / GitHub Releases (ver [`data/README.md`](../data/README.md)).
