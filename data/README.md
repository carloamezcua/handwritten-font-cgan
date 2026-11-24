# Datos y Protocolo de Captura

Esta carpeta almacena únicamente la documentación, licencias (`fonts/licenses.csv`) y el diseño de plantilla (`templates/template_layout.json`). **Los datasets pesados y escaneos crudos no se versionan en Git** (ver `.gitignore`).

## Descarga rápida de artefactos (GitHub Releases)

Para descargar los pesos entrenados (`.onnx` / `.pth`) y las 300 capturas manuscritas reales (`real_pairs.zip`) desde el Release `v1.0-final`:

```bash
# Solo modelos ONNX para ejecutar la demo en tiempo real:
python -m src.data.download_artifacts --only-demo

# Todos los pesos y el dataset real descomprimido en data/real_pairs/:
python -m src.data.download_artifacts --all
```

El dataset sintético (3,825 pares) y el set OOD (72 imágenes) se regeneran de forma 100 % determinista (`SEED = 42`):

```bash
python -m src.data.03_generar_sinteticos --download-fonts
python -m src.data.04_generar_ood
```

## Estructura esperada (local)
```text
data/
├── templates/        # Plantillas imprimibles y template_layout.json (01_generar_plantilla.py)
├── raw_scans/        # 7 hojas manuscritas capturadas con stylus o escaneadas
├── real_pairs/       # 300 pares (x, y) en 128x128 divididos en train/, val/, test_unseen/
├── synthetic_pairs/  # 3,825 pares (Lato-Regular, 17 Google Fonts manuscritas)
├── fonts/            # Fuentes .ttf base y manuscritas abiertas + licenses.csv
├── ood/              # 72 imágenes OOD programáticas en 6 categorías
└── ood_real/         # >= 50 capturas reales en cámara web (capturar_ood_real.py)
```

## Juego de Caracteres (75 símbolos × 4 repeticiones = 300 muestras)
- **Mayúsculas (26):** `A-Z`
- **Minúsculas (26):** `a-z`
- **Dígitos (10):** `0-9`
- **Signos básicos (13):** `. , ! ? ( ) + - = / : ; &`

### Partición estricta sin fuga de información
- `train`: 64 símbolos × 3 repeticiones (192 muestras reales).
- `val`: 64 símbolos × 1 repetición (64 muestras reales de caracteres vistos).
- `test_unseen`: 11 símbolos reservados (`G, Q, k, z, 4, 8, ?, &, +, ;, )`) × 4 repeticiones (44 muestras reales **nunca vistas** en entrenamiento para evaluación *Zero-Shot*).

## Reglas de Captura Manual
1. **Con stylus (recomendado):** ejecutar `python -m src.data.captura_stylus` para guiar símbolo por símbolo en navegador y generar automáticamente `data/raw_scans/hoja_01..07.png`.
2. **Con papel e impresora:** imprimir `data/templates/plantilla_completa.pdf`, escribir con pluma negra (0.5–0.7 mm), fotografiar con los **4 marcadores negros de esquina** visibles y guardar en `data/raw_scans/`.
3. Extraer y normalizar los 300 pares `128×128` con `python -m src.data.02_extraer_recortes`.
4. Capturar muestras OOD reales con cámara web usando `python -m src.data.capturar_ood_real`.
