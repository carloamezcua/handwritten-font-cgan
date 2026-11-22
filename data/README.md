# Datos y Protocolo de Captura

Esta carpeta almacena únicamente la documentación y scripts de captura. **Los datasets pesados y escaneos crudos no se versionan en Git** (ver `.gitignore`).

## Estructura esperada (local / Google Drive)
```text
data/
├── templates/        # Plantillas imprimibles y template_layout.json (generado por 01_generar_plantilla.py)
├── raw_scans/        # Fotos cenitales o escaneos de las 7 hojas llenas a mano
├── real_pairs/       # Pares (x, y) en 128x128 divididos en train/, val/, test_unseen/
├── synthetic_pairs/  # Pares (fuente_base, Google Fonts manuscritas) para preentrenamiento
├── fonts/            # Fuentes .ttf base y manuscritas abiertas
└── ood/              # >= 50 imágenes fuera de distribución (ruido, inclinación, trazos extremos)
```

## Juego de Caracteres (75 símbolos × 4 repeticiones = 300 muestras)
- **Mayúsculas (26):** `A-Z`
- **Minúsculas (26):** `a-z`
- **Dígitos (10):** `0-9`
- **Signos básicos (13):** `. , ! ? ( ) + - = / : ; &`

### Partición estricta
- `train`: 64 símbolos × 3 repeticiones (192 muestras reales).
- `val`: 64 símbolos × 1 repetición (64 muestras reales de caracteres vistos).
- `test_unseen`: 11 símbolos reservados (`G, Q, k, z, 4, 8, ?, &, +, ;, )`) × 4 repeticiones (44 muestras reales **nunca vistas** en entrenamiento para evaluación *Zero-Shot*).

## Reglas de Captura Manual
1. Imprimir `data/templates/plantilla_completa.pdf` en hoja blanca lisa (o abrir `plantilla_hoja_01..07.png` en tablet/iPad).
2. Escribir con **pluma negra de gel o estilógrafo (0.5–0.7 mm)** para mantener contraste alto y grosor uniforme.
3. Tomar foto cenital con iluminación uniforme (sin sombras duras) o escanear a 300 DPI, asegurando que los **4 marcadores negros de las esquinas** sean visibles.
4. Guardar las imágenes resultantes en `data/raw_scans/`.
