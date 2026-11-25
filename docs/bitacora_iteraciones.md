# Bitácora de Iteraciones, Fallos y Decisiones Técnicas

Este documento registra la evolución real del proyecto: qué funcionó en los primeros intentos, qué problemas concretos aparecieron al probar con datos propios y en la demo en vivo, y qué decisiones de arquitectura e ingeniería se tomaron para resolverlos.

---

## Iteración 1 — Baseline `Pix2Pix` y choque de dominio sobre trazos reales

- **Hipótesis inicial:** Preentrenar una cGAN `Pix2Pix` (`UNetGenerator` + `PatchGANDiscriminator`) con 3,825 pares sintéticos de 17 Google Fonts manuscritas (`Lato-Regular -> manuscrita`) y luego hacer *fine-tuning* sobre 300 muestras propias capturadas con stylus (`data/real_pairs/`).
- **Qué falló (`notebooks/03_colab_finetuning.ipynb`):**
  - Al comparar `pretrained`, `scratch` y `finetune` sobre `data/real_pairs/`, **`finetune` bajó su desempeño** en `val` (`62.50 %` Acc con `ResNet-18`, frente a `75.00 %` de `scratch` y `71.88 %` de `pretrained`) y **colapsó en `test_unseen` (Zero-Shot)** hasta `9.09 %` (`ResNet-18`).
  - Además, en pruebas de ablación reiniciando el discriminador y seleccionando el mejor checkpoint únicamente por `val_l1` mínimo, el generador colapsó a salidas casi en blanco (`val_l1 = 0.0455`) porque los trazos de stylus tienen muy poca tinta (`2.57 %` promedio) y producir fondo blanco minimizaba artificialmente el error L1.
- **Diagnóstico:** En `Pix2Pix` estándar, el estilo tipográfico (grosor de trazo, curvatura, densidad de tinta) está mezclado dentro de todos los pesos de la U-Net. El choque de grosor entre las 17 fuentes sintéticas (trazo grueso) y el stylus real (trazo fino) hacía que ajustar la red completa borrara su capacidad de generalizar a símbolos no vistos.

---

## Iteración 2 — Cambio de arquitectura a `StyleUNet` (`StyleEncoder` + `AdaIN`)

- **Decisión técnica (`src/models/style_unet.py`, `notebooks/04_colab_style_encoder.ipynb`):**
  - Desacoplar explícitamente **contenido** (la letra base `Lato`) y **estilo** ($K=4$ glifos de referencia del mismo escritor) mediante un `StyleEncoder` que produce un vector `z_style` (`128-d`) inyectado en el decoder vía capas **`AdaIN2d`**.
  - Eliminar `InstanceNorm` dentro de `StyleEncoder` (siguiendo MUNIT/FUNIT) para no borrar la densidad de tinta ni el grosor del trazo fino.
  - Agregar aumento morfológico emparejado de grosor en `pretrain` y penalización `InkDiff` en la pérdida y selección de checkpoint para evitar el colapso a salidas en blanco.
- **Resultado verificado sobre `data/real_pairs/` (`ResNet-18`):**
  - **Sin *fine-tuning* (*Few-Shot*, 0 épocas, 4 referencias reales):** `84.38 %` Acc en `val` y `54.55 %` en Zero-Shot (`test_unseen`).
  - **Con *fine-tuning* ligero (mejor época 4, `StyleEncoder` congelado):** igualó el techo humano en `val` (**`89.06 %` Acc / `85.94 %` Macro-F1**) y elevó Zero-Shot de `9.09–27.27 %` hasta **`72.73 %`** (reduciendo FID de dominio de `499.83` a **`302.42`**).

---

## Iteración 3 — Centralización de hiperparámetros (`src/config.py`)

- **Problema detectado:** Los hiperparámetros de entrenamiento (`epochs`, `batch_size`, `lr`, `betas`, `lambda_l1`, `patience`, pesos de pérdida) estaban repartidos dentro de cada script (`train_judges.py`, `train_pix2pix.py`, `train_style_unet.py`), dificultando inspeccionar la configuración completa o ejecutar experimentos controlados.
- **Solución aplicada (`src/config.py`):** Se consolidaron en tres bloques centralizados (`JUDGES_CONFIG`, `PIX2PIX_CONFIG` y `STYLE_UNET_CONFIG`) importados por los scripts de entrenamiento como valores por defecto, conservando el soporte completo de argumentos CLI (`--lr`, `--epochs`, etc.) para mantener 100 % de compatibilidad con los notebooks de Google Colab.

---

## Iteración 4 — Diagnóstico y correcciones de la demo en tiempo real (`app/demo_realtime.py`)

- **Problema detectado al probar en vivo:** Al usar la demo en modo **dibujo libre con mouse** y **webcam**, la clasificación del juez `ResNet-18` fallaba con frecuencia incluso en letras claramente legibles.
- **Causas raíz identificadas en el código:**
  1. **Desajuste de fuente base (`src/data/glyph_render.py`):** `get_default_base_font_path()` buscaba `Roboto-Regular.ttf` y caía en `Arial` del sistema, ignorando `data/fonts/base/Lato-Regular.ttf` (la fuente exacta con la que se entrenaron los modelos).
  2. **Dibujo libre sin normalización espacial:** A diferencia de la webcam, el trazo dibujado con mouse entraba crudo en `128×128` sin recortar su *bounding box*, sin escalar al tamaño esperado ni alinear a la línea base (`0.72`).
  3. **Doble degradación en entradas manuscritas (`x -> Pix2Pix -> Juez`):** La demo enviaba siempre la salida `y_hat` de `Pix2Pix` al juez `ResNet-18`. Esto es correcto en modo teclado (`Lato` limpio → generar manuscrito → auditar legibilidad), pero en **mouse y webcam** el usuario ya entrega un trazo manuscrito: pasarlo por `Pix2Pix` (entrenado solo con entrada `Lato` limpia) distorsionaba el trazo antes de clasificarlo.
  4. **Umbral binario duro en webcam (`resized < 170`):** Al reescalar un trazo fino capturado en papel, la binarización dura podía adelgazar o romper el carácter.
- **Correcciones aplicadas (`src/data/glyph_render.py` y `app/demo_realtime.py`):**
  1. Priorización de `data/fonts/base/Lato-Regular.ttf` en `get_default_base_font_path()`.
  2. Nueva función `_normalize_freehand_canvas()` que recorta, escala (`~56 %`), suaviza y alinea a la línea base (`0.72`) el trazo de mouse, mostrando una miniatura del glifo normalizado en el panel izquierdo.
  3. Modo de evaluación automático del juez (`AUTO`): clasifica `y_hat` en modo teclado (`Lato`) y clasifica directamente `x` normalizado en **dibujo libre y webcam**, manteniendo visible la estilización `y_hat` en el panel central y permitiendo alternar en vivo con la tecla **`Enter`**.
  4. Contraste suave con antialiasing y dilatación preventiva para trazos muy finos en `_preprocess_webcam_roi()`.
