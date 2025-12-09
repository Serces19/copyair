# Guía de Inferencia 🚀

Este proyecto ofrece múltiples estrategias de inferencia para adaptarse a diferentes necesidades de calidad y recursos.

## Comando Básico

```bash
python scripts/predict.py --model models/best_model.pth --video input.mp4 --output output.mp4
```

## Opciones Avanzadas

### 1. Inferencia por Tiles (Recomendado para HD/4K) 🧩
Ideal para imágenes de alta resolución (1080p, 4K) donde la memoria GPU es limitada o se entrenó con parches pequeños.

- **Cómo funciona**: Divide la imagen en parches solapados, procesa cada uno y los mezcla suavemente.
- **Ventajas**: 
  - Calidad superior en detalles finos.
  - Evita errores de memoria (OOM).
  - Sin líneas de corte visibles gracias al blending Gaussiano.

```bash
python scripts/predict.py \
    --model models/best_model.pth \
    --video input_4k.mp4 \
    --output output_4k.mp4 \
    --tiled \
    --tile-size 512 \
    --overlap 64
```

### 2. Resolución Nativa vs Resize 📏

- **--native-resolution**: Procesa el video en su tamaño original. Si no se usa `--tiled`, la imagen entera se pasa a la red (cuidado con la VRAM).
- **(Por defecto)**: Redimensiona la imagen al tamaño definido en `params.yaml` (ej. 256x256).

```bash
# Procesar en 1080p real (usando tiles para seguridad)
python scripts/predict.py ... --native-resolution --tiled
```

### 3. Calidad de Video (Lossless) 💎

- **--lossless**: Usa CRF 0 y espacio de color BT.709 para máxima fidelidad.
- **--backend ffmpeg**: Usa FFmpeg directamente para mejor compresión y calidad que OpenCV.

```bash
python scripts/predict.py ... --lossless --backend ffmpeg
```

### 4. Extracción de Frames 🎞️

Si prefieres trabajar con secuencias de imágenes:

```bash
# Extraer frames
python scripts/predict.py --video input.mp4 --extract-frames

# Inferencia sobre directorio de imágenes
python scripts/predict.py --video data/01_raw/input --output ../output_inference
python scripts/predict.py --video data/01_raw/input --output ./output_inference --model models/best_model_unet.pth --native-resolution
```

## Resumen de Argumentos

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--tiled` | Activa inferencia por tiles | False |
| `--tile-size` | Tamaño del tile (debe ser par) | 512 |
| `--overlap` | Solapamiento entre tiles | 64 |
| `--native-resolution` | Mantiene resolución original | False |
| `--lossless` | Modo sin pérdidas (CRF 0) | False |
| `--backend` | Motor de video (`ffmpeg` o `opencv`) | `ffmpeg` |
