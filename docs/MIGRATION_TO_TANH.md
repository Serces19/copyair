# Migración a Normalización [-1, 1]

## Resumen de Cambios

Se ha migrado todo el pipeline de normalización de ImageNet stats a normalización estándar [-1, 1], que es el estándar en image-to-image translation (Pix2Pix, CycleGAN, StyleGAN).

## Razones del Cambio

1. **No usamos Transfer Learning**: ImageNet stats son para modelos pre-entrenados
2. **Mejor convergencia**: Rango simétrico centrado en 0
3. **Estándar de la industria**: Todos los modelos I2I modernos usan [-1, 1]
4. **Preparación para HDR**: Más fácil extender a rangos mayores
5. **Consistencia**: Input y GT en el mismo rango

## Archivos Modificados

### 1. `src/models/unet.py`
- **Línea 123**: Cambio de `Sigmoid()` → `Tanh()`
- **Línea 143**: Output ahora en rango [-1, 1]

### 2. `src/data/augmentations.py`
- **get_transforms()**: 
  - Input: Normalización a [-1, 1] usando `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`
  - GT: Normalización a [-1, 1] (antes estaba en [0, 1])
- **get_inference_transforms()**: Normalización a [-1, 1]

### 3. `src/training/inference.py`
- **predict_frame()**: Desnormalización de [-1, 1] → [0, 255]
- **predict_on_video()**: 
  - Normalización manual a [-1, 1] en modo native_resolution
  - Desnormalización de [-1, 1] → [0, 255]
  - **BONUS**: Agregado soporte FFmpeg con color space correcto (fix del gamma)

### 4. `src/models/losses.py`
- **LaplacianPyramidLoss**: Comentario actualizado
- **PSNRLoss**: `max_val` cambiado de 1.0 → 2.0

### 5. `src/training/train.py`
- **validate()**: PSNR con `max_val=2.0`
- **compute_metrics()**: PSNR con `max_val=2.0`

### 6. `src/data/dataset.py`
- **PairedImageDataset._to_tensor()**: Fallback normaliza a [-1, 1]
- **VideoFrameDataset**: Fallback normaliza a [-1, 1]

## Fórmulas de Conversión

### Normalización (Input)
```python
# De [0, 255] a [-1, 1]
normalized = (pixel / 255.0) * 2.0 - 1.0

# Equivalente con Albumentations
A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0)
```

### Desnormalización (Output)
```python
# De [-1, 1] a [0, 255]
pixel = ((normalized + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
```

## Impacto

⚠️ **IMPORTANTE**: El modelo debe ser **reentrenado** con estos cambios porque:
1. La activación final cambió (Sigmoid → Tanh)
2. El rango de salida cambió ([0, 1] → [-1, 1])
3. El GT ahora está normalizado

## Beneficios Esperados

1. ✅ **Mejor convergencia**: Gradientes más estables
2. ✅ **Mejor calidad**: Aprovecha todo el rango de Tanh
3. ✅ **Más consistente**: Input y GT en mismo espacio
4. ✅ **Estándar**: Alineado con papers SOTA
5. ✅ **Extensible**: Fácil migrar a HDR después

## Fix Adicional: Color Space

También se arregló el problema del "gamma bajado" reemplazando `cv2.VideoWriter` con FFmpeg que especifica:
- `-color_range pc`: Full range (0-255)
- `-colorspace bt709`: HD color space
- `-color_trc iec61966-2-1`: sRGB gamma

## Testing

Después de reentrenar, ejecutar:
```bash
python scripts/predict.py --model models/best_model.pth --video data/01_raw/input.mp4 --output output.mp4 --native-resolution
```

Los colores deberían verse correctos sin el efecto de gamma bajado.
