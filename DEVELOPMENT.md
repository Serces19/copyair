# CopyAir - Guía de Desarrollo

## Arquitectura del Código

### Módulo `src/data/`
**Responsabilidad**: Carga y procesamiento de datos

- `dataset.py`:
  - `PairedImageDataset`: Carga pares de imágenes durante entrenamiento
  - `VideoFrameDataset`: Carga frames individuales para inferencia
  
- `augmentations.py`:
  - `get_transforms()`: Crea transformaciones con augmentación
  - `get_inference_transforms()`: Transformaciones sin augmentación (normalización solo)

### Módulo `src/models/`
**Responsabilidad**: Arquitecturas y pérdidas

- `unet.py`:
  - `ConvBlock`: Convolution + BatchNorm + ReLU
  - `DownBlock`: 2x ConvBlock + MaxPool
  - `UpBlock`: Deconvolution + Skip + 2x ConvBlock
  - `UNet`: Arquitectura completa (4 levels encoder/decoder)

- `losses.py`:
  - `L1Loss`: Simple L1 (MAE)
  - `SSIMLoss`: Similitud estructural
  - `PerceptualLoss`: Características perceptuales
  - `HybridLoss`: Combinación ponderada (λ1*L1 + λ2*SSIM + λ3*Perceptual)

### Módulo `src/training/`
**Responsabilidad**: Entrenamiento e inferencia

- `train.py`:
  - `train_epoch()`: Loop de una época
  - `validate()`: Validación y cálculo de PSNR
  - `compute_metrics()`: MSE, PSNR, SSIM

- `inference.py`:
  - `predict_frame()`: Predice un frame individual
  - `predict_on_video()`: Procesa video completo y guarda salida
  - `extract_frames_from_video()`: Extrae frames de video

---

## Flujo de Trabajo

### Entrenamiento
```
1. Cargar configuración (params.yaml)
2. Preparar dataset (input/, ground_truth/)
3. Crear modelo U-Net
4. Configurar optimizador (Adam) + scheduler (Cosine)
5. Loop por épocas:
   - train_epoch(): Forward, pérdida, backward
   - validate(): Calcular PSNR y val_loss
   - Guardar checkpoint si mejora
   - Early stopping si no progresa
6. Guardar best_model.pth
```

### Inferencia
```
1. Cargar modelo preentrenado
2. Para cada frame del video:
   - Normalizar [0, 1]
   - Aplicar augmentaciones
   - Forward pass en GPU/CPU
   - Denormalizar y guardar frame
3. Reconstruir video desde frames
```

---

## Modificaciones Comunes

### Cambiar Arquitectura del Modelo

En `src/models/unet.py`:

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        # Aumentar base_channels para más capacidad
        # Añadir más bloques DownBlock/UpBlock para profundidad
```

### Cambiar Función de Pérdida

En `configs/params.yaml`:

```yaml
loss:
  type: "l1"  # O "ssim", "hybrid"
  lambda_l1: 1.0
```

### Añadir Augmentaciones

En `src/data/augmentations.py`:

```python
def get_transforms(...):
    return A.Compose([
        # Añadir aquí nuevas transformaciones
        A.GaussianBlur(blur_limit=5, p=0.3),  # Ejemplo
        ...
    ])
```

---

## Mejores Prácticas

### 1. Versionado de Modelos
```bash
# Guardar con timestamp
torch.save(model.state_dict(), f"models/checkpoint_{epoch}_{timestamp}.pth")
```

### 2. Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Mensaje importante")
```

### 3. Reproducibilidad
```python
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
```

### 4. Profiling
```bash
# Ver memoria GPU
nvidia-smi

# Profiling en PyTorch
from torch.profiler import profile, record_function
with profile(...) as prof:
    model(x)
```

---

## Debugging

### Problema: Out of Memory (OOM)
- Reducir `batch_size` en params.yaml
- Reducir `base_channels` en modelo
- Usar `torch.cuda.empty_cache()`

### Problema: Modelo no converge
- Aumentar `learning_rate`
- Usar `warmup_epochs`
- Verificar datos (están bien organizados?)
- Aumentar épocas de entrenamiento

### Problema: Frames borrosos
- Aumentar `lambda_ssim` en la pérdida
- Añadir más pares de entrenamiento
- Aumentar épocas

---

## Próximos Pasos

1. **MLOps**: Integrar MLflow para tracked experiments
2. **DVC**: Versionar datos y modelos
3. **CI/CD**: GitHub Actions para tests automáticos
4. **API**: FastAPI para servir modelo
5. **Optimización**: Quantization, pruning, ONNX export

---

## Recursos

- [PyTorch Optimization Guide](https://pytorch.org/docs/stable/optim.html)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Albumentations Documentation](https://albumentations.ai/docs/)

