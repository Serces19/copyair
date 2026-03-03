# NAFNet y ConvNeXt - Guía de Uso

## Resumen

Se han implementado **NAFNet HD** y **ConvNeXt** optimizados para **few-shot learning** con pocas imágenes (5-15) de alta resolución variable.

## Arquitecturas Implementadas

### 1. **NAFNet HD** (Nonlinear Activation Free Network)
**Paper**: "Simple Baselines for Image Restoration" (ECCV 2022)

**Características:**
- ✅ Sin activaciones no lineales tradicionales (ReLU, GELU)
- ✅ SimpleGate: activación eficiente basada en multiplicación
- ✅ Simplified Channel Attention (SCA)
- ✅ **Muy eficiente computacionalmente** (8.4% del costo de SOTA)
- ✅ **Excelente para pocas imágenes**
- ✅ Maneja alta resolución nativamente

**Tamaños disponibles:**
| Tamaño | Parámetros | Recomendado para | Dropout |
|--------|-----------|------------------|---------|
| `small` | ~15M | 5-8 imágenes | 0.10 |
| `base` | ~28M | 8-12 imágenes | 0.05 |
| `large` | ~45M | >12 imágenes | 0.00 |

---

### 2. **ConvNeXt U-Net**
**Paper**: "A ConvNet for the 2020s" (CVPR 2022)

**Características:**
- ✅ CNN moderna inspirada en Vision Transformers
- ✅ Depthwise convolutions grandes (7x7)
- ✅ LayerNorm + GELU
- ✅ Inverted bottleneck design
- ✅ **Excelente para few-shot learning**
- ✅ **Escalable a alta resolución**

**Tamaños disponibles:**
| Tamaño | Parámetros | Recomendado para | Drop Path |
|--------|-----------|------------------|-----------|
| `nano` | ~15M | 5-8 imágenes | 0.05 |
| `tiny` | ~28M | 8-12 imágenes | 0.10 |
| `small` | ~50M | 10-15 imágenes | 0.15 |
| `base` | ~89M | >15 imágenes | 0.20 |

---

## Configuración en params.yaml

### NAFNet Small (5-8 imágenes)
```yaml
model:
  architecture: "nafnet"
  size: "small"
  in_channels: 3
  out_channels: 3
```

### NAFNet Base (8-12 imágenes) - **RECOMENDADO**
```yaml
model:
  architecture: "nafnet"
  size: "base"
  in_channels: 3
  out_channels: 3
```

### ConvNeXt Nano (5-8 imágenes)
```yaml
model:
  architecture: "convnext"
  size: "nano"
  in_channels: 3
  out_channels: 3
  drop_path_rate: 0.05  # Regularización
```

### ConvNeXt Tiny (8-12 imágenes) - **RECOMENDADO**
```yaml
model:
  architecture: "convnext"
  size: "tiny"
  in_channels: 3
  out_channels: 3
  drop_path_rate: 0.10
```

---

## Comparación: NAFNet vs ConvNeXt vs U-Net

| Característica | NAFNet | ConvNeXt | U-Net |
|---------------|--------|----------|-------|
| **Parámetros** | 15-45M | 15-89M | 30M |
| **Velocidad** | ⚡⚡⚡ Muy rápida | ⚡⚡ Rápida | ⚡⚡⚡ Muy rápida |
| **Memoria** | 💾 Baja | 💾💾 Media | 💾 Baja |
| **Few-shot** | ⭐⭐⭐ Excelente | ⭐⭐⭐ Excelente | ⭐⭐ Buena |
| **Alta Resolución** | ⭐⭐⭐ Excelente | ⭐⭐⭐ Excelente | ⭐⭐ Buena |
| **Calidad** | ⭐⭐⭐ SOTA | ⭐⭐⭐ SOTA | ⭐⭐ Buena |
| **Overfitting** | ⭐⭐⭐ Resistente | ⭐⭐⭐ Resistente | ⭐ Propenso |

---

## Recomendaciones por Caso de Uso

### 📸 5-8 Imágenes de Alta Resolución
**Opción 1 (Más rápida):**
```yaml
model:
  architecture: "nafnet"
  size: "small"
```

**Opción 2 (Mejor calidad):**
```yaml
model:
  architecture: "convnext"
  size: "nano"
  drop_path_rate: 0.10
```

**Training:**
```yaml
training:
  epochs: 1000  # Más épocas para pocas imágenes
  batch_size: 4  # Batch pequeño
  learning_rate: 5e-4  # LR bajo
  scheduler:
    type: "cosine"
    params:
      T_max: 1000
      eta_min: 1e-6
```

---

### 📸 8-12 Imágenes de Alta Resolución (TU CASO)
**Opción 1 (Recomendada - Balance):**
```yaml
model:
  architecture: "nafnet"
  size: "base"
```

**Opción 2 (Mejor para detalles finos):**
```yaml
model:
  architecture: "convnext"
  size: "tiny"
  drop_path_rate: 0.10
```

**Training:**
```yaml
training:
  epochs: 800
  batch_size: 6
  learning_rate: 1e-3
  scheduler:
    type: "cosine"
    params:
      T_max: 800
      eta_min: 1e-6
```

---

### 📸 12-15 Imágenes de Alta Resolución
**Opción 1:**
```yaml
model:
  architecture: "nafnet"
  size: "base"
```

**Opción 2:**
```yaml
model:
  architecture: "convnext"
  size: "small"
  drop_path_rate: 0.15
```

**Training:**
```yaml
training:
  epochs: 600
  batch_size: 8
  learning_rate: 1e-3
  scheduler:
    type: "plateau"
    params:
      patience: 20
      factor: 0.5
```

---

## Ventajas Específicas

### NAFNet HD
1. **Eficiencia extrema**: 8.4% del costo computacional de métodos SOTA
2. **Sin activaciones no lineales**: Menos parámetros, más rápido
3. **SimpleGate**: Activación aprendible sin overhead
4. **Mejor para pocas imágenes**: Menos propenso a overfitting
5. **Alta resolución nativa**: No necesita resize agresivo

### ConvNeXt U-Net
1. **Arquitectura moderna**: Incorpora mejores prácticas de ViTs
2. **Depthwise 7x7**: Captura contexto amplio
3. **Inverted bottleneck**: Mejor flujo de información
4. **Drop Path**: Regularización efectiva para few-shot
5. **Escalabilidad**: Fácil ajustar tamaño según datos

---

## Técnicas de Regularización para Few-Shot

### 1. **Dropout / DropPath**
```yaml
model:
  architecture: "convnext"
  drop_path_rate: 0.10  # 0.05-0.20 según cantidad de imágenes
```

### 2. **Data Augmentation Agresiva**
```yaml
augmentation:
  enabled: true
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5
  rotation_limit: 30  # Más rotación
  # Agregar más augmentations
```

### 3. **Learning Rate Bajo**
```yaml
training:
  learning_rate: 5e-4  # Más bajo que usual
  weight_decay: 1e-4
```

### 4. **Más Épocas**
```yaml
training:
  epochs: 1000  # 2-3x más que con muchas imágenes
```

### 5. **Early Stopping Paciente**
```yaml
training:
  early_stopping_patience: 500  # Muy paciente
```

---

## Entrenamiento

```bash
# Editar configs/params.yaml con la configuración deseada

# Entrenar
python scripts/train.py --config configs/params.yaml

# Inferencia
python scripts/predict.py --model models/best_model.pth --video input.mp4 --output output.mp4 --native-resolution
```

---

## Monitoreo con MLflow

Ambas arquitecturas loggean automáticamente:
- **Parámetros del modelo**: arquitectura, size, drop_path_rate
- **Métricas**: train_loss, val_loss, PSNR
- **Learning rate**: train/lr
- **Imágenes de validación**: cada 100 épocas

```bash
mlflow ui
# Abrir http://localhost:5000
```

---

## Troubleshooting

### Problema: Out of Memory (OOM)
**Solución:**
1. Reducir `batch_size`
2. Usar tamaño más pequeño (`nano` o `small`)
3. Reducir `img_size` en augmentations
4. Usar `--native-resolution` solo si es necesario

### Problema: Overfitting rápido
**Solución:**
1. Aumentar `drop_path_rate` (0.15-0.20)
2. Más data augmentation
3. Reducir tamaño del modelo
4. LR más bajo
5. Más weight decay

### Problema: Underfitting
**Solución:**
1. Modelo más grande (`base` o `large`)
2. Más épocas
3. LR más alto
4. Menos regularización

---

## Benchmarks Esperados

### NAFNet Base (8-12 imágenes)
- **PSNR**: 35-40 dB (después de 500 épocas)
- **Tiempo/época**: ~2-3 min (GPU RTX 3090, 256x256)
- **Memoria**: ~4-6 GB VRAM

### ConvNeXt Tiny (8-12 imágenes)
- **PSNR**: 36-41 dB (después de 500 épocas)
- **Tiempo/época**: ~3-4 min (GPU RTX 3090, 256x256)
- **Memoria**: ~6-8 GB VRAM

---

## Referencias

- **NAFNet**: [Simple Baselines for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)
- **ConvNeXt**: [A ConvNet for the 2020s (CVPR 2022)](https://arxiv.org/abs/2201.03545)
- **Few-Shot Learning**: [ConvNeXt-ECA for Few-Shot Classification](https://ieeexplore.ieee.org/document/10222084)

---

## Próximos Pasos

1. ✅ Implementar NAFNet y ConvNeXt
2. ✅ Configurar para few-shot learning
3. ⏳ Entrenar con tus 5-15 imágenes
4. ⏳ Comparar resultados entre arquitecturas
5. ⏳ Fine-tuning de hiperparámetros
6. ⏳ Evaluar en video de alta resolución
