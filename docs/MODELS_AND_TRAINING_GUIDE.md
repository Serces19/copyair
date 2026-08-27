# 🧠 Guía Maestra de Arquitecturas SOTA, Modelos y Entrenamiento en CopyAir

Esta guía documenta el catálogo oficial y consolidado de modelos de **CopyAir**, sus fundamentos matemáticos y papers (2022-2025), cómo están configurados para **Image-to-Image Translation / VFX / De-Aging / Few-Shot**, y cómo entrenar e inferir cada uno.

---

## 1. Catálogo Oficial de Arquitecturas

CopyAir implementa un sistema modular unificado coordinado por `src/models/factory.py`. Todos los modelos han sido auditados, refactorizados y adaptados para alta fidelidad visual y estabilidad en GPU:

```
src/models/
├── nafnet.py        -> NAFNet HD (Nonlinear Activation Free Network - SOTA Few-Shot)
├── restormer.py     -> Restormer (MDTA Channel-Attention + GDFN Transformer)
├── scope_unet.py    -> ScopeUNet / GatedUNet (PixelUnshuffle + ICNR PixelShuffle)
├── mambair.py       -> MambaIR / VMamba (Real 2D Cross-Scan State Space Model)
├── convnext.py      -> ConvNeXt-V2 (Pretrained 22K + LayerNorm/GRN Decoder)
├── residual_unet.py -> Residual U-Net (Pre-Act Mish + True Identity Shortcuts)
├── modern_unet.py   -> Modern U-Net (FiLM Conditioning + Multi-head Attention)
├── smartunet.py     -> Smart U-Net (Attention Gates + Frequency Filtering)
└── basic_unet.py    -> U-Net Estándar de Referencia
```

---

## 2. Comparativa Técnica y Recomendaciones por Caso de Uso

| Arquitectura (`architecture`) | Variantes / Tamaños | Papers Clave (2022-2025) | Fortalezas para VFX / De-aging | VRAM Requerida | Velocidad | Resistencia Overfitting (Few-Shot: 5-20 frames) |
|---|---|---|---|---|---|---|
| **`nafnet`** | `small`, `base`, `large` | *NAFNet (ECCV 2022)* | **Recomendado SOTA General.** Sin activaciones no lineales complejas (SimpleGate), LayerScale ($\beta, \gamma$), ultrarrápido y preserva texturas finas. | 4-6 GB | ⚡⚡⚡ Ultrarrápida | ⭐⭐⭐ Excelente |
| **`restormer`** | `tiny`, `small`, `base` | *Restormer (CVPR 2022)*, *PromptIR (NeurIPS 2023)* | **Recomendado SOTA Transformer.** Atención transpuesta multi-cabezal (MDTA) con complejidad lineal $O(HW)$ a través de canales + GDFN. Modela contexto global sin cuadrículas fijas. | 4-6 GB | ⚡⚡ Rápida | ⭐⭐⭐ Excelente |
| **`scope_unet`** / **`gated_unet`** | `small`, `base` | *Gated Restoration (2023)* | Downsampling sin pérdida con `PixelUnshuffle` + Upsampling libre de checkerboard con `ICNR PixelShuffle` + SimpleGate. Ideal para secuencias de video HD/4K. | 4-6 GB | ⚡⚡⚡ Ultrarrápida | ⭐⭐⭐ Excelente |
| **`convnext`** | `nano`, `tiny`, `small`, `base` | *ConvNeXt V2 (CVPR 2023)* | Convoluciones 7x7 depthwise con backbone ImageNet-22K (`timm`). Decoder con GroupNorm y Global Response Normalization (GRN) para evitar saturación de textura. | 6-8 GB | ⚡⚡ Rápida | ⭐⭐⭐ Excelente |
| **`mambair`** / **`vmamba`** | `tiny`, `base`, `large` | *VMamba (2024)*, *MambaIR (ECCV 2024)* | 2D Selective Scan (SS2D) en 4 direcciones transversales con complejidad lineal $O(N)$ en memoria. Acelerado con kernel CUDA o fallback vectorial nativo. | 4-6 GB | ⚡⚡ Rápida | ⭐⭐ Buena |
| **`residual_unet`** | Base configurable (`base_channels: 64`) | *Pre-Act ResNet Style* | Baseline robusto con GroupNorm, activación Mish, Dilated Bottleneck (dilation=2) y atajos de identidad puros ($f(x) + x$). | 4-6 GB | ⚡⚡⚡ Muy Rápida | ⭐⭐ Buena |

---

## 3. Presets de Configuración en `configs/params.yaml`

### Preset 1: NAFNet Base (SOTA Few-Shot VFX)
```yaml
model:
  architecture: nafnet
  size: base          # small (5-8 imgs), base (8-15 imgs), large (>15 imgs)
  in_channels: 3
  out_channels: 3
  dropout_p: 0.05

training:
  epochs: 1000
  batch_size: 4
  learning_rate: 1e-3
  scheduler:
    type: cosine
    params:
      T_max: 1000
      eta_min: 1e-6
```

### Preset 2: Restormer Small (SOTA Vision Transformer)
```yaml
model:
  architecture: restormer
  size: small         # tiny, small, base
  in_channels: 3
  out_channels: 3

training:
  epochs: 1000
  batch_size: 4
  learning_rate: 3e-4
  scheduler:
    type: cosine
    params:
      T_max: 1000
      eta_min: 1e-6
```

### Preset 3: ScopeUNet / GatedUNet Base (Anti-Checkerboard Video)
```yaml
model:
  architecture: scope_unet
  size: base          # small, base
  in_channels: 3
  out_channels: 3

training:
  epochs: 1000
  batch_size: 4
  learning_rate: 1e-3
```

### Preset 4: ConvNeXt-V2 Tiny (Máximo Detalle Facial con 22K Backbone)
```yaml
model:
  architecture: convnext
  size: tiny          # nano, tiny, small, base
  in_channels: 3
  out_channels: 3
  drop_path_rate: 0.10

training:
  epochs: 800
  batch_size: 4
  learning_rate: 5e-4
```

### Preset 5: MambaIR Base (2D Selective Scan)
```yaml
model:
  architecture: mambair
  size: base          # tiny, base, large
  in_channels: 3
  out_channels: 3

training:
  epochs: 1000
  batch_size: 4
  learning_rate: 5e-4
```

---

## 4. Configuración de Pérdidas Híbridas (`HybridLoss`)

CopyAir combina pérdidas espaciales, de frecuencia y perceptuales adaptadas a VFX:

```yaml
loss:
  lambda_l1: 0.0            # L1 MAE clásico (rápido, pero tiende a suavizar)
  lambda_charbonnier: 0.2   # L1 suavizado robusto (mantiene bordes nítidos)
  lambda_perceptual: 0.8    # LPIPS perceptual (AlexNet/VGG) - Calidad visual
  lambda_laplacian: 0.1     # Pirámide Laplaciana (nitidez de frecuencias altas)
  lambda_dino: 0.2          # Features semánticas de DINOv2 / DINOv3
  lambda_ssim: 0.0          # SSIM estructural
  lambda_ffl: 0.0           # Focal Frequency Loss
  lambda_dreamsim: 0.0      # Perceptual de OpenCLIP-ViT (opcional)
  lambda_sobel: 0.0         # Gradientes Sobel
```

---

## 5. Comandos de Ejecución y Testing

### 🔍 Verificación Integral de Todos los Modelos
```bash
python scripts/verify_models.py
```

### 🚀 Entrenamiento
```bash
python scripts/train.py --config configs/params.yaml --device cuda
```

### 🔮 Inferencia en Video con Tiled HD/4K
```bash
python scripts/predict.py \
  --model models/<run_id>/best_model_nafnet.pth \
  --video input_video.mp4 \
  --output output_video.mp4 \
  --tiled \
  --tile-size 512 \
  --overlap 64 \
  --backend ffmpeg \
  --device cuda
```