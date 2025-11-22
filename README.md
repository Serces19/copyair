# CopyAir - Image-to-Image Translation

## 📋 Descripción del Proyecto

CopyAir es una aplicación de **Image-to-Image Translation** que utiliza una red U-Net profunda para aplicar transformaciones visuales (como rejuvenecimiento facial) a través de los frames de un video. El proyecto transforma un notebook de Jupyter en una arquitectura profesional, escalable y reproducible.

### Características principales:
- ✅ Arquitectura U-Net personalizada con skip connections
- ✅ Pérdida híbrida (L1 + SSIM + Perceptual)
- ✅ Datos organizados con versionado
- ✅ Configuración flexible mediante YAML
- ✅ Entrenamiento y validación automatizados
- ✅ Inferencia en video completo
- ✅ Pruebas unitarias incluidas

---

## 📁 Estructura del Proyecto

```
copyair/
├── data/
│   ├── 01_raw/              # Videos originales sin procesar
│   ├── 02_interim/          # Frames extraídos, archivos temporales
│   └── 03_processed/        # Datos listos para entrenamiento
│       ├── input/           # Imágenes de entrada
│       └── ground_truth/    # Imágenes de referencia (objetivo)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py       # PairedImageDataset, VideoFrameDataset
│   │   └── augmentations.py # Transformaciones (albumentations)
│   │
│   ├── models/
│   │   ├── unet.py          # Arquitectura U-Net
│   │   └── losses.py        # L1, SSIM, Perceptual, HybridLoss
│   │
│   └── training/
│       ├── train.py         # Loop de entrenamiento
│       └── inference.py     # Predicción en videos
│
├── notebooks/
│   └── SergioCespedes_TrabajoFinal.ipynb  # Notebook original
│
├── configs/
│   └── params.yaml          # Hiperparámetros centralizados
│
├── scripts/
│   ├── train.py             # Script de entrenamiento
│   └── predict.py           # Script de inferencia
│
├── models/                  # Checkpoints guardados
├── output_inference/        # Videos/frames generados
├── tests/                   # Pruebas unitarias
│
├── requirements.txt         # Dependencias Python
├── .gitignore              # Archivos ignorados por Git
└── README.md               # Este archivo
```

---

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Serces19/copyair.git
cd copyair

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar MLFlow
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "*"
```

### 2. Preparar Datos

Organiza tus datos en `data/03_processed/`:

```
data/03_processed/
├── input/
│   ├── frame_1.jpg
│   ├── frame_2.jpg
│   └── ...
└── ground_truth/
    ├── frame_1.jpg
    ├── frame_2.jpg
    └── ...
```

### 3. Configurar Parámetros

Edita `configs/params.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
```

### 4. Entrenar Modelo

```bash
python scripts/train.py --config configs/params.yaml --device cuda
```

### 5. Inferencia en Video

```bash
# Aplicar modelo y generar video de salida
# El script detecta automáticamente la arquitectura del modelo desde el checkpoint
python scripts/predict.py --model models/best_model.pth --video data/01_raw/input.mp4 --output output3.mp4 --native-resolution
```

**🎯 Detección Automática de Arquitectura**: El script `predict.py` ahora detecta automáticamente la arquitectura del modelo (UNet, NAFNet, ConvNeXt, etc.) desde los metadatos guardados en el checkpoint. No necesitas especificar la arquitectura manualmente.

#### Convertir Modelos Antiguos

Si tienes un modelo entrenado con la versión anterior (sin metadatos), puedes convertirlo:

```bash
# Convertir checkpoint antiguo al nuevo formato con metadatos
python scripts/convert_checkpoint.py --model models/old_model.pth --architecture nafnet --output models/new_model.pth
```

---

## 📊 Archivos Principales

### `src/data/dataset.py`
- `PairedImageDataset`: Carga pares de imágenes (entrada/objetivo)
- `VideoFrameDataset`: Carga frames para inferencia

### `src/models/unet.py`
- `ConvBlock`: Bloque convolucional básico
- `DownBlock`: Downsampling con max pooling
- `UpBlock`: Upsampling con skip connections
- `UNet`: Arquitectura completa

### `src/models/losses.py`
- `L1Loss`: Pérdida L1
- `SSIMLoss`: Índice de similitud estructural
- `PerceptualLoss`: Pérdida basada en características
- `HybridLoss`: Combinación ponderada

### `src/training/train.py`
- `train_epoch()`: Loop de entrenamiento
- `validate()`: Validación
- `compute_metrics()`: Métricas (MSE, PSNR, SSIM)

### `src/training/inference.py`
- `predict_frame()`: Predicción de frame individual
- `predict_on_video()`: Procesamiento de video completo
- `extract_frames_from_video()`: Extrae frames

---

## ⚙️ Configuración (params.yaml)

```yaml
# Parámetros principales
model:
  base_channels: 64      # Canales iniciales
  
training:
  epochs: 50             # Número de épocas
  batch_size: 8          # Tamaño del batch
  learning_rate: 0.001   # Tasa de aprendizaje
  val_split: 0.2         # 20% para validación
  
loss:
  lambda_l1: 0.6         # Peso L1
  lambda_ssim: 0.2       # Peso SSIM
  lambda_perceptual: 0.2 # Peso Perceptual

device: "cuda"           # GPU o CPU
```

---

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
pytest tests/ -v

# Pruebas específicas
pytest tests/test_dataset.py -v
pytest tests/test_models.py -v
```

---

## 📈 Monitoreo de Experimentos

### Usar MLflow (opcional)

```bash
# Instalar
pip install mlflow

# Habilitar en params.yaml
mlflow:
  enabled: true
  experiment_name: "copyair"

# Ejecutar UI
mlflow ui
```

---

## 🐳 Containerización (Docker)

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py"]
```

Construir y ejecutar:

```bash
docker build -t copyair .
docker run --gpus all copyair
```

---

## 📝 Notas Importantes

1. **Datos**: Asegúrate que input/ y ground_truth/ tengan archivos con los mismos nombres
2. **GPU**: Usa `--device cuda` para entrenamientos más rápidos
3. **Memoria**: Para imágenes grandes, reduce `batch_size` en params.yaml
4. **Early Stopping**: Se detiene si no mejora después de N épocas

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Abre un Issue o Pull Request.

---

## 📄 Licencia

MIT License - Ver LICENSE para detalles

---

## 👤 Autor

**Sergio Céspedes** - Trabajo Final

---

## 🔗 Recursos Adicionales

- [PyTorch Documentation](https://pytorch.org/docs/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Albumentations](https://albumentations.ai/)

---

**¡Feliz entrenamiento! 🚀**
