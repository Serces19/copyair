# CopyAir / NeuralShot Platform

CopyAir es el core engine profesional para entrenamiento e inferencia de modelos de **Image-to-Image translation** (VFX / De-aging). Diseñado como una arquitectura de "Nodo de Ejecución" escalable desde local hasta servidores de GPU en la nube (Vast.ai).

---

## 🚀 Guía de Instalación (Desde Cero)

Sigue estos pasos para preparar el entorno en tu máquina local o en un servidor remoto.

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Serces19/copyair.git
cd copyair
```

### 2. Configurar el Entorno (usando `uv`)
Se prefiere `uv` por su velocidad y manejo estricto de dependencias en Windows/Linux.

**En Windows:**
```bash
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

**En Linux (Server/Cloud):**
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Iniciar MLFlow:**
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

### 3. Lanzar el "Control Center"
Este es el método recomendado para gestionar el core. Inicia la API y la interfaz profesional de una vez.

```bash
python run_control_center.py
```

- **Local:** Se abrirá automáticamente [http://localhost:8000/](http://localhost:8000/).
- **Nube (Remote):** Accede a la IP de tu servidor en el puerto `8000`.
- **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🎛️ Modos de Uso

### A. Gestión vía Control Center (Recomendado)
Desde la UI puedes:
- Editar visualmente todos los parámetros de `configs/params.yaml`.
- Lanzar entrenamientos (`Run Training`) y ver logs en tiempo real vía WebSockets.
- Ejecutar inferencias (`Inference`) con feedback inmediato.
- Detener procesos en ejecución (`Emergency Stop`).

### B. Ejecución vía CLI (Modo Experto)

**Entrenamiento:**
```bash
uv run scripts/train.py --config configs/params.yaml --device cuda
```

**Inferencia Single/Video:**
```bash
uv run scripts/predict.py --model models/best_model_unet.pth --video data/input.mov --output output.mp4
```

---

## 🏗️ Arquitectura del Sistema

CopyAir funciona como un **Nodo de Ejecución** (Edge Node) que se conecta a la plataforma NeuralShot:
- **FastAPI Layer:** Provee la comunicación REST/WS para el frontend (ScopeAir).
- **ML Engine:** Basado en PyTorch con soporte para arquitecturas U-Net híbridas y pérdidas de alta fidelidad.
- **Cloud Ready:** Totalmente compatible con despliegues en contenedores para Vast.ai.

## ☁️ Google Colab (Ejecución en GPU Cloud)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Serces19/copyair/blob/main/notebooks/copyair_colab_training.ipynb)

Puedes entrenar y realizar inferencias directamente en Google Colab con aceleración GPU (T4/A100) utilizando el notebook oficial:
- [`notebooks/copyair_colab_training.ipynb`](notebooks/copyair_colab_training.ipynb)
- Incluye sincronización directa con **Google Drive / Google Drive Desktop**, registro offline de MLflow (`sqlite:///mlflow.db`) e inferencia HD/4K por tiles.

---

## 📚 Documentación Detallada (Índice `/docs/`)
Para más detalles técnicos y guías de arquitectura, consulta la carpeta `/docs/`:
- [`MODELS_AND_TRAINING_GUIDE.md`](docs/MODELS_AND_TRAINING_GUIDE.md): **Guía maestra de modelos (NAFNet, ConvNeXt, Residual U-Net, etc.), configuración de pérdidas híbridas y comandos de entrenamiento/inferencia.**
- [`CLOUD_ARCHITECTURE.md`](docs/CLOUD_ARCHITECTURE.md): Guía de despliegue en la nube (Vast.ai, AWS Serverless).
- [`DEVELOPMENT.md`](docs/DEVELOPMENT.md): Detalles de implementación del core y flujos de trabajo.
- [`API_INTEGRATION.md`](docs/API_INTEGRATION.md): Documentación de los endpoints y WebSocket del backend.
- [`NAFNET_CONVNEXT_GUIDE.md`](docs/NAFNET_CONVNEXT_GUIDE.md): Guía técnica especializada en Few-Shot Learning con NAFNet y ConvNeXt.
- [`SCHEDULERS_GUIDE.md`](docs/SCHEDULERS_GUIDE.md): Guía de schedulers (Cosine, OneCycle, Plateau).
- [`BATCH_INFERENCE.md`](docs/BATCH_INFERENCE.md): Inferencia por lotes y comparación de modelos de MLflow.

---

Snippet para actualizar el repo facilmente
```bash
git config --global alias.sync '!git fetch origin && git reset --hard origin/main'
```
*Desarrollado para NeuralShot Platform.*
