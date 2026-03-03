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

## � Documentación Detallada
Para más detalles técnicos, consulta la carpeta `docs/`:
- [`CLOUD_ARCHITECTURE.md`](docs/CLOUD_ARCHITECTURE.md): Guía de despliegue en la nube.
- [`DEVELOPMENT.md`](docs/DEVELOPMENT.md): Detalles de implementación del core.
- [`API_INTEGRATION.md`](docs/API_INTEGRATION.md): Documentación de los endpoints.

---
*Desarrollado para NeuralShot Platform.*
