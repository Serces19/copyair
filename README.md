# CopyAir / NeuralShot Platform

CopyAir es una arquitectura profesional para entrenamiento iterativo y escalable de modelos de **Image-to-Image translation**, especializados en VFX y De-aging. Permite escalar desde scripts de experimentación locales hasta una orquestación en la nube robusta y costo-eficiente.

## 📋 Características Principales
- **Arquitectura Modular:** Soporta U-Net clásica, U-Net residual y diseños modernos inspirados en modelos de de-aging SOTA.
- **Pérdida Híbrida:** Combinación de L1, SSIM, y Perceptual Loss explícitamente diseñada para preservar textura generada para postproducción.
- **Configuración Centralizada:** Todo factor de influencia es gobernado por `configs/params.yaml`.
- **Inferencia en batch y video:** Pipeline optimizado para inferencia rápida por frames.
- **Integración con MLflow:** Seguimiento estricto y ordenado de experimentos.

## 🚀 Inicio Rápido

### 1. Instalación usando `uv`
Se prefiere fuertemente el uso de `uv` para control de paquetes (especialmente en entornos Windows).

```bash
# Clonar el repositorio
git clone https://github.com/Serces19/copyair.git
cd copyair

# Crear entorno virtual con uv
uv venv .venv

# Activar entorno (Windows)
.venv\Scripts\activate

# Instalar dependencias
uv pip install -r requirements.txt
```

### 2. Entrenar Modelo
Asegúrate de preparar y emparejar tus imágenes en `data/03_processed/input/` y `data/03_processed/ground_truth/`. Configura el archivo `configs/params.yaml` e inicia el ciclo:

```bash
uv run scripts/train.py --config configs/params.yaml --device cuda
```

### 3. Inferencia
Aplica tu modelo a un video o extrae y empalma frames al vuelo:
```bash
uv run scripts/predict.py --model models/best_model_unet.pth --video data/01_raw/input.mov --output output.mp4 --native-resolution --lossless
```
Para inferencia masiva y testeando resultados frente a múltilpes iteraciones alojadas en MLflow:
```bash
uv run scripts/batch_inference.py --input-dir data/01_raw/input --output-dir output_inference --device cuda
```

## 📚 Documentación Adicional

Para mantener este archivo limpio, las especificaciones profundas fueron movidas a la carpeta `docs/`:

- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): Detalles de la arquitectura de código, módulos internos y debuggear entrenamiento.
- [`docs/BATCH_INFERENCE.md`](docs/BATCH_INFERENCE.md): Guía de descubrimiento, descarga automática e inferencia global para docenas de runs utilizando la API de MLflow.
- [`docs/GIT_SETUP.md`](docs/GIT_SETUP.md): Estrategia de ramas y control de versiones.
- [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md): Hitos fundacionales e históricos.

> Nota: Los documentos en `.kiro/specs/` pertenecen a las especificaciones teóricas para la arquitectura Cloud Native.

---
### 🌐 Nexo con el Frontend 
Actualmente, CopyAir actúa como el núcleo ML backend. La interfaz de usuario recae en el sistema **ScopeAir** ubicado en un repositorio separado (enfoque Polyrepo). Futuras iteraciones sustituirán la edición manual de `params.yaml` con la ingesta automatizada por una API local FastAPI (Desarrollo) y endpoints serverless API Gateway en AWS (Production).
