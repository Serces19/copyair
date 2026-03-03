# CopyAir AI Agent Guidelines 🤖

Este archivo sirve como referencia maestra y manifiesto de estilo para agentes de IA operando sobre la base de código del proyecto CopyAir. Contiene todo el contexto de arquitectura, herramientas de desarrollo y preferencias rigurosas del mantenedor.

## 🛠 Tech Stack y Entorno
- **Dominio:** Cloud Architeture, MLOps, Computer Vision orientado a VFX (Virtual Production, De-aging).
- **Core de Machine Learning:** Python 3.x, PyTorch `2.1.0-cuda12.1`.
- **CV & Transformaciones:** OpenCV, y primariamente Albumentations (por su velocidad comprobada).
- **Tracking:** MLflow (para versionado y evaluación cruzada).
- **Control de dependencias:** **`uv` es OBLIGATORIO** para creación de entornos virtuales e instalación.
- **Sistema Operativo Default:** Windows (PowerShell/CMD).

## 📐 Estrategia de Arquitectura & Reglas de Negocio
### 1. Parámetros Basados en Configuración (`configs/params.yaml`)
El código debe evitar valores hardcodeados a toda costa. El "cerebro" que orquesta desde capas del modelo (ModernUnet vs SimpleUnet) hasta learning rates reside en `params.yaml`. Antes de crear lógicas nuevas, considera si los atributos deberían residir ahí.

### 2. Estrategia Polyrepo ("Backend and UI Separation")
En el plan a producción (NeuralShot Platform), CopyAir es puramente un 'Worker' que se hosteará en Vast.ai coordinado por AWS Serverless. La UI (React/Vercel) vive en el repo `scopeair_frontend`. 
- **Integración:** Las invocaciones a procesos deben planificarse para poder ser envueltas por un framework REST/WebSocket (FastAPI).

### 3. Código "Production-Ready" & "Less is More"
Las soluciones deben ser simples, robustas, y pensadas para escalar a entornos de producción Serverless y de automatización. 
- Omite abstracciones OOP complejas si no son extrictamente necesarias. 
- El código de testeo debe ser modular y directo (pytest).

## 💻 Convenciones de Código y Ejecución
1. Siempre se requiere iniciar transacciones de código via entorno aislado. Prefiere explícitamente `.venv\Scripts\activate` en comandos shell.
2. Al proponer comandos, respeta el formato de Windows y asume el entono en CMD (`cmd /c`) si fuese requerido un comando empaquetado; el uso de `&&` requeriría entornos robustos.
3. Al manipular paths (rutas de directorios) en Python: Siempre hazlos cross-OS con `os.path.join` o `pathlib.Path`, pero en scripting de consola asume barras de Windows (`\`).
4. Si un comando en consola de sistema operativo falla, no dudes en programar scripts Python de rescate (`os`, `shutil`) para lograr la misma meta a la fuerza.

## 🏗 Estructura del Código
- `data/`: `01_raw`, `02_interim`, `03_processed`. Estructura robusta y ordenada, donde Input y Ground Truth deben hacer match de frames.
- `src/`: Lógica Core:
    - `data/`: Carga de frames y pares. Augmentations orientadas sutilmente a VFX.
    - `models/`: Constructores de PyTorch Unet y combinacón de Losses (Híbridas).
    - `training/`: Funciones aisladas de validación y de loop (Epochs).
- `scripts/`: Herramientas ejecutables de usuario final (`train.py`, `predict.py`, `batch_inference.py`), fáciles de integrar.
- `.kiro/specs/`: Documentos de diseño de agentes previos para planear el SaaS futuro.

## 📝 Resumen Operativo de Inteligencia Artificial
En caso de duda sobre redacciones, el código se comenta usualmente en inglés para mantener un estándar limpio y global. Sin embargo, explicaciones de diseño pueden discutirse en español dependiendo de la ventana actual de contexto con el usuario. Las variables, funciones y estructuras de JSON/YAML van estricta y rígidamente en **inglés**.
