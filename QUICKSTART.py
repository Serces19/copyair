"""
Guía Rápida de Inicio - CopyAir

Este archivo te guía paso a paso para comenzar con el proyecto.
"""

# ============================================================================
# PASO 1: INSTALAR DEPENDENCIAS
# ============================================================================

"""
Opción A (Recomendado - Windows PowerShell):
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Opción B (Linux/Mac):
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Opción C (Usar Makefile):
    make install
"""

# ============================================================================
# PASO 2: PREPARAR DATOS
# ============================================================================

"""
Copia tus imágenes de entrenamiento en esta estructura:

    data/03_processed/
    ├── input/
    │   ├── frame_001.jpg
    │   ├── frame_002.jpg
    │   └── ...
    └── ground_truth/
        ├── frame_001.jpg
        ├── frame_002.jpg
        └── ...

⚠️  IMPORTANTE: Los nombres de archivos DEBEN coincidir en ambas carpetas
"""

# ============================================================================
# PASO 3: CONFIGURAR PARÁMETROS
# ============================================================================

"""
Edita 'configs/params.yaml' con tus parámetros:

    training:
      epochs: 50              # Número de épocas
      batch_size: 8           # Tamaño del lote (reduce si hay OOM)
      learning_rate: 0.001    # Tasa de aprendizaje
      
    model:
      base_channels: 64       # Canales iniciales (aumentar = más capacidad)
    
    loss:
      lambda_l1: 0.6          # Peso L1
      lambda_ssim: 0.2        # Peso SSIM
      lambda_perceptual: 0.2  # Peso Perceptual
"""

# ============================================================================
# PASO 4: ENTRENAR MODELO
# ============================================================================

"""
Opción A (Script directo):
    python scripts/train.py --config configs/params.yaml --device cuda

Opción B (Con GPU automático):
    python scripts/train.py --config configs/params.yaml

Opción C (Sin GPU):
    python scripts/train.py --config configs/params.yaml --device cpu

Opción D (Usar Makefile):
    make train          # Con GPU (CUDA)
    make train-cpu      # Sin GPU
"""

# ============================================================================
# PASO 5: INFERENCIA EN VIDEO
# ============================================================================

"""
Opción A (Paso a paso):
    # 1. Extraer frames del video
    python scripts/predict.py --model models/best_model.pth \\
                               --video input.mp4 \\
                               --extract-frames
    
    # 2. Aplicar modelo y generar video de salida
    python scripts/predict.py --model models/best_model.pth \\
                               --video input.mp4 \\
                               --output output.mp4

Opción B (Directamente):
    python scripts/predict.py --model models/best_model.pth \\
                               --video input.mp4 \\
                               --output output.mp4
"""

# ============================================================================
# PASO 6: VERIFICAR CON PRUEBAS
# ============================================================================

"""
Ejecutar todas las pruebas:
    pytest tests/ -v

Pruebas específicas:
    pytest tests/test_dataset.py -v
    pytest tests/test_models.py -v

Con cobertura:
    pytest tests/ --cov=src --cov-report=html
"""

# ============================================================================
# COMANDOS ÚTILES
# ============================================================================

"""
Limpiar archivos temporales:
    make clean

Lintear código:
    make lint

Construir imagen Docker:
    make docker-build
    make docker-run

Ver estructura del proyecto:
    tree /F (Windows)
    tree (Linux/Mac)
"""

# ============================================================================
# SOLUCIÓN DE PROBLEMAS
# ============================================================================

"""
❌ Error: "ModuleNotFoundError: No module named 'torch'"
✓ Solución: pip install -r requirements.txt

❌ Error: "CUDA out of memory"
✓ Solución: Reduce batch_size en configs/params.yaml

❌ Error: "No se encontraron imágenes"
✓ Solución: Verifica que data/03_processed/input/ y ground_truth/ tengan archivos

❌ Error: "RuntimeError: Expected 3D or 4D input"
✓ Solución: Las imágenes deben ser RGB (3 canales), no escala de grises

❌ Error: "CUDA is not available"
✓ Solución: Usa --device cpu en los comandos
"""

# ============================================================================
# ESTRUCTURA DEL PROYECTO
# ============================================================================

"""
copyair/
├── data/                   # Datos
│   ├── 01_raw/            # Videos originales
│   ├── 02_interim/        # Frames extraídos
│   └── 03_processed/      # Pares para entrenamiento
│
├── src/                   # Código fuente
│   ├── data/              # Carga de datos
│   ├── models/            # Arquitecturas
│   └── training/          # Entrenamiento/Inferencia
│
├── scripts/               # Scripts ejecutables
│   ├── train.py          # Entrena modelo
│   └── predict.py        # Inferencia
│
├── configs/               # Configuración
│   └── params.yaml       # Parámetros
│
├── models/                # Checkpoints guardados
├── output_inference/      # Videos generados
├── tests/                 # Pruebas unitarias
│
├── README.md             # Documentación
├── requirements.txt      # Dependencias
└── Dockerfile           # Para Docker
"""

# ============================================================================
# DOCUMENTACIÓN ADICIONAL
# ============================================================================

"""
📖 Leer:
  - README.md: Descripción general del proyecto
  - DEVELOPMENT.md: Guía de desarrollo y arquitectura
  - configs/params.yaml: Todos los parámetros disponibles

🔗 Recursos:
  - PyTorch: https://pytorch.org/
  - U-Net Paper: https://arxiv.org/abs/1505.04597
  - Albumentations: https://albumentations.ai/

💬 Soporte:
  - Revisa los logs en logs/
  - Ejecuta: pytest tests/ -v
"""

print(__doc__)
