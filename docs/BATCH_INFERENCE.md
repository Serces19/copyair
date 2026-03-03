# Batch Inference - Guía de Uso

## Descripción

`batch_inference.py` es un script que automáticamente descubre todos los modelos entrenados en MLflow, los descarga, y ejecuta inferencia en una secuencia de frames, guardando los resultados organizados por modelo.

## Características

- ✅ **Descubrimiento automático**: Busca todos los runs de MLflow con modelos guardados
- ✅ **Descarga automática**: Descarga checkpoints desde MLflow (local o nube)
- ✅ **Resolución nativa**: Procesa frames sin pérdida de calidad (solo padding)
- ✅ **Formato lossless**: Guarda resultados en PNG sin compresión
- ✅ **Metadata completa**: Guarda información del modelo en cada carpeta
- ✅ **Resumen global**: Genera reporte de todos los modelos procesados
- ✅ **Manejo de errores**: Continúa procesando si un modelo falla

## Requisitos

```bash
# Activar entorno
conda activate base

# Verificar que MLflow esté instalado
python -c "import mlflow; print(mlflow.__version__)"
```

## Uso Básico

### 1. Inferencia en todos los modelos (FINISHED)

```bash
python scripts/batch_inference.py \
  --input-dir data/01_raw/input \
  --output-dir output_inference \
  --device cuda
```

Esto procesará todos los runs con estado `FINISHED` del experimento `copyair_experiments`.

### 2. Usar MLflow en la nube

Si tus modelos están en un servidor MLflow remoto:

```bash
python scripts/batch_inference.py \
  --mlflow-uri http://tu-servidor-mlflow:5000 \
  --experiment-name copyair_experiments \
  --input-dir data/01_raw/input \
  --output-dir output_inference \
  --device cuda
```

### 3. Procesar solo los mejores N modelos

```bash
python scripts/batch_inference.py \
  --max-runs 5 \
  --device cuda
```

### 4. Incluir runs en progreso o fallidos

```bash
# Todos los runs (sin filtro de estado)
python scripts/batch_inference.py --filter-status none

# Solo runs fallidos (para debugging)
python scripts/batch_inference.py --filter-status FAILED
```

### 5. Usar inferencia por tiles (para alta resolución)

Si tus frames son muy grandes (>1080p) y tienes problemas de memoria:

```bash
python scripts/batch_inference.py \
  --tiled \
  --tile-size 512 \
  --device cuda
```

## Argumentos Completos

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--config` | `configs/params.yaml` | Archivo de configuración |
| `--mlflow-uri` | Desde config o `mlruns` | URI del servidor MLflow |
| `--experiment-name` | Desde config o `copyair_experiments` | Nombre del experimento |
| `--input-dir` | `data/01_raw/input` | Directorio con frames de entrada |
| `--output-dir` | `output_inference` | Directorio base de salida |
| `--device` | `cuda` | Dispositivo (cuda/cpu) |
| `--filter-status` | `FINISHED` | Filtrar por estado (FINISHED/RUNNING/FAILED/none) |
| `--max-runs` | `None` | Máximo número de runs a procesar |
| `--tiled` | `False` | Usar inferencia por tiles |
| `--tile-size` | `512` | Tamaño de tiles (si --tiled) |

## Estructura de Salida

```
output_inference/
├── unet_adamw_2500ep_loss0.0234/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   ├── ...
│   ├── frame_000200.png
│   └── metadata.json
├── nafnet_adamw_1500ep_loss0.0189/
│   ├── frame_000000.png
│   ├── ...
│   └── metadata.json
└── batch_summary.json
```

### Metadata JSON

Cada carpeta de modelo contiene `metadata.json` con:

```json
{
  "run_id": "abc123...",
  "run_name": "unet_adamw",
  "architecture": "unet",
  "epoch": 2500,
  "train_loss": 0.0234,
  "val_loss": 0.0189,
  "status": "FINISHED",
  "start_time": 1701234567890,
  "params": {
    "model.architecture": "unet",
    "training.learning_rate": "0.001",
    ...
  },
  "metrics": {
    "train/loss": 0.0234,
    "val/psnr": 32.45,
    ...
  }
}
```

### Resumen Global

`batch_summary.json` contiene un resumen de todos los runs procesados:

```json
[
  {
    "run_id": "abc123...",
    "run_name": "unet_adamw",
    "architecture": "unet",
    "status": "success",
    "output_dir": "output_inference/unet_adamw_2500ep_loss0.0234"
  },
  {
    "run_id": "def456...",
    "run_name": "nafnet_adamw",
    "architecture": "nafnet",
    "status": "download_failed"
  }
]
```

## Flujo de Trabajo Recomendado

### En tu máquina local (sin GPU)

1. **Listar modelos disponibles** (sin ejecutar inferencia):

```bash
# Crear un script simple para listar runs
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient('http://tu-servidor-mlflow:5000')
experiment = client.get_experiment_by_name('copyair_experiments')
runs = client.search_runs([experiment.experiment_id], filter_string='attributes.status = \"FINISHED\"')

print(f'Total runs: {len(runs)}')
for run in runs[:10]:
    params = run.data.params
    metrics = run.data.metrics
    print(f\"  {run.data.tags.get('mlflow.runName', 'N/A')}: {params.get('model.architecture', 'N/A')} | Loss: {metrics.get('train/loss', 'N/A'):.4f}\")
"
```

### En tu máquina en la nube (con GPU)

2. **Ejecutar batch inference**:

```bash
# Conectar a tu servidor MLflow y procesar todos los modelos
python scripts/batch_inference.py \
  --mlflow-uri http://tu-servidor-mlflow:5000 \
  --device cuda \
  --input-dir data/01_raw/input \
  --output-dir output_inference
```

3. **Revisar resultados**:

```bash
# Ver resumen
cat output_inference/batch_summary.json

# Ver frames de un modelo específico
ls output_inference/unet_adamw_2500ep_loss0.0234/
```

## Troubleshooting

### Error: "No se encontraron runs con modelos"

**Causa**: No hay runs en MLflow con artifacts de modelo guardados.

**Solución**: Verifica que `mlflow.pytorch.log_model()` se esté llamando en `train.py` (línea 408).

### Error: "No se pudo descargar modelo"

**Causa**: El artifact no existe o no tienes permisos.

**Solución**: 
- Verifica la URI de MLflow: `--mlflow-uri`
- Verifica que el run tenga artifacts: `mlflow.tracking.MlflowClient().list_artifacts(run_id)`

### Error: "CUDA out of memory"

**Causa**: Los frames son muy grandes para la GPU.

**Solución**: Usa inferencia por tiles:

```bash
python scripts/batch_inference.py --tiled --tile-size 512 --device cuda
```

O usa CPU (más lento):

```bash
python scripts/batch_inference.py --device cpu
```

### Error: "No se pudo cargar modelo"

**Causa**: El checkpoint tiene un formato incompatible.

**Solución**: Verifica que el checkpoint tenga la estructura correcta:

```python
checkpoint = torch.load('model.pth')
print(checkpoint.keys())  # Debe tener 'model_state_dict', 'model_config', 'architecture'
```

## Comparación con predict.py

| Característica | `predict.py` | `batch_inference.py` |
|----------------|--------------|----------------------|
| **Input** | Un modelo específico | Todos los modelos de MLflow |
| **Descubrimiento** | Manual | Automático |
| **Metadata** | No guarda | Guarda JSON completo |
| **Resumen** | No | Sí (`batch_summary.json`) |
| **Uso** | Testing individual | Comparación masiva |

## Ejemplos de Uso Avanzado

### Procesar solo modelos UNet

```bash
# Filtrar en el código o usar max-runs con orden
python scripts/batch_inference.py --max-runs 10
```

### Procesar en CPU para no bloquear GPU de entrenamiento

```bash
python scripts/batch_inference.py --device cpu
```

### Guardar en disco externo

```bash
python scripts/batch_inference.py \
  --output-dir /mnt/external/copyair_results \
  --device cuda
```

## Notas Importantes

- ⚠️ **Espacio en disco**: Cada secuencia de 201 frames en PNG (~1920x1080) ocupa ~400MB. Planifica espacio suficiente.
- ⚠️ **Tiempo de procesamiento**: Con GPU, cada modelo toma ~2-5 minutos para 201 frames. Con CPU, ~15-30 minutos.
- ⚠️ **Descarga de modelos**: Los modelos se descargan a un directorio temporal y se eliminan automáticamente al finalizar.
- ✅ **Resolución nativa**: Los frames se procesan en su resolución original (sin resize), solo se aplica padding para cumplir requisitos de la arquitectura.
- ✅ **Formato lossless**: PNG garantiza cero pérdida de calidad.

## Siguiente Paso

Después de ejecutar batch inference, puedes:

1. **Comparar visualmente** los resultados de diferentes modelos
2. **Crear videos** de comparación lado a lado
3. **Calcular métricas** (PSNR, SSIM, LPIPS) si tienes ground truth
4. **Seleccionar el mejor modelo** para producción

---

**¿Necesitas ayuda?** Revisa los logs del script o abre un issue con el error completo.
