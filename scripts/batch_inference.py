"""
Script de inferencia por lotes: Prueba todos los modelos de MLflow
Uso: python scripts/batch_inference.py --mlflow-uri mlruns --input-dir data/01_raw/input
"""

import argparse
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
from tqdm import tqdm
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from mlflow.tracking import MlflowClient

from src.models.factory import get_model
from src.training.inference import predict_on_video
from src.data.augmentations import get_inference_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_name: str) -> torch.device:
    """Configura el dispositivo"""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Usando CPU")
    return device


def discover_mlflow_runs(
    client: MlflowClient,
    experiment_name: str,
    filter_status: str = "FINISHED"
) -> List[Dict[str, Any]]:
    """
    Descubre todos los runs de MLflow que tienen modelos guardados
    
    Args:
        client: MLflow client
        experiment_name: Nombre del experimento
        filter_status: Filtrar por estado (FINISHED, RUNNING, FAILED, o None para todos)
    
    Returns:
        Lista de diccionarios con información de cada run
    """
    logger.info(f"Buscando runs en experimento: {experiment_name}")
    
    # Obtener experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experimento '{experiment_name}' no encontrado")
        return []
    
    experiment_id = experiment.experiment_id
    logger.info(f"Experiment ID: {experiment_id}")
    
    # Buscar runs
    filter_string = f"attributes.status = '{filter_status}'" if filter_status else ""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"]
    )
    
    logger.info(f"Total de runs encontrados: {len(runs)}")
    
    # Filtrar runs que tienen modelos guardados
    runs_with_models = []
    for run in runs:
        run_id = run.info.run_id
        
        # Verificar si tiene artifacts de modelo
        try:
            artifacts = client.list_artifacts(run_id)
            has_model = any('checkpoint' in art.path or 'model' in art.path for art in artifacts)
            
            if has_model:
                # Extraer metadata
                params = run.data.params
                metrics = run.data.metrics
                tags = run.data.tags
                
                # Formatear timestamp para logging
                from datetime import datetime
                start_time_str = datetime.fromtimestamp(run.info.start_time / 1000.0).strftime('%Y-%m-%d %H:%M')
                
                run_info = {
                    'run_id': run_id,
                    'run_name': tags.get('mlflow.runName', f'run_{run_id[:8]}'),
                    'architecture': params.get('model.architecture', 'unknown'),
                    'epoch': int(params.get('training.epochs', 0)),
                    'train_loss': metrics.get('train/loss', None),
                    'val_loss': metrics.get('val/loss', None),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'params': params,
                    'metrics': metrics,
                    'artifacts': [art.path for art in artifacts]
                }
                runs_with_models.append(run_info)
                logger.info(f"  ✓ Run: {run_id[:8]} | {start_time_str} | {run_info['architecture']} | Epoch: {run_info['epoch']}")
        except Exception as e:
            logger.warning(f"Error al procesar run {run_id}: {e}")
            continue
    
    logger.info(f"Runs con modelos: {len(runs_with_models)}")
    return runs_with_models


def download_model_from_run(
    client: MlflowClient,
    run_id: str,
    temp_dir: Path
) -> Optional[Path]:
    """
    Descarga el modelo de un run de MLflow
    
    Args:
        client: MLflow client
        run_id: ID del run
        temp_dir: Directorio temporal para descargar
    
    Returns:
        Path al modelo descargado o None si falla
    """
    try:
        # Buscar artifact de checkpoint
        artifacts = client.list_artifacts(run_id)
        
        # Buscar el archivo .pth en los artifacts
        model_artifact = None
        for art in artifacts:
            if art.path.endswith('.pth'):
                model_artifact = art.path
                break
            # También buscar en subdirectorios comunes
            if 'checkpoint' in art.path.lower() or 'model' in art.path.lower():
                sub_artifacts = client.list_artifacts(run_id, art.path)
                for sub_art in sub_artifacts:
                    if sub_art.path.endswith('.pth'):
                        model_artifact = sub_art.path
                        break
        
        if model_artifact is None:
            logger.warning(f"No se encontró archivo .pth en run {run_id}")
            return None
        
        # Descargar artifact
        download_path = temp_dir / run_id
        download_path.mkdir(parents=True, exist_ok=True)
        
        local_path = client.download_artifacts(run_id, model_artifact, str(download_path))
        
        logger.info(f"  Modelo descargado: {local_path}")
        return Path(local_path)
        
    except Exception as e:
        logger.error(f"Error al descargar modelo de run {run_id}: {e}")
        return None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict,
    device: torch.device
) -> Optional[torch.nn.Module]:
    """
    Carga modelo desde checkpoint
    
    Args:
        checkpoint_path: Path al checkpoint
        config: Configuración base
        device: Device
    
    Returns:
        Modelo cargado o None si falla
    """
    try:
        logger.info(f"  Cargando modelo desde: {checkpoint_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Detectar formato
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Nuevo formato con metadata
            model_config = checkpoint.get('model_config', config['model'])
            state_dict = checkpoint['model_state_dict']
        else:
            # Formato antiguo (solo state_dict)
            model_config = config['model']
            state_dict = checkpoint
        
        # Crear modelo
        model = get_model(model_config)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        logger.info(f"  ✓ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}")
        return None


def run_inference_on_frames(
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    config: dict,
    tiled: bool = False,
    tile_size: int = 512
):
    """
    Ejecuta inferencia en todos los frames del directorio de entrada
    
    Args:
        model: Modelo cargado
        input_dir: Directorio con frames de entrada
        output_dir: Directorio de salida
        device: Device
        config: Configuración
        tiled: Usar inferencia por tiles
        tile_size: Tamaño de tiles
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Usar predict_on_video que ya soporta directorios de imágenes
    predict_on_video(
        model=model,
        video_path=str(input_dir),
        output_path=str(output_dir),
        device=device,
        transform=None,  # Sin transform para resolución nativa
        target_fps=24,
        native_resolution=True,  # Resolución nativa sin pérdida
        backend='opencv',
        lossless=True,  # PNG lossless
        tiled=tiled,
        tile_size=tile_size,
        overlap=64
    )


def create_output_folder_name(run_info: Dict[str, Any]) -> str:
    """
    Crea nombre de carpeta de salida con metadata del run
    
    Args:
        run_info: Información del run
    
    Returns:
        Nombre de carpeta
    """
    from datetime import datetime
    
    run_id_short = run_info['run_id'][:8]
    arch = run_info['architecture']
    epoch = run_info['epoch']
    
    # Formatear timestamp
    start_time = datetime.fromtimestamp(run_info['start_time'] / 1000.0).strftime('%Y%m%d_%H%M')
    
    # Incluir loss si está disponible
    if run_info['train_loss'] is not None:
        loss_str = f"_loss{run_info['train_loss']:.4f}"
    else:
        loss_str = ""
    
    # Formato: runid_fecha_arquitectura_epocas_loss
    folder_name = f"{run_id_short}_{start_time}_{arch}_{epoch}ep{loss_str}"
    
    return folder_name


def save_run_metadata(output_dir: Path, run_info: Dict[str, Any]):
    """Guarda metadata del run en JSON"""
    metadata_path = output_dir / "metadata.json"
    
    # Convertir a formato serializable
    metadata = {
        'run_id': run_info['run_id'],
        'run_name': run_info['run_name'],
        'architecture': run_info['architecture'],
        'epoch': run_info['epoch'],
        'train_loss': run_info['train_loss'],
        'val_loss': run_info['val_loss'],
        'status': run_info['status'],
        'start_time': run_info['start_time'],
        'params': run_info['params'],
        'metrics': run_info['metrics']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  ✓ Metadata guardada: {metadata_path}")


def batch_inference(
    config: dict,
    mlflow_uri: str,
    experiment_name: str,
    input_dir: Path,
    output_base_dir: Path,
    device: torch.device,
    filter_status: str = "FINISHED",
    max_runs: Optional[int] = None,
    tiled: bool = False,
    tile_size: int = 512
):
    """
    Ejecuta inferencia por lotes en todos los modelos de MLflow
    
    Args:
        config: Configuración
        mlflow_uri: URI de MLflow
        experiment_name: Nombre del experimento
        input_dir: Directorio con frames de entrada
        output_base_dir: Directorio base de salida
        device: Device
        filter_status: Filtrar runs por estado
        max_runs: Máximo número de runs a procesar
        tiled: Usar inferencia por tiles
        tile_size: Tamaño de tiles
    """
    # Configurar MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(mlflow_uri)
    
    # Descubrir runs
    runs = discover_mlflow_runs(client, experiment_name, filter_status)
    
    if not runs:
        logger.warning("No se encontraron runs con modelos")
        return
    
    # Limitar número de runs si se especifica
    if max_runs is not None:
        runs = runs[:max_runs]
        logger.info(f"Procesando solo los primeros {max_runs} runs")
    
    # Crear directorio temporal para descargas
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Procesar cada run
        summary = []
        for i, run_info in enumerate(runs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Procesando run {i}/{len(runs)}: {run_info['run_name']}")
            logger.info(f"{'='*80}")
            
            try:
                # Descargar modelo
                model_path = download_model_from_run(client, run_info['run_id'], temp_path)
                if model_path is None:
                    logger.warning(f"  ✗ No se pudo descargar modelo, saltando...")
                    summary.append({**run_info, 'status': 'download_failed'})
                    continue
                
                # Cargar modelo
                model = load_model_from_checkpoint(model_path, config, device)
                if model is None:
                    logger.warning(f"  ✗ No se pudo cargar modelo, saltando...")
                    summary.append({**run_info, 'status': 'load_failed'})
                    continue
                
                # Crear carpeta de salida
                folder_name = create_output_folder_name(run_info)
                output_dir = output_base_dir / folder_name
                
                logger.info(f"  Guardando resultados en: {output_dir}")
                
                # Ejecutar inferencia
                run_inference_on_frames(
                    model=model,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    device=device,
                    config=config,
                    tiled=tiled,
                    tile_size=tile_size
                )
                
                # Guardar metadata
                save_run_metadata(output_dir, run_info)
                
                logger.info(f"  ✓ Inferencia completada para {run_info['run_name']}")
                summary.append({**run_info, 'status': 'success', 'output_dir': str(output_dir)})
                
                # Limpiar modelo de memoria
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"  ✗ Error procesando run {run_info['run_name']}: {e}")
                summary.append({**run_info, 'status': 'error', 'error': str(e)})
                continue
        
        # Guardar resumen
        summary_path = output_base_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Procesamiento completado")
        logger.info(f"  Total runs procesados: {len(runs)}")
        logger.info(f"  Exitosos: {sum(1 for s in summary if s.get('status') == 'success')}")
        logger.info(f"  Fallidos: {sum(1 for s in summary if s.get('status') != 'success')}")
        logger.info(f"  Resumen guardado en: {summary_path}")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Inferencia por lotes con modelos de MLflow")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Configuración')
    parser.add_argument('--mlflow-uri', type=str, default=None, help='MLflow tracking URI (default: desde config)')
    parser.add_argument('--experiment-name', type=str, default=None, help='Nombre del experimento (default: desde config)')
    parser.add_argument('--input-dir', type=str, default='data/01_raw/input', help='Directorio con frames de entrada')
    parser.add_argument('--output-dir', type=str, default='output_inference', help='Directorio base de salida')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo (cuda/cpu)')
    parser.add_argument('--filter-status', type=str, default='FINISHED', help='Filtrar runs por estado (FINISHED, RUNNING, FAILED, o None para todos)')
    parser.add_argument('--max-runs', type=int, default=None, help='Máximo número de runs a procesar')
    parser.add_argument('--tiled', action='store_true', help='Usar inferencia por tiles (para HD/4K)')
    parser.add_argument('--tile-size', type=int, default=512, help='Tamaño de tiles')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # MLflow URI
    mlflow_uri = args.mlflow_uri or config.get('mlflow', {}).get('tracking_uri', 'mlruns')
    experiment_name = args.experiment_name or config.get('mlflow', {}).get('experiment_name', 'copyair_experiments')
    
    # Paths
    input_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        return
    
    # Device
    device = setup_device(args.device)
    
    # Ejecutar batch inference
    batch_inference(
        config=config,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        input_dir=input_dir,
        output_base_dir=output_base_dir,
        device=device,
        filter_status=args.filter_status if args.filter_status.lower() != 'none' else None,
        max_runs=args.max_runs,
        tiled=args.tiled,
        tile_size=args.tile_size
    )


if __name__ == '__main__':
    main()
