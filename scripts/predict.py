"""
Script de inferencia: Aplicar modelo a video
Uso: python scripts/predict.py --model models/best_model.pth --video input.mp4 --output output.mp4
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import get_model
from src.training.inference import predict_on_video, extract_frames_from_video
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


def load_model(model_path: str, config: dict, device: torch.device):
    """Carga modelo preentrenado detectando arquitectura automáticamente"""
    logger.info(f"Cargando modelo desde: {model_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detectar si el checkpoint tiene metadatos (nuevo formato) o solo pesos (formato antiguo)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Nuevo formato: checkpoint con metadatos
        model_config = checkpoint['model_config']
        architecture = checkpoint['architecture']
        state_dict = checkpoint['model_state_dict']
        
        logger.info(f"✓ Metadatos detectados en checkpoint")
        logger.info(f"  Arquitectura: {architecture}")
        logger.info(f"  Época: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if checkpoint.get('val_loss') else "")
    else:
        # Formato antiguo: solo state_dict
        logger.warning("⚠ Checkpoint sin metadatos (formato antiguo)")
        logger.warning("  Usando configuración de params.yaml")
        model_config = config['model']
        architecture = config['model'].get('architecture', 'unet')
        state_dict = checkpoint
        logger.info(f"  Arquitectura asumida: {architecture}")
    
    # Crear modelo usando factory con la configuración detectada
    model = get_model(model_config)
    
    # Cargar pesos
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ Modelo cargado exitosamente")
    
    return model


def predict(config: dict, model_path: str, video_path: str, output_path: str, device: torch.device, native_resolution: bool = False, backend: str = 'opencv'):
    """Predice en video"""
    
    # Cargar modelo
    model = load_model(model_path, config, device)
    
    # Transformaciones
    # Si native_resolution es True, NO redimensionamos (resize=False)
    should_resize = not native_resolution
    transform = get_inference_transforms(config['augmentation']['img_size'], resize=should_resize)
    
    # Predicción
    predict_on_video(
        model,
        video_path,
        output_path,
        device,
        transform=transform,
        target_fps=config['inference']['target_fps'],
        native_resolution=native_resolution,
        backend=backend
    )
    
    logger.info(f"✓ Video generado: {output_path}")


def extract_frames(video_path: str, output_dir: str, sample_rate: int = 1):
    """Extrae frames de un video"""
    logger.info(f"Extrayendo frames de: {video_path}")
    
    extract_frames_from_video(video_path, output_dir, sample_rate)
    
    logger.info(f"✓ Frames extraídos en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inferencia con U-Net")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Configuración')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo preentrenado')
    parser.add_argument('--video', type=str, help='Ruta al video de entrada')
    parser.add_argument('--output', type=str, help='Ruta al video de salida')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo')
    parser.add_argument('--extract-frames', action='store_true', help='Solo extraer frames')
    parser.add_argument('--native-resolution', action='store_true', help='Usar resolución nativa del video (sin resize)')
    parser.add_argument('--backend', type=str, default='opencv', choices=['opencv', 'ffmpeg'], help='Backend de video (opencv: seguro color, ffmpeg: mejor compresión)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    config['device'] = args.device
    
    # Dispositivo
    device = setup_device(config['device'])
    
    if args.extract_frames and args.video:
        extract_frames(args.video, config['data']['frames_dir'])
    elif args.video and args.output:
        predict(config, args.model, args.video, args.output, device, args.native_resolution, args.backend)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
