"""
Inferencia: Predicción en videos
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def predict_frame(
    model: nn.Module,
    frame: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Predice un frame individual
    
    Args:
        model: Modelo U-Net
        frame: Frame normalizado [1, 3, H, W]
        device: CPU o GPU
    
    Returns:
        Frame predicho como numpy array [H, W, 3]
    """
    model.eval()
    
    with torch.no_grad():
        frame = frame.to(device)
        output = model(frame)
        output = output.squeeze(0).cpu().numpy()  # [3, H, W]
        output = np.transpose(output, (1, 2, 0))  # [H, W, 3]
        output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
    
    return output


def predict_on_video(
    model: nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    transform=None,
    target_fps: int = 30
) -> str:
    """
    Aplica el modelo a todos los frames de un video
    
    Args:
        model: Modelo U-Net
        video_path: Ruta al video de entrada
        output_path: Ruta al video de salida
        device: CPU o GPU
        transform: Transformaciones a aplicar (albumentations)
        target_fps: FPS del video de salida
    
    Returns:
        Ruta del video generado
    """
    logger.info(f"Procesando video: {video_path}")
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    # Obtener propiedades
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Preparar escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    model.eval()
    frame_count = 0
    
    logger.info(f"Total de frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformaciones
        if transform is not None:
            transformed = transform(image=frame_rgb)
            frame_tensor = transformed['image'].unsqueeze(0)  # [1, 3, H, W]
        else:
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Predicción
        with torch.no_grad():
            frame_tensor = frame_tensor.to(device)
            pred = model(frame_tensor)
            pred = pred.squeeze(0).cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            pred = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        
        # Convertir RGB a BGR para escribir en video
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        out.write(pred_bgr)
        
        frame_count += 1
        if (frame_count) % 10 == 0:
            logger.info(f"Procesados {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    logger.info(f"Video guardado en: {output_path}")
    return output_path


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    sample_rate: int = 1
) -> List[str]:
    """
    Extrae frames de un video
    
    Args:
        video_path: Ruta al video
        output_dir: Directorio de salida
        sample_rate: Cada N frames extraer uno
    
    Returns:
        Lista de rutas de frames extraídos
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    
    logger.info(f"Extrayendo frames de: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame_path = Path(output_dir) / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extraídos {len(saved_frames)} frames")
    
    return saved_frames
