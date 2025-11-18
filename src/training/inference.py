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


def pad_to_divisor(img: np.ndarray, divisor: int = 32) -> tuple:
    """Rellena la imagen para que sea divisible por el divisor (necesario para U-Net)"""
    h, w = img.shape[:2]
    h_new = ((h + divisor - 1) // divisor) * divisor
    w_new = ((w + divisor - 1) // divisor) * divisor
    
    pad_h = h_new - h
    pad_w = w_new - w
    
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
        
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return img_padded, pad_h, pad_w


def predict_on_video(
    model: nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    transform=None,
    target_fps: int = 30,
    native_resolution: bool = False
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
        native_resolution: Si es True, mantiene la resolución original (padding si es necesario)
    
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
    
    logger.info(f"Total de frames: {total_frames} | Resolución: {width}x{height}")
    if native_resolution:
        logger.info("Modo: Resolución Nativa (se usará padding para ajustar a U-Net)")
    else:
        logger.info("Modo: Resize (se ajustará al tamaño de entrenamiento)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocesamiento
        if native_resolution:
            # Padding manual para cumplir requisitos de U-Net (divisible por 32)
            frame_padded, pad_h, pad_w = pad_to_divisor(frame_rgb, 32)
            
            # Normalización manual (para no depender de Resize en transform)
            frame_tensor = torch.from_numpy(frame_padded).permute(2, 0, 1).float() / 255.0
            
            # Aplicar normalización estándar de ImageNet si es necesario
            # Aquí asumimos simple /255.0 para simplificar, o usamos transform si se pasa uno especial
            # Idealmente usamos el transform solo para Normalize si existe
            if transform is not None:
                # Hack: Usamos transform solo para normalize si es posible, 
                # pero transform suele tener Resize.
                # Si native_resolution es True, ignoramos el transform de resize
                # y aplicamos solo normalización si pudiéramos.
                # Por ahora, normalización simple /255.0 suele funcionar bien con modelos entrenados con ImageNet stats
                # si se reentrena. Si no, deberíamos aplicar (x - mean) / std.
                # Vamos a aplicar la normalización estándar manualmente para ser robustos:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                frame_tensor = (frame_tensor - mean) / std
            
            frame_tensor = frame_tensor.unsqueeze(0) # [1, 3, H, W]
            
        else:
            # Modo Resize estándar
            if transform is not None:
                transformed = transform(image=frame_rgb)
                frame_tensor = transformed['image'].unsqueeze(0)
            else:
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Predicción
        with torch.no_grad():
            frame_tensor = frame_tensor.to(device)
            pred = model(frame_tensor)
            pred = pred.squeeze(0).cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            pred = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        
        # Post-procesamiento
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        
        if native_resolution:
            # Quitar padding (Crop)
            h_padded, w_padded = pred_bgr.shape[:2]
            # El padding se añadió abajo y a la derecha
            pred_bgr = pred_bgr[:height, :width]
        else:
            # Redimensionar al tamaño original si hubo resize
            if pred_bgr.shape[0] != height or pred_bgr.shape[1] != width:
                pred_bgr = cv2.resize(pred_bgr, (width, height))
            
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
