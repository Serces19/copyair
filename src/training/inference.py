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
        frame: Frame normalizado a [-1, 1] con shape [1, 3, H, W]
        device: CPU o GPU
    
    Returns:
        Frame predicho como numpy array [H, W, 3] en rango [0, 255]
    """
    model.eval()
    
    with torch.no_grad():
        frame = frame.to(device)
        output = model(frame)  # Output en [-1, 1]
        output = output.squeeze(0).cpu().numpy()  # [3, H, W]
        output = np.transpose(output, (1, 2, 0))  # [H, W, 3]
        
        # Desnormalizar de [-1, 1] a [0, 255]
        output = ((output + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    
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
    native_resolution: bool = False,
    backend: str = 'opencv',
    lossless: bool = False
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
        backend: 'opencv' (seguro para color) o 'ffmpeg' (mejor compresión)
    
    Returns:
        Ruta del video generado
    """
    import subprocess
    import shutil
    
    logger.info(f"Procesando video: {video_path}")
    
    # Detectar si es directorio o archivo
    video_path_obj = Path(video_path)
    is_directory = video_path_obj.is_dir()
    
    if is_directory:
        # Modo Directorio de Imágenes
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = sorted([
            p for p in video_path_obj.glob('*')
            if p.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No se encontraron imágenes en {video_path}")
            
        logger.info(f"Modo Entrada: Directorio de Imágenes ({len(image_files)} frames)")
        
        # Leer primer frame para obtener dimensiones
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            raise ValueError(f"No se pudo leer la primera imagen: {image_files[0]}")
            
        height, width = first_frame.shape[:2]
        total_frames = len(image_files)
        fps = target_fps  # Usar target_fps ya que no hay FPS intrínseco
        
        # Generador de frames para directorio
        def frame_provider():
            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.warning(f"No se pudo leer frame: {img_path}")
                    continue
                yield True, frame
            yield False, None
            
    else:
        # Modo Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener propiedades
        fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generador de frames para video
        def frame_provider():
            while True:
                ret, frame = cap.read()
                yield ret, frame
            cap.release()
            yield False, None
            
    # Inicializar generador
    frames_iter = frame_provider()
    
    # Determinar tipo de salida: Video o Secuencia de Imágenes
    output_path_obj = Path(output_path)
    is_sequence = output_path_obj.suffix == '' or output_path.endswith('/') or output_path.endswith('\\')
    
    if is_sequence:
        # Modo Secuencia de Imágenes
        output_path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"Modo Salida: Secuencia de Imágenes (PNG) en {output_path}")
        use_ffmpeg = False
        out = None
    else:
        # Modo Video
        use_ffmpeg = False
        
        # Configurar backend
        if backend == 'ffmpeg' and shutil.which('ffmpeg') is not None:
            # Configuración para FFmpeg
            # Si es lossless, usamos CRF 0 y flags de color estrictos
            crf_val = '0' if lossless else '18'
            preset = 'veryslow' if crf_val == '0' else 'medium'
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', crf_val,
                '-preset', preset,
            ]
            
            # Flags de color para lossless/alta fidelidad
            if crf_val == '0':
                ffmpeg_cmd.extend([
                    '-color_range', 'tv',
                    '-colorspace', 'bt709',
                    '-color_primaries', 'bt709',
                    '-color_trc', 'bt709'
                ])
                logger.info("Modo FFmpeg: LOSSLESS (CRF 0) + Color BT.709")
            else:
                logger.info(f"Modo FFmpeg: Alta Calidad (CRF {crf_val})")
                
            ffmpeg_cmd.append(output_path)
            
            try:
                process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                use_ffmpeg = True
            except Exception as e:
                logger.warning(f"Error al iniciar FFmpeg: {e}, cayendo a OpenCV")
                use_ffmpeg = False
        
        if not use_ffmpeg:
            # Backend OpenCV
            logger.info("Usando backend: OpenCV")
            # Intentar codec avc1 (H.264) primero, luego mp4v
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise Exception("avc1 no soportado")
                logger.info("Codec: avc1 (H.264)")
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info("Codec: mp4v (Fallback)")
    
    model.eval()
    frame_count = 0
    
    logger.info(f"Total de frames: {total_frames} | Resolución: {width}x{height}")
    if native_resolution:
        logger.info("Modo: Resolución Nativa (se usará padding para ajustar a U-Net)")
    else:
        logger.info("Modo: Resize (se ajustará al tamaño de entrenamiento)")
    
    while True:
        ret, frame = next(frames_iter)
        if not ret:
            break
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocesamiento
        if native_resolution:
            # Padding manual para cumplir requisitos de U-Net (divisible por 32)
            frame_padded, pad_h, pad_w = pad_to_divisor(frame_rgb, 32)
            
            # Normalizar a [-1, 1]
            frame_tensor = torch.from_numpy(frame_padded).permute(2, 0, 1).float()
            frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0
            frame_tensor = frame_tensor.unsqueeze(0)
            
        else:
            # Modo Resize estándar
            if transform is not None:
                transformed = transform(image=frame_rgb)
                frame_tensor = transformed['image'].unsqueeze(0)
            else:
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
                frame_tensor = (frame_tensor / 255.0) * 2.0 - 1.0
 
        # Predicción
        with torch.no_grad():
            frame_tensor = frame_tensor.to(device)
            pred = model(frame_tensor)
            pred = pred.squeeze(0).cpu().numpy()
            pred = np.transpose(pred, (1, 2, 0))
            
            # Desnormalizar de [-1, 1] a [0, 255]
            pred = ((pred + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        # Post-procesamiento
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        
        if native_resolution:
            # Quitar padding (Crop)
            pred_bgr = pred_bgr[:height, :width]
        else:
            # Redimensionar al tamaño original si hubo resize
            if pred_bgr.shape[0] != height or pred_bgr.shape[1] != width:
                pred_bgr = cv2.resize(pred_bgr, (width, height))
        
        # Escribir salida
        if is_sequence:
            # Guardar frame individual (PNG es lossless)
            frame_name = f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(output_path_obj / frame_name), pred_bgr)
        elif use_ffmpeg:
            try:
                process.stdin.write(pred_bgr.tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe roto")
                break
        else:
            out.write(pred_bgr)
        
        frame_count += 1
        if (frame_count) % 10 == 0:
            logger.info(f"Procesados {frame_count}/{total_frames} frames")
    
    cap.release()
    
    if not is_sequence:
        if use_ffmpeg:
            process.stdin.close()
            process.wait()
            if process.returncode != 0:
                stderr = process.stderr.read().decode()
                logger.error(f"FFmpeg error: {stderr}")
        else:
            out.release()
        logger.info(f"Video guardado en: {output_path}")
    else:
        logger.info(f"Secuencia guardada en: {output_path}")
        
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
