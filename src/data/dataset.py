"""
Dataset personalizado para pares de imágenes (input/ground truth)
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable


class PairedImageDataset(Dataset):
    """
    Dataset para cargar pares de imágenes (input y ground truth).
    
    Estructura esperada:
    data/03_processed/
    ├── input/
    │   ├── img_1.jpg
    │   └── img_2.jpg
    └── ground_truth/
        ├── img_1.jpg
        └── img_2.jpg
    """
    
    def __init__(
        self, 
        input_dir: str, 
        gt_dir: str,
        transform: Optional[Callable] = None,
        img_format: str = "png"
    ):
        """
        Args:
            input_dir: Ruta al directorio con imágenes de entrada
            gt_dir: Ruta al directorio con imágenes ground truth
            transform: Transformaciones a aplicar (albumentations)
            img_format: Formato de imagen (jpg, png, etc.)
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.img_format = img_format
        
        # Cargar lista de archivos
        self.img_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.endswith(f".{img_format}")
        ])
        
        if len(self.img_files) == 0:
            raise ValueError(f"No se encontraron imágenes en {input_dir}")
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retorna un diccionario con:
        {
            'input': tensor de imagen entrada,
            'gt': tensor de imagen ground truth
        }
        """
        img_name = self.img_files[idx]
        
        # Cargar imágenes
        input_path = os.path.join(self.input_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        
        input_img = cv2.imread(input_path)
        gt_img = cv2.imread(gt_path)
        
        if input_img is None or gt_img is None:
            raise FileNotFoundError(f"No se pudo cargar: {img_name}")
        
        # Convertir BGR a RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformaciones
        if self.transform is not None:
            # Nuevo formato: diccionario con 'common', 'input', 'gt'
            if isinstance(self.transform, dict):
                # 1. Transformaciones geométricas (ambas imágenes)
                if 'common' in self.transform:
                    transformed = self.transform['common'](image=input_img, image0=gt_img)
                    input_img = transformed['image']
                    gt_img = transformed['image0']
                
                # 2. Transformaciones de input (pixel-level + normalize)
                if 'input' in self.transform:
                    transformed_input = self.transform['input'](image=input_img)
                    input_img = transformed_input['image']
                
                # 3. Transformación de GT (solo ToTensor, sin normalize)
                if 'gt' in self.transform:
                    transformed_gt = self.transform['gt'](image=gt_img)
                    gt_img = transformed_gt['image']
            else:
                # Formato antiguo (backward compatibility)
                transformed = self.transform(image=input_img, image1=gt_img)
                input_img = transformed['image']
                gt_img = transformed['image1']

        # Convertir a tensores si aún no lo son
        def _to_tensor(img):
            """Fallback para convertir a tensor si transforms no lo hicieron"""
            if isinstance(img, np.ndarray):
                # Normalizar a [-1, 1]
                img = img.astype(np.float32)
                img = (img / 255.0) * 2.0 - 1.0  # [0, 255] → [-1, 1]
                img = torch.from_numpy(img).permute(2, 0, 1)
                return img
            if isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                img = img.float()
                # Si está en rango [0, 255], normalizar a [-1, 1]
                if img.max() > 2.0:
                    img = (img / 255.0) * 2.0 - 1.0
                return img
            raise TypeError(f"Tipo de imagen no soportado: {type(img)}")

        # Solo convertir si no es tensor (transforms ya lo convirtieron)
        if not isinstance(input_img, torch.Tensor):
            input_tensor = _to_tensor(input_img)
        else:
            input_tensor = input_img
            
        if not isinstance(gt_img, torch.Tensor):
            gt_tensor = _to_tensor(gt_img)
        else:
            gt_tensor = gt_img

        return {
            'input': input_tensor,
            'gt': gt_tensor,
            'filename': img_name
        }


class VideoFrameDataset(Dataset):
    """
    Dataset para inferencia: carga frames de un video
    """
    
    def __init__(
        self,
        frames_dir: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            frames_dir: Ruta al directorio con frames extraídos
            transform: Transformaciones a aplicar
        """
        self.frames_dir = frames_dir
        self.transform = transform
        
        self.frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
    
    def __len__(self) -> int:
        return len(self.frame_files)
    
    def __getitem__(self, idx: int) -> dict:
        frame_name = self.frame_files[idx]
        frame_path = os.path.join(self.frames_dir, frame_name)
        
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            transformed = self.transform(image=frame)
            frame = transformed['image']

        # Convertir a tensor CHW float en [-1, 1] (fallback si transform no lo hizo)
        if isinstance(frame, np.ndarray):
            frame = frame.astype(np.float32)
            frame = (frame / 255.0) * 2.0 - 1.0  # [0, 255] → [-1, 1]
            frame = torch.from_numpy(frame).permute(2, 0, 1)
        elif isinstance(frame, torch.Tensor):
            if frame.ndim == 3 and frame.shape[0] != 3 and frame.shape[-1] == 3:
                frame = frame.permute(2, 0, 1)
            frame = frame.float()
            if frame.max() > 2.0:
                frame = (frame / 255.0) * 2.0 - 1.0
        else:
            raise TypeError(f"Tipo de frame no soportado: {type(frame)}")

        return {
            'frame': frame,
            'filename': frame_name
        }
