"""
Dataset personalizado para pares de imágenes (input/ground truth)
"""

import os
import cv2
import numpy as np
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
        img_format: str = "jpg"
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
        
        # Aplicar transformaciones si existen
        if self.transform is not None:
            transformed = self.transform(image=input_img, image1=gt_img)
            input_img = transformed['image']
            gt_img = transformed['image1']
        
        return {
            'input': input_img.astype(np.float32) / 255.0,
            'gt': gt_img.astype(np.float32) / 255.0,
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
        
        return {
            'frame': frame.astype(np.float32) / 255.0,
            'filename': frame_name
        }
