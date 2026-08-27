"""
Dataset personalizado para pares de imágenes (input/ground truth) con emparejamiento inteligente por número de frame
"""

import os
import re
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable, List, Dict


def extract_frame_number(filename: str) -> Optional[int]:
    """
    Extrae el número de frame de un nombre de archivo.
    En pipelines de VFX, el último grupo de dígitos corresponde al número de frame
    (ej: 'ToGT_EP8_VFX_041_24P_00124.png' -> 124, 'shot_input_45.jpg' -> 45).
    """
    nums = re.findall(r'\d+', os.path.splitext(filename)[0])
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def natural_sort_key(s: str):
    """Clave de ordenamiento natural (ordena 1, 2, 10 en lugar de 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class PairedImageDataset(Dataset):
    """
    Dataset inteligente para pares de imágenes (input y ground truth).
    
    Estrategias de emparejamiento automático:
    1. Match por número de frame extraído (ej: 'input_0124.jpg' <-> 'clean_0124.png')
    2. Match por nombre exacto o stem base (ej: 'frame_1.jpg' <-> 'frame_1.png')
    3. Match 1 a 1 por orden secuencial natural (si los nombres difieren completamente)
    """
    
    def __init__(
        self, 
        input_dir: str, 
        gt_dir: str,
        transform: Optional[Callable] = None,
        img_format: Optional[str] = None,
        mask_config: Optional[dict] = None
    ):
        self.input_dir = str(input_dir)
        self.gt_dir = str(gt_dir)
        self.transform = transform
        self.img_format = img_format
        self.mask_config = mask_config or {'enabled': False}
        
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Directorio de entrada no encontrado: {self.input_dir}")
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Directorio de Ground Truth no encontrado: {self.gt_dir}")

        valid_exts = (f".{img_format.lower()}",) if img_format else ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')
        
        # 1. Obtener todos los archivos válidos
        all_input_files = [
            f for f in os.listdir(self.input_dir)
            if any(f.lower().endswith(ext) for ext in valid_exts)
        ]
        all_gt_files = [
            f for f in os.listdir(self.gt_dir)
            if any(f.lower().endswith(ext) for ext in valid_exts)
        ]
        
        if not all_input_files:
            raise ValueError(f"No se encontraron imágenes válidas en el directorio input: {self.input_dir}")
        if not all_gt_files:
            raise ValueError(f"No se encontraron imágenes válidas en el directorio GT: {self.gt_dir}")

        # Ordenar naturalmente
        all_input_files = sorted(all_input_files, key=natural_sort_key)
        all_gt_files = sorted(all_gt_files, key=natural_sort_key)

        # 2. Estrategia 1: Match por número de frame
        gt_by_frame: Dict[int, str] = {}
        for f in all_gt_files:
            f_num = extract_frame_number(f)
            if f_num is not None and f_num not in gt_by_frame:
                gt_by_frame[f_num] = f

        pairs: List[Tuple[str, str]] = []
        matched_gt = set()

        for in_file in all_input_files:
            f_num = extract_frame_number(in_file)
            if f_num is not None and f_num in gt_by_frame:
                gt_file = gt_by_frame[f_num]
                pairs.append((
                    os.path.join(self.input_dir, in_file),
                    os.path.join(self.gt_dir, gt_file)
                ))
                matched_gt.add(gt_file)

        # 3. Estrategia 2: Match por stem (sin extensión) si no hubo match por número
        if len(pairs) == 0:
            gt_by_stem = {os.path.splitext(f)[0].lower(): f for f in all_gt_files}
            for in_file in all_input_files:
                in_stem = os.path.splitext(in_file)[0].lower()
                if in_stem in gt_by_stem:
                    gt_file = gt_by_stem[in_stem]
                    pairs.append((
                        os.path.join(self.input_dir, in_file),
                        os.path.join(self.gt_dir, gt_file)
                    ))
                    matched_gt.add(gt_file)

        # 4. Estrategia 3: Match 1 a 1 por orden natural secuencial (fallback)
        if len(pairs) == 0:
            min_count = min(len(all_input_files), len(all_gt_files))
            for i in range(min_count):
                pairs.append((
                    os.path.join(self.input_dir, all_input_files[i]),
                    os.path.join(self.gt_dir, all_gt_files[i])
                ))
            print(f"[Dataset] Emparejados {min_count} frames 1-a-1 por orden secuencial natural.")
        else:
            print(f"[Dataset] Emparejados exitosamente {len(pairs)} pares por número de frame / coincidencia.")

        if len(pairs) == 0:
            raise ValueError(
                f"No se pudieron emparejar archivos entre:\n"
                f"  Input ({len(all_input_files)} archivos): {self.input_dir}\n"
                f"  GT ({len(all_gt_files)} archivos): {self.gt_dir}"
            )

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        input_path, gt_path = self.pairs[idx]
        
        input_img = cv2.imread(input_path)
        gt_img = cv2.imread(gt_path)
        
        if input_img is None:
            raise FileNotFoundError(f"No se pudo cargar input: {input_path}")
        if gt_img is None:
            raise FileNotFoundError(f"No se pudo cargar GT: {gt_path}")
        
        # Convertir BGR a RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        # Generar máscara dinámica si está habilitada
        mask = None
        if self.mask_config.get('enabled', False):
            diff = cv2.absdiff(input_img, gt_img)
            diff = np.mean(diff, axis=2) / 255.0
            
            threshold = self.mask_config.get('threshold', 0.05)
            mask_bin = (diff > threshold).astype(np.float32)
            
            dilation_k = self.mask_config.get('dilation_kernel', 5)
            if dilation_k > 0:
                kernel = np.ones((dilation_k, dilation_k), np.uint8)
                mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
            
            blur_k = self.mask_config.get('blur_kernel', 5)
            if blur_k > 0:
                if blur_k % 2 == 0:
                    blur_k += 1
                mask_bin = cv2.GaussianBlur(mask_bin, (blur_k, blur_k), 0)
            
            mask = mask_bin[:, :, np.newaxis]
        
        # Aplicar transformaciones
        if self.transform is not None:
            if isinstance(self.transform, dict):
                if 'common' in self.transform:
                    if mask is not None:
                        transformed = self.transform['common'](image=input_img, image0=gt_img, mask=mask)
                        input_img = transformed['image']
                        gt_img = transformed['image0']
                        mask = transformed['mask']
                    else:
                        transformed = self.transform['common'](image=input_img, image0=gt_img)
                        input_img = transformed['image']
                        gt_img = transformed['image0']
                
                if 'input' in self.transform:
                    transformed_input = self.transform['input'](image=input_img)
                    input_img = transformed_input['image']
                
                if 'gt' in self.transform:
                    transformed_gt = self.transform['gt'](image=gt_img)
                    gt_img = transformed_gt['image']
            else:
                if mask is not None:
                    transformed = self.transform(image=input_img, image1=gt_img, mask=mask)
                    input_img = transformed['image']
                    gt_img = transformed['image1']
                    mask = transformed['mask']
                else:
                    transformed = self.transform(image=input_img, image1=gt_img)
                    input_img = transformed['image']
                    gt_img = transformed['image1']

        def _to_tensor(img):
            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                img = (img / 255.0) * 2.0 - 1.0
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
                img = torch.from_numpy(img).permute(2, 0, 1)
                return img
            if isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                img = img.float()
                if img.max() > 2.0:
                    img = (img / 255.0) * 2.0 - 1.0
                return img
            raise TypeError(f"Tipo de imagen no soportado: {type(img)}")

        input_tensor = _to_tensor(input_img) if not isinstance(input_img, torch.Tensor) else input_img
        gt_tensor = _to_tensor(gt_img) if not isinstance(gt_img, torch.Tensor) else gt_img

        result = {
            'input': input_tensor,
            'gt': gt_tensor,
            'filename': os.path.basename(input_path),
            'input_name': os.path.basename(input_path),
            'gt_name': os.path.basename(gt_path),
            'input_path': input_path,
            'gt_path': gt_path
        }

        
        if mask is not None:
            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = mask[:, :, np.newaxis]
                mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()
            else:
                mask_tensor = mask.float()
            result['mask'] = mask_tensor
            
        return result
