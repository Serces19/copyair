"""
Augmentaciones de datos usando albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_transforms(
    img_size: int = 256,
    augment: bool = True
) -> dict:
    """
    Define transformaciones separadas para geometría (ambas imágenes) y normalización.
    
    IMPORTANTE: Para pares de imágenes (input/GT) donde la única diferencia es el efecto
    deseado (ej: quitar arrugas), NO debemos aplicar noise/blur de forma diferente.
    Solo geometric augmentations se aplican a ambas para mantener alignment.
    
    Normalizamos a [-1, 1] que es el estándar para image-to-image translation.
    
    Args:
        img_size: Tamaño de imagen destino
        augment: Si aplicar augmentaciones geométricas
    
    Returns:
        Diccionario con 'common' (geometric), 'input' y 'gt' (normalización)
    """
    
    # Transformaciones geométricas (aplicar a Input y GT por igual)
    if augment:
        common_transforms = A.Compose([
            A.RandomCrop(width=img_size, height=img_size, p=1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
        ], additional_targets={'image0': 'image'})
    else:
        common_transforms = A.Compose([
            A.RandomCrop(width=img_size, height=img_size, p=1),
        ], additional_targets={'image0': 'image'})

    # Normalización a [-1, 1] para Input
    # Fórmula: (pixel / 255.0) * 2 - 1  →  [0, 255] → [0, 1] → [-1, 1]
    input_norm = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],  # Centra en 0
            std=[0.5, 0.5, 0.5],   # Escala a [-1, 1]
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    # GT también a [-1, 1] (mismo rango que el output del modelo)
    gt_norm = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    return {
        'common': common_transforms,      # Geometric (ambos)
        'input': input_norm,               # Normalize a [-1, 1]
        'gt': gt_norm                      # Normalize a [-1, 1]
    }


def get_inference_transforms(img_size: int = 256, resize: bool = True) -> A.Compose:
    """
    Transformaciones para inferencia (solo input) - Normalización a [-1, 1]
    
    Args:
        img_size: Tamaño destino si resize=True
        resize: Si es False, NO redimensiona (útil para inferencia en resolución nativa)
    """
    transforms_list = []
    
    if resize:
        transforms_list.append(A.Resize(img_size, img_size))
        
    transforms_list.extend([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)
