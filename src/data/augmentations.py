"""
Augmentaciones de datos usando albumentations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_transforms(
    img_size: int = 256,
    augment: bool = True,
    num_targets: int = 1
) -> Optional[A.Compose]:
    """
    Define transformaciones para entrenamiento/validación
    
    Args:
        img_size: Tamaño de imagen destino
        augment: Si aplicar augmentaciones
        num_targets: Número de imágenes adicionales a aumentar
    
    Returns:
        Composición de augmentaciones
    """
    
    if augment:
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={
            f'image{i}': 'image' for i in range(1, num_targets + 1)
        })
    else:
        transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={
            f'image{i}': 'image' for i in range(1, num_targets + 1)
        })
    
    return transforms


def get_inference_transforms(img_size: int = 256) -> A.Compose:
    """Transformaciones para inferencia (sin augmentación)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
