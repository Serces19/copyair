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
    Define transformaciones separadas para geometría (ambas imágenes) y píxeles (solo input).
    
    Args:
        img_size: Tamaño de imagen destino
        augment: Si aplicar augmentaciones
    
    Returns:
        Diccionario con 'common' (geometric) y 'input' (pixel/norm) transforms
    """
    
    # Transformaciones geométricas (aplicar a Input y GT)
    if augment:
        common_transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
        ], additional_targets={'image0': 'image'})
    else:
        common_transforms = A.Compose([
            A.Resize(img_size, img_size),
        ], additional_targets={'image0': 'image'})

    # Transformaciones de píxeles y normalización (aplicar SOLO a Input)
    if augment:
        input_transforms = A.Compose([
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        input_transforms = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
    # Transformación para GT (solo ToTensor, sin normalizar mean/std)
    gt_transforms = A.Compose([
        A.ToFloat(max_value=255.0),  # Convert uint8 to float [0, 1]
        ToTensorV2(),
    ])
    
    return {
        'common': common_transforms,
        'input': input_transforms,
        'gt': gt_transforms
    }


def get_inference_transforms(img_size: int = 256) -> A.Compose:
    """Transformaciones para inferencia (solo input)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
