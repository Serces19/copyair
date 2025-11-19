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
    Define transformaciones separadas para geometría (ambas imágenes) y normalización (solo input).
    
    IMPORTANTE: Para pares de imágenes (input/GT) donde la única diferencia es el efecto
    deseado (ej: quitar arrugas), NO debemos aplicar noise/blur de forma diferente.
    Solo geometric augmentations se aplican a ambas para mantener alignment.
    
    Args:
        img_size: Tamaño de imagen destino
        augment: Si aplicar augmentaciones geométricas
    
    Returns:
        Diccionario con 'common' (geometric) y 'input_norm' (normalización input)
    """
    
    # Transformaciones geométricas (aplicar a Input y GT por igual)
    if augment:
        common_transforms = A.Compose([
            A.RandomCrop(width=img_size, height=img_size, p=1),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
            # NO agregamos noise, blur, o color jitter aquí porque:
            # - Input ya tiene las imperfecciones reales (arrugas, marcas, etc.)
            # - GT tiene las mismas condiciones de luz/noise, solo sin las imperfecciones
            # - Si agregamos noise diferente, el modelo aprende a "quitar el noise artificial"
            #   en lugar de aprender a quitar las imperfecciones reales
        ], additional_targets={'image0': 'image'})
    else:
        common_transforms = A.Compose([
            A.RandomCrop(width=img_size, height=img_size, p=1),
        ], additional_targets={'image0': 'image'})

    # Normalización solo para Input (ImageNet stats para transfer learning)
    input_norm = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    # GT solo necesita convertirse a tensor float [0, 1]
    gt_to_tensor = A.Compose([
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])
    
    return {
        'common': common_transforms,      # Geometric (ambos)
        'input': input_norm,               # Normalize (solo input)
        'gt': gt_to_tensor                 # ToTensor (solo GT)
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
