"""
Augmentaciones de datos optimizadas para VFX y Few-Shot Video Restoration
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Dict


def get_transforms(
    img_size: int = 512,
    augment: bool = True,
    aug_config: Optional[dict] = None
) -> dict:
    """
    Define transformaciones geométricas y normalización para pares de imágenes VFX.
    Por defecto, escala el frame completo (LongestMaxSize + Pad) para que el modelo
    vea toda la escena y el efecto (wire removal, de-aging, etc.) sin recortarlo aleatoriamente.
    """
    if aug_config is None:
        aug_config = {}

    mode = aug_config.get('mode', 'resize')
    geometric_transforms = []

    if mode == 'crop':
        geometric_transforms.append(A.RandomCrop(width=img_size, height=img_size, p=1.0))
    elif mode == 'direct_resize':
        geometric_transforms.append(A.Resize(height=img_size, width=img_size))
    else:
        # Modo VFX recomendado: Escalar manteniendo relación de aspecto y rellenar a múltiplo de 32
        geometric_transforms.extend([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=0
            )
        ])

    if augment:
        # Flips sutiles
        if aug_config.get('horizontal_flip_p', 0.5) > 0:
            geometric_transforms.append(A.HorizontalFlip(p=aug_config.get('horizontal_flip_p', 0.5)))
        if aug_config.get('vertical_flip_p', 0.0) > 0:
            geometric_transforms.append(A.VerticalFlip(p=aug_config.get('vertical_flip_p', 0.0)))
        
        # Rotación ligera si está configurada
        rot_limit = aug_config.get('rotation_limit', 0)
        if rot_limit > 0:
            geometric_transforms.append(A.Rotate(limit=rot_limit, p=0.5, border_mode=0))

    common_transforms = A.Compose(
        geometric_transforms,
        additional_targets={'image0': 'image', 'mask': 'mask'}
    )

    # Normalización a [-1, 1]
    input_norm = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    gt_norm = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    return {
        'common': common_transforms,
        'input': input_norm,
        'gt': gt_norm
    }


def get_inference_transforms(img_size: int = 512, resize: bool = True) -> A.Compose:
    """
    Transformaciones para inferencia (solo input) - Normalización a [-1, 1]
    """
    transforms_list = []
    
    if resize:
        transforms_list.extend([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(pad_height_divisor=32, pad_width_divisor=32, border_mode=0)
        ])
    else:
        transforms_list.append(
            A.PadIfNeeded(pad_height_divisor=32, pad_width_divisor=32, border_mode=0)
        )
        
    transforms_list.extend([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)
