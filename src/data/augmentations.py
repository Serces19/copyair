"""
Augmentaciones y transformaciones de datos para CopyAir
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
    Transformaciones estándar para pares de entrenamiento (Input / GT).
    Redimensiona directamente a img_size x img_size y normaliza a [-1, 1].
    """
    if aug_config is None:
        aug_config = {}

    geometric_transforms = [
        A.Resize(height=img_size, width=img_size)
    ]

    if augment:
        if aug_config.get('horizontal_flip_p', 0.5) > 0:
            geometric_transforms.append(A.HorizontalFlip(p=aug_config.get('horizontal_flip_p', 0.5)))
        if aug_config.get('vertical_flip_p', 0.0) > 0:
            geometric_transforms.append(A.VerticalFlip(p=aug_config.get('vertical_flip_p', 0.0)))
        rot_limit = aug_config.get('rotation_limit', 0)
        if rot_limit > 0:
            geometric_transforms.append(A.Rotate(limit=rot_limit, p=0.5))

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
        transforms_list.append(A.Resize(height=img_size, width=img_size))
        
    transforms_list.extend([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)
