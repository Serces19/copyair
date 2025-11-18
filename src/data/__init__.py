"""Módulo de procesamiento de datos"""
from .dataset import PairedImageDataset
from .augmentations import get_transforms

__all__ = ["PairedImageDataset", "get_transforms"]
