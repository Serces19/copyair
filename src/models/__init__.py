"""Módulo de modelos"""
from .unet import UNet
from .losses import HybridLoss

__all__ = ["UNet", "HybridLoss"]
