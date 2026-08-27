"""Módulo de modelos de CopyAir"""
from .factory import get_model, get_optimizer
from .losses import HybridLoss
from .nafnet import nafnet_small, nafnet_base, nafnet_large, NAFNetHD
from .scope_unet import GatedUNet, gated_unet_small, gated_unet_base, scope_unet_base
from .restormer import restormer_tiny, restormer_small, restormer_base, Restormer
from .mambair import mambair_tiny, mambair_base, mambair_large, MambaIRv2
from .convnext import convnext_nano, convnext_tiny, convnext_small, convnext_base, ConvNeXtUNet
from .residual_unet import ResidualUNet
from .modern_unet import ModernUNet
from .smartunet import SmartUNet
from .basic_unet import BasicUNet
from .swin_unet import SwinV2UNet
from .unet import UNet

__all__ = [
    "get_model",
    "get_optimizer",
    "HybridLoss",
    "nafnet_small", "nafnet_base", "nafnet_large", "NAFNetHD",
    "gated_unet_small", "gated_unet_base", "scope_unet_base", "GatedUNet",
    "restormer_tiny", "restormer_small", "restormer_base", "Restormer",
    "mambair_tiny", "mambair_base", "mambair_large", "MambaIRv2",
    "convnext_nano", "convnext_tiny", "convnext_small", "convnext_base", "ConvNeXtUNet",
    "ResidualUNet",
    "ModernUNet",
    "SmartUNet",
    "BasicUNet",
    "SwinV2UNet",
    "UNet"
]
