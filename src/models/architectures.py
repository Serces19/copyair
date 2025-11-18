"""
Nuevas arquitecturas avanzadas para Image Restoration / Translation.
Nota: Estas son implementaciones base/placeholder que siguen la interfaz requerida.
Para producción, se deben rellenar con las implementaciones completas de los papers respectivos.
"""

import torch
import torch.nn as nn
from .unet import UNet

class ConvexNet(nn.Module):
    """
    Placeholder para ConvexNet.
    Paper: "ConvexNet: Convex Optimization for Image Restoration" (Conceptual)
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, activation="relu"):
        super().__init__()
        # Por ahora usamos una U-Net como base, pero aquí iría la implementación real
        self.model = UNet(in_channels, out_channels, base_channels, activation)
        
    def forward(self, x):
        return self.model(x)

class MambaIRv2(nn.Module):
    """
    Placeholder para MambaIR v2.
    Basado en State Space Models (SSM) para restauración de imágenes.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, activation="silu"):
        super().__init__()
        # Mamba suele usar SiLU/Swish
        self.model = UNet(in_channels, out_channels, base_channels, activation="silu")
        
    def forward(self, x):
        return self.model(x)

class UMamba(nn.Module):
    """
    Placeholder para U-Mamba.
    Hybrid CNN-SSM architecture.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, activation="silu"):
        super().__init__()
        self.model = UNet(in_channels, out_channels, base_channels, activation="silu")
        
    def forward(self, x):
        return self.model(x)

class NAFNetHD(nn.Module):
    """
    Placeholder para NAFNet (Nonlinear Activation Free Network).
    SOTA en deblurring y denoising.
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, activation="gelu"):
        super().__init__()
        # NAFNet usa GELU o SimpleGate
        self.model = UNet(in_channels, out_channels, base_channels, activation="gelu")
        
    def forward(self, x):
        return self.model(x)
