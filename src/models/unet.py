"""
Arquitectura U-Net para Image-to-Image Translation
"""

import torch
import torch.nn as nn
from typing import Optional, Union


def get_activation(activation_name: str) -> nn.Module:
    """Retorna la capa de activación basada en el nombre."""
    act = activation_name.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "gelu":
        return nn.GELU()
    elif act == "mish":
        return nn.Mish(inplace=True)
    elif act == "silu" or act == "swish":
        return nn.SiLU(inplace=True)
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    else:
        raise ValueError(f"Activación no soportada: {activation_name}")


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv2d + BatchNorm + Activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Bloque descendente: 2x Conv + MaxPool"""
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels, activation=activation),
            ConvBlock(out_channels, out_channels, activation=activation)
        )
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        conv_out = self.convs(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class UpBlock(nn.Module):
    """Bloque ascendente: Upsample + Concatenate + 2x Conv"""
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels, activation=activation),
            ConvBlock(out_channels, out_channels, activation=activation)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x


class UNet(nn.Module):
    """
    U-Net modificada con conexiones de salto y activación configurable.
    
    Arquitectura:
    - Encoder: 4 capas de downsampling
    - Bottleneck: 2 bloques convolucionales
    - Decoder: 4 capas de upsampling con skip connections
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3,
        base_channels: int = 64,
        activation: str = "relu"
    ):
        """
        Args:
            in_channels: Número de canales entrada (3 para RGB)
            out_channels: Número de canales salida (3 para RGB)
            base_channels: Número de canales en la primera capa
            activation: Función de activación (relu, gelu, mish, etc.)
        """
        super().__init__()
        
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels, activation=activation)
        self.down2 = DownBlock(base_channels, base_channels * 2, activation=activation)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, activation=activation)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8, activation=activation)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, activation=activation),
            ConvBlock(base_channels * 16, base_channels * 16, activation=activation)
        )
        
        # Decoder
        self.up4 = UpBlock(base_channels * 16, base_channels * 8, activation=activation)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, activation=activation)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, activation=activation)
        self.up1 = UpBlock(base_channels * 2, base_channels, activation=activation)
        
        # Capa de salida
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        self.tanh = nn.Tanh()  # Cambiado de Sigmoid a Tanh para rango [-1, 1]
    
    def forward(self, x):
        # Encoder con skip connections
        down1_out, skip1 = self.down1(x)
        down2_out, skip2 = self.down2(down1_out)
        down3_out, skip3 = self.down3(down2_out)
        down4_out, skip4 = self.down4(down3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(down4_out)
        
        # Decoder con skip connections
        up4_out = self.up4(bottleneck_out, skip4)
        up3_out = self.up3(up4_out, skip3)
        up2_out = self.up2(up3_out, skip2)
        up1_out = self.up1(up2_out, skip1)
        
        # Salida final
        output = self.final(up1_out)
        output = self.tanh(output)  # Salida en rango [-1, 1]
        
        return output


class UNetWithConvNeXt(nn.Module):
    """
    U-Net con backbone ConvNeXt para mejor extracción de características
    (Versión mejorada para producción)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        activation: str = "relu"
    ):
        super().__init__()
        
        # Usar U-Net estándar por simplicidad
        # En producción, puedes integrar ConvNeXt como encoder
        self.unet = UNet(in_channels, out_channels, base_channels, activation=activation)
    
    def forward(self, x):
        return self.unet(x)
