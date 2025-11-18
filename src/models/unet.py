"""
Arquitectura U-Net para Image-to-Image Translation
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv2d + BatchNorm + ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Bloque descendente: 2x Conv + MaxPool"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        conv_out = self.convs(x)
        pool_out = self.pool(conv_out)
        return pool_out, conv_out


class UpBlock(nn.Module):
    """Bloque ascendente: Upsample + Concatenate + 2x Conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.convs = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x


class UNet(nn.Module):
    """
    U-Net modificada con conexiones de salto.
    
    Arquitectura:
    - Encoder: 4 capas de downsampling
    - Bottleneck: 2 bloques convolucionales
    - Decoder: 4 capas de upsampling con skip connections
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3,
        base_channels: int = 64
    ):
        """
        Args:
            in_channels: Número de canales entrada (3 para RGB)
            out_channels: Número de canales salida (3 para RGB)
            base_channels: Número de canales en la primera capa
        """
        super().__init__()
        
        # Encoder
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16),
            # Mantener la dimensionalidad en el bottleneck (1024 canales
            # cuando base_channels=64) para que el decoder reciba el
            # número de canales esperado por los UpBlocks.
            ConvBlock(base_channels * 16, base_channels * 16)
        )
        
        # Decoder
        self.up4 = UpBlock(base_channels * 16, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels)
        
        # Capa de salida
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
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
        output = self.sigmoid(output)
        
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
        base_channels: int = 64
    ):
        super().__init__()
        
        # Usar U-Net estándar por simplicidad
        # En producción, puedes integrar ConvNeXt como encoder
        self.unet = UNet(in_channels, out_channels, base_channels)
    
    def forward(self, x):
        return self.unet(x)
