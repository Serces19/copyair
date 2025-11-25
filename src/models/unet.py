import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Union


class ResidualBlock(nn.Module):
    """Bloque Residual para el Decoder"""
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = self._get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut para igualar dimensiones si es necesario
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def _get_activation(self, name):
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'leaky_relu': return nn.LeakyReLU(0.2, inplace=True)
        if name == 'gelu': return nn.GELU()
        return nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.act(out)
        return out


class UNet(nn.Module):
    """
    U-Net con backbone ResNet34 pre-entrenado.
    
    Mejoras:
    1. Encoder: ResNet34 (ImageNet weights) -> Mejor extracción de features.
    2. Decoder: PixelShuffle + ResidualBlocks -> Mejor reconstrucción y gradientes.
    3. Input Handling: Acepta [-1, 1], convierte internamente a normalización ImageNet.
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3,
        base_channels: int = 64, # No usado directamente en ResNet, pero mantenido por compatibilidad
        activation: str = "relu",
        use_batchnorm: bool = True,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
        use_transpose: bool = False # Ignorado, usamos PixelShuffle
    ):
        super().__init__()
        
        # --- Encoder (ResNet34) ---
        # Usamos weights='DEFAULT' que equivale a los mejores pesos disponibles
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Si in_channels != 3, adaptamos la primera capa
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = resnet.layer1 # 64
        self.enc2 = resnet.layer2 # 128
        self.enc3 = resnet.layer3 # 256
        self.enc4 = resnet.layer4 # 512
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # --- Decoder ---
        # Usamos PixelShuffle para upsampling (factor 2)
        # PixelShuffle reduce canales en factor 4 (r^2), así que necesitamos convs previas
        
        # Bottleneck -> Up4
        # Input: 512 (enc4). Concat con 256 (enc3).
        self.up4_conv = nn.Conv2d(512, 256 * 4, kernel_size=1) # Prep for PixelShuffle
        self.up4_ps = nn.PixelShuffle(2) # 256*4 -> 256
        self.dec4 = ResidualBlock(256 + 256, 256, activation) # 256 (up) + 256 (skip)
        
        # Up3
        # Input: 256. Concat con 128 (enc2).
        self.up3_conv = nn.Conv2d(256, 128 * 4, kernel_size=1)
        self.up3_ps = nn.PixelShuffle(2)
        self.dec3 = ResidualBlock(128 + 128, 128, activation)
        
        # Up2
        # Input: 128. Concat con 64 (enc1).
        self.up2_conv = nn.Conv2d(128, 64 * 4, kernel_size=1)
        self.up2_ps = nn.PixelShuffle(2)
        self.dec2 = ResidualBlock(64 + 64, 64, activation)
        
        # Up1
        # Input: 64. Concat con 64 (enc0).
        self.up1_conv = nn.Conv2d(64, 64 * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(2)
        self.dec1 = ResidualBlock(64 + 64, 64, activation)
        
        # Final Upsample (para recuperar resolución original tras el primer conv stride=2 y maxpool)
        # ResNet reduce x4 en las primeras capas (conv1 stride 2 + maxpool stride 2)
        # Hemos hecho 4 upsamples, pero necesitamos asegurar el tamaño final.
        # enc0 es H/2. enc1 es H/4. enc2 H/8. enc3 H/16. enc4 H/32.
        # up4 (from enc4) -> H/16.
        # up3 -> H/8.
        # up2 -> H/4.
        # up1 -> H/2.
        # Falta un upsample final para llegar a H.
        
        self.up0_conv = nn.Conv2d(64, 32 * 4, kernel_size=1)
        self.up0_ps = nn.PixelShuffle(2)
        self.dec0 = ResidualBlock(32, 32, activation)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        
        # Normalización ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # 1. Input Handling: [-1, 1] -> ImageNet Norm
        # Denormalize [-1, 1] -> [0, 1]
        x_norm = (x + 1) * 0.5
        # Normalize with ImageNet stats
        x_norm = (x_norm - self.mean) / self.std
        
        # 2. Encoder
        # x: (B, 3, H, W)
        e0 = self.enc0(x_norm)      # (B, 64, H/2, W/2)
        e0_pool = self.pool(e0)     # (B, 64, H/4, W/4)
        
        e1 = self.enc1(e0_pool)     # (B, 64, H/4, W/4) -> Layer1 no cambia dimensión espacial
        e2 = self.enc2(e1)          # (B, 128, H/8, W/8)
        e3 = self.enc3(e2)          # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)          # (B, 512, H/32, W/32)
        
        # 3. Decoder
        # Up4: e4 -> e3 size
        d4 = self.up4_ps(self.up4_conv(e4)) # (B, 256, H/16, W/16)
        d4 = torch.cat([d4, e3], dim=1)     # (B, 512, ...)
        d4 = self.dec4(d4)                  # (B, 256, ...)
        
        # Up3: d4 -> e2 size
        d3 = self.up3_ps(self.up3_conv(d4)) # (B, 128, H/8, W/8)
        d3 = torch.cat([d3, e2], dim=1)     # (B, 256, ...)
        d3 = self.dec3(d3)                  # (B, 128, ...)
        
        # Up2: d3 -> e1 size
        d2 = self.up2_ps(self.up2_conv(d3)) # (B, 64, H/4, W/4)
        d2 = torch.cat([d2, e1], dim=1)     # (B, 128, ...)
        d2 = self.dec2(d2)                  # (B, 64, ...)
        
        # Up1: d2 -> e0 size
        d1 = self.up1_ps(self.up1_conv(d2)) # (B, 64, H/2, W/2)
        # Nota: e0 tiene 64 canales.
        d1 = torch.cat([d1, e0], dim=1)     # (B, 128, ...)
        d1 = self.dec1(d1)                  # (B, 64, ...)
        
        # Up0: d1 -> Original size
        d0 = self.up0_ps(self.up0_conv(d1)) # (B, 32, H, W)
        d0 = self.dec0(d0)
        
        # Output
        out = self.final(d0)
        out = self.tanh(out)
        
        return out


class UNetWithConvNeXt(nn.Module):
    """
    Wrapper para mantener compatibilidad si se usa en configs antiguas.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        activation: str = "relu"
    ):
        super().__init__()
        self.unet = UNet(in_channels, out_channels, base_channels, activation=activation)
    
    def forward(self, x):
        return self.unet(x)
