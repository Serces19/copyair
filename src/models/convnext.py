"""
ConvNeXt para Image-to-Image Translation
Usando backbone pre-entrenado (ImageNet-22K) via `timm`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List

class ResidualBlock(nn.Module):
    """Bloque Residual para el Decoder"""
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = self._get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def _get_activation(self, name):
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'gelu': return nn.GELU()
        return nn.GELU() # Default to GELU

    def forward(self, x):
        return self.act(self.conv2(self.act(self.bn1(self.conv1(x)))) + self.shortcut(x))


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt U-Net usando backbone de `timm`.
    Soporta pesos pre-entrenados ImageNet-22K (fb_in22k).
    
    Mejoras:
    - Resize-Convolution (Nearest Neighbor + Conv) para nitidez.
    - Global Skip Connection (Residual Learning).
    - Activaciones GELU consistentes.
    """
    def __init__(
        self,
        backbone_name: str,
        in_channels: int = 3,
        out_channels: int = 3,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        use_transpose: bool = False # Ignorado
    ):
        super().__init__()
        
        # 1. Encoder (timm)
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
            in_chans=in_channels
        )
        
        # Obtener canales dinámicamente
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 224, 224)
            features = self.encoder(dummy)
            dims = [f.shape[1] for f in features]
            
        self.dims = dims
        
        # 2. Decoder
        # Usamos Nearest Neighbor Upsample para preservar bordes (crispness)
        
        # Up3: dims[3] -> dims[2] size
        self.up3_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[3], dims[2], kernel_size=3, padding=1)
        )
        self.dec3 = ResidualBlock(dims[2] + dims[2], dims[2], activation='gelu')
        
        # Up2: dims[2] -> dims[1] size
        self.up2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[2], dims[1], kernel_size=3, padding=1)
        )
        self.dec2 = ResidualBlock(dims[1] + dims[1], dims[1], activation='gelu')
        
        # Up1: dims[1] -> dims[0] size
        self.up1_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[1], dims[0], kernel_size=3, padding=1)
        )
        self.dec1 = ResidualBlock(dims[0] + dims[0], dims[0], activation='gelu')
        
        # Up0: dims[0] -> Original size
        
        # Paso 1: stride 4 -> stride 2
        mid_channels = dims[0] // 2
        self.up0a_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[0], mid_channels, kernel_size=3, padding=1)
        )
        self.dec0a = ResidualBlock(mid_channels, mid_channels, activation='gelu')
        
        # Paso 2: stride 2 -> stride 1
        final_channels = mid_channels // 2
        self.up0b_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(mid_channels, final_channels, kernel_size=3, padding=1)
        )
        self.dec0b = ResidualBlock(final_channels, final_channels, activation='gelu')
        
        # Head
        self.head = nn.Conv2d(final_channels, out_channels, kernel_size=1)
        # No usamos Tanh aquí porque haremos la suma residual primero
        
        # Normalización ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _pad_to_match(self, x, target):
        if x.shape[-2:] != target.shape[-2:]:
            diffY = target.size()[2] - x.size()[2]
            diffX = target.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x):
        # Guardar input original para Global Skip Connection
        x_input = x
        
        # 1. Input Handling: [-1, 1] -> ImageNet Norm
        x_norm = (x + 1) * 0.5
        x_norm = (x_norm - self.mean) / self.std
        
        # 2. Encoder
        features = self.encoder(x_norm)
        f0, f1, f2, f3 = features
        
        # 3. Decoder
        # Up3
        d3 = self.up3_conv(f3)
        d3 = self._pad_to_match(d3, f2)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.dec3(d3)
        
        # Up2
        d2 = self.up2_conv(d3)
        d2 = self._pad_to_match(d2, f1)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec2(d2)
        
        # Up1
        d1 = self.up1_conv(d2)
        d1 = self._pad_to_match(d1, f0)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.dec1(d1)
        
        # Up0
        d0 = self.up0a_conv(d1)
        d0 = self.dec0a(d0)
        
        d0 = self.up0b_conv(d0)
        if d0.shape[-2:] != x.shape[-2:]:
            d0 = self._pad_to_match(d0, x)
        d0 = self.dec0b(d0)
        
        # Head (Residual)
        residual = self.head(d0)
        
        # Global Skip Connection: Output = Input + Residual
        # Esto facilita aprender solo los detalles que cambian
        out = x_input + residual
        
        # Clamp final para asegurar rango válido [-1, 1]
        out = torch.clamp(out, -1.0, 1.0)
        
        return out


# --- Factory Wrappers ---

def convnext_tiny(in_channels=3, out_channels=3, drop_path_rate=0.1, use_transpose=False):
    """ConvNeXt-Tiny (ImageNet-22K)"""
    return ConvNeXtUNet('convnext_tiny.fb_in22k', in_channels, out_channels, 
                        drop_path_rate=drop_path_rate)

def convnext_small(in_channels=3, out_channels=3, drop_path_rate=0.15, use_transpose=False):
    """ConvNeXt-Small (ImageNet-22K)"""
    return ConvNeXtUNet('convnext_small.fb_in22k', in_channels, out_channels, 
                        drop_path_rate=drop_path_rate)

def convnext_base(in_channels=3, out_channels=3, drop_path_rate=0.2, use_transpose=False):
    """ConvNeXt-Base (ImageNet-22K)"""
    return ConvNeXtUNet('convnext_base.fb_in22k', in_channels, out_channels, 
                        drop_path_rate=drop_path_rate)

def convnext_nano(in_channels=3, out_channels=3, drop_path_rate=0.05, use_transpose=False):
    """ConvNeXt-Nano (ImageNet-1K) - No hay 22k oficial para nano"""
    return ConvNeXtUNet('convnext_nano', in_channels, out_channels, 
                        drop_path_rate=drop_path_rate)
