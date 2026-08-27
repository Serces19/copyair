"""
ConvNeXt para Image-to-Image Translation
Usando backbone pre-entrenado (ImageNet-22K) via `timm`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class LayerNorm2d(nn.Module):
    """LayerNorm aplicado a (B, C, H, W)"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ResidualBlock(nn.Module):
    """Bloque Residual para el Decoder con GRN y LayerNorm2d (ConvNeXt style)"""
    def __init__(self, in_channels, out_channels, activation='gelu'):
        super().__init__()
        
        # Como ConvNeXt: Depthwise -> LayerNorm -> Pointwise -> GELU -> GRN -> Pointwise
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.norm = LayerNorm2d(in_channels)
        self.pw1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = self._get_activation(activation)
        self.grn = GRN(out_channels)
        self.pw2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Shortcut
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                LayerNorm2d(out_channels)
            )

    def _get_activation(self, name):
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'gelu': return nn.GELU()
        return nn.GELU()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return x + res


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt U-Net usando backbone de `timm`.
    Soporta pesos pre-entrenados ImageNet-22K (fb_in22k).
    """
    def __init__(
        self,
        backbone_name: str,
        in_channels: int = 3,
        out_channels: int = 3,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        use_transpose: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
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
        
        # Global residual proj si los canales no coinciden
        self.global_proj = nn.Identity()
        if in_channels != out_channels:
            self.global_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
        # 2. Decoder
        self.up3_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[3], dims[2], kernel_size=3, padding=1)
        )
        self.dec3 = ResidualBlock(dims[2] + dims[2], dims[2], activation='gelu')
        
        self.up2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[2], dims[1], kernel_size=3, padding=1)
        )
        self.dec2 = ResidualBlock(dims[1] + dims[1], dims[1], activation='gelu')
        
        self.up1_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[1], dims[0], kernel_size=3, padding=1)
        )
        self.dec1 = ResidualBlock(dims[0] + dims[0], dims[0], activation='gelu')
        
        mid_channels = dims[0] // 2
        self.up0a_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dims[0], mid_channels, kernel_size=3, padding=1)
        )
        self.dec0a = ResidualBlock(mid_channels, mid_channels, activation='gelu')
        
        final_channels = mid_channels // 2
        self.up0b_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(mid_channels, final_channels, kernel_size=3, padding=1)
        )
        self.dec0b = ResidualBlock(final_channels, final_channels, activation='gelu')
        
        self.head = nn.Conv2d(final_channels, out_channels, kernel_size=1)
        
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
        x_input = x
        
        # 1. Input Handling: [-1, 1] -> ImageNet Norm
        if self.in_channels == 3:
            x_norm = (x + 1) * 0.5
            x_norm = (x_norm - self.mean) / self.std
        else:
            x_norm = x
            
        # 2. Encoder
        features = self.encoder(x_norm)
        f0, f1, f2, f3 = features
        
        # 3. Decoder
        d3 = self.up3_conv(f3)
        d3 = self._pad_to_match(d3, f2)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2_conv(d3)
        d2 = self._pad_to_match(d2, f1)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1_conv(d2)
        d1 = self._pad_to_match(d1, f0)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up0a_conv(d1)
        d0 = self.dec0a(d0)
        
        d0 = self.up0b_conv(d0)
        if d0.shape[-2:] != x.shape[-2:]:
            d0 = self._pad_to_match(d0, x)
        d0 = self.dec0b(d0)
        
        residual = self.head(d0)
        
        # Global Skip Connection
        out = self.global_proj(x_input) + residual
        out = torch.clamp(out, -1.0, 1.0)
        
        return out

# --- Factory Wrappers ---
def convnext_tiny(in_channels=3, out_channels=3, drop_path_rate=0.1, use_transpose=False):
    return ConvNeXtUNet('convnext_tiny.fb_in22k', in_channels, out_channels, drop_path_rate=drop_path_rate)

def convnext_small(in_channels=3, out_channels=3, drop_path_rate=0.15, use_transpose=False):
    return ConvNeXtUNet('convnext_small.fb_in22k', in_channels, out_channels, drop_path_rate=drop_path_rate)

def convnext_base(in_channels=3, out_channels=3, drop_path_rate=0.2, use_transpose=False):
    return ConvNeXtUNet('convnext_base.fb_in22k', in_channels, out_channels, drop_path_rate=drop_path_rate)

def convnext_nano(in_channels=3, out_channels=3, drop_path_rate=0.05, use_transpose=False):
    return ConvNeXtUNet('convnext_nano', in_channels, out_channels, drop_path_rate=drop_path_rate)
