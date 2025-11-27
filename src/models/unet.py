import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """(Conv => Norm => Act) * 2"""
    def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = self._get_norm(norm, out_ch)
        self.act1 = self._get_act(activation)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = self._get_norm(norm, out_ch)
        self.act2 = self._get_act(activation)

    def _get_norm(self, norm, channels):
        if norm == 'batch':
            return nn.BatchNorm2d(channels)
        if norm == 'instance':
            return nn.InstanceNorm2d(channels, affine=True)
        groups = min(32, channels) if channels >= 8 else 1
        return nn.GroupNorm(groups, channels)

    def _get_act(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        if name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        if name == 'gelu':
            return nn.GELU()
        return nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch, **kw):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, **kw)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv. Uses ConvTranspose2d or bilinear upsample."""
    def __init__(self, in_ch, out_ch, bilinear=False, norm='group', activation='silu'):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # learnable upsample
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # after up, we'll concat with skip -> channels = out_ch + skip_ch (which equals out_ch)
        # so conv input is out_ch*2, output is out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch, norm=norm, activation=activation)

    def forward(self, x1, x2=None):
        # x1: from previous layer; x2: skip connection
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY != 0 or diffX != 0:
                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            # If no skip, duplicate channels to match conv input expectations
            x = torch.cat([x1, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet desde cero, compatible con la API usada en `factory.get_model`.

    Parámetros principales:
    - `base_channels`: anchura inicial (por defecto 64). Tests pueden usar valores pequeños (ej. 8).
    - `depth`: número de niveles (default 4).
    - `use_batchnorm`: para compatibilidad con configs antiguas; si True, fuerza `norm='batch'`.
    - `use_transpose`: si True, usa ConvTranspose2d (learnable) para upsampling; si False y `bilinear=True`, usa bilinear upsample.
    - `tanh` en salida para mantener rango [-1, 1].
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        activation: str = 'silu',
        norm: str = 'group',
        depth: int = 4,
        bilinear: bool = False,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
        # Compatibilidad con la interfaz anterior
        use_batchnorm: bool = True,
        use_transpose: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        # Compat: si se pide batchnorm explícitamente, priorizar
        if use_batchnorm:
            norm = 'batch'

        # If use_transpose is True, prefer ConvTranspose (i.e. bilinear=False)
        if use_transpose:
            bilinear = False

        # Initial conv (no downsample here)
        self.inc = DoubleConv(in_channels, base_channels, norm=norm, activation=activation)

        # Encoder
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(channels[i], channels[i + 1], norm=norm, activation=activation))

        # Bottleneck
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2, norm=norm, activation=activation)

        # Decoder
        self.ups = nn.ModuleList()
        up_in_ch = channels[-1] * 2
        for i in reversed(range(depth - 1)):
            up_out = channels[i]
            # Up expects in_ch=up_in_ch, out_ch=up_out
            self.ups.append(Up(up_in_ch, up_out, bilinear=bilinear, norm=norm, activation=activation))
            up_in_ch = up_out

        # Final conv
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

        # Optional dropout
        self.use_dropout = use_dropout
        if use_dropout and dropout_p > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        skips = [x1]
        xi = x1
        for down in self.downs:
            xi = down(xi)
            skips.append(xi)

        xb = self.bottleneck(xi)

        x = xb
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        # Ensure spatial match with input (if needed)
        if x.shape[2:] != x1.shape[2:]:
            diffY = x1.size(2) - x.size(2)
            diffX = x1.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = self.outc(x)
        x = self.dropout(x)
        x = self.tanh(x)
        return x


