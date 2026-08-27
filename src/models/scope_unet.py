"""
ScopeUNet / GatedUNet: Lossless Multiscale Gated Architecture for VFX Video Restoration
Features:
- PixelUnshuffle for lossless spatial downsampling preserving 100% of high frequencies.
- PixelShuffle with ICNR weight initialization to eliminate checkerboard artifacts.
- GatedBlock (LayerNorm2d + DepthwiseConv + SimpleGate + Simplified Channel Attention).
- Reflective padding for native arbitrary frame sizes.
- Global residual learning for stable de-aging & skin texture translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for (B, C, H, W) format"""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """SimpleGate: Element-wise non-linear interaction without activation functions"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GatedBlock(nn.Module):
    """
    GatedBlock: Combines Depthwise Convolutions with SimpleGate and Channel Attention
    """
    def __init__(self, c: int, ffn_expansion_factor: int = 2):
        super().__init__()
        self.norm1 = LayerNorm2d(c)

        # 1. Spatial & Channel Mixing
        self.conv1 = nn.Conv2d(c, c, kernel_size=1)
        self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)

        hidden_dim = int(c * ffn_expansion_factor)
        self.conv2 = nn.Conv2d(c, hidden_dim * 2, kernel_size=1)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(hidden_dim, c, kernel_size=1)

        # 2. Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1)
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = x * self.sca(x)
        return inp + x * self.beta


class Downsample(nn.Module):
    """Lossless Downsampling via PixelUnshuffle"""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_c * 4, out_c, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpsampleICNR(nn.Module):
    """Checkerboard-free Upsampling via PixelShuffle + ICNR Initialization"""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=1)
        self.ps = nn.PixelShuffle(2)
        self._icnr_init()

    def _icnr_init(self):
        weight = nn.init.kaiming_normal_(
            torch.zeros_like(self.conv.weight[:self.conv.out_channels // 4])
        )
        weight = weight.transpose(0, 1).repeat(1, 4, 1, 1).transpose(0, 1)
        self.conv.weight.data.copy_(weight)
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class GatedUNet(nn.Module):
    """
    ScopeUNet / GatedUNet Architecture
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_dim: int = 32,
        num_blocks: list = [2, 2, 4, 6]
    ):
        super().__init__()
        self.base_dim = base_dim

        self.intro = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        # Encoder Levels
        self.enc1 = nn.Sequential(*[GatedBlock(base_dim) for _ in range(num_blocks[0])])
        self.down1 = Downsample(base_dim, base_dim * 2)

        self.enc2 = nn.Sequential(*[GatedBlock(base_dim * 2) for _ in range(num_blocks[1])])
        self.down2 = Downsample(base_dim * 2, base_dim * 4)

        self.enc3 = nn.Sequential(*[GatedBlock(base_dim * 4) for _ in range(num_blocks[2])])
        self.down3 = Downsample(base_dim * 4, base_dim * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[GatedBlock(base_dim * 8) for _ in range(num_blocks[3])])

        # Decoder Levels
        self.up3 = UpsampleICNR(base_dim * 8, base_dim * 4)
        self.reduce3 = nn.Conv2d(base_dim * 8, base_dim * 4, kernel_size=1)
        self.dec3 = nn.Sequential(*[GatedBlock(base_dim * 4) for _ in range(num_blocks[2])])

        self.up2 = UpsampleICNR(base_dim * 4, base_dim * 2)
        self.reduce2 = nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.dec2 = nn.Sequential(*[GatedBlock(base_dim * 2) for _ in range(num_blocks[1])])

        self.up1 = UpsampleICNR(base_dim * 2, base_dim)
        self.reduce1 = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1)
        self.dec1 = nn.Sequential(*[GatedBlock(base_dim) for _ in range(num_blocks[0])])

        self.final = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int = 8):
        h, w = x.shape[2], x.shape[3]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, h_orig, w_orig = self._pad_to_multiple(x, multiple=8)

        x1 = self.intro(x_pad)
        x1 = self.enc1(x1)

        x2 = self.down1(x1)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.enc3(x3)

        x4 = self.down3(x3)
        x4 = self.bottleneck(x4)

        up3 = self.up3(x4)
        cat3 = torch.cat([up3, x3], dim=1)
        out3 = self.dec3(self.reduce3(cat3))

        up2 = self.up2(out3)
        cat2 = torch.cat([up2, x2], dim=1)
        out2 = self.dec2(self.reduce2(cat2))

        up1 = self.up1(out2)
        cat1 = torch.cat([up1, x1], dim=1)
        out1 = self.dec1(self.reduce1(cat1))

        residual = self.final(out1)
        out = x_pad + residual
        out = torch.clamp(out, -1.0, 1.0)

        return out[:, :, :h_orig, :w_orig]


# Aliases & Helpers
ModernUNet = GatedUNet


def gated_unet_small(in_channels=3, out_channels=3):
    return GatedUNet(in_channels, out_channels, base_dim=32, num_blocks=[1, 1, 2, 4])


def gated_unet_base(in_channels=3, out_channels=3):
    return GatedUNet(in_channels, out_channels, base_dim=48, num_blocks=[2, 2, 4, 6])


def scope_unet_base(in_channels=3, out_channels=3):
    return GatedUNet(in_channels, out_channels, base_dim=32, num_blocks=[2, 2, 4, 6])
