"""
Residual U-Net for Image-to-Image Translation
Features:
- Pre-activation ResBlocks (GroupNorm + Mish/SiLU) with True Identity Shortcuts (f(x) + x).
- Dilated Bottleneck (dilation=2) to double the receptive field without loss of spatial detail.
- Global Residual Learning (Output = Input + Residual) with range [-1, 1].
- Dynamic padding for arbitrary frame resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock


class ResidualUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        norm_type: str = 'group',
        activation: str = 'mish',
        dropout: float = 0.0,
        use_dilated_bottleneck: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.use_dilated_bottleneck = use_dilated_bottleneck

        # --- ENCODER ---
        self.inc = ResBlock(in_channels, base_channels, activation, norm_type, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_channels, base_channels * 2, activation, norm_type, dropout=dropout)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_channels * 2, base_channels * 4, activation, norm_type, dropout=dropout)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_channels * 4, base_channels * 8, activation, norm_type, dropout=dropout)
        )

        # --- BOTTLENECK ---
        if use_dilated_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=2, dilation=2),
                nn.GroupNorm(32, base_channels * 16),
                nn.Mish(inplace=True) if activation == 'mish' else nn.SiLU(inplace=True),
                nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=2, dilation=2),
                nn.GroupNorm(32, base_channels * 8),
                nn.Mish(inplace=True) if activation == 'mish' else nn.SiLU(inplace=True)
            )
            bn_out = base_channels * 8
        else:
            self.bottleneck = ResBlock(base_channels * 8, base_channels * 16, activation, norm_type, dropout=dropout)
            bn_out = base_channels * 16

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = ResBlock(bn_out + base_channels * 4, base_channels * 4, activation, norm_type, dropout=dropout)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, activation, norm_type, dropout=dropout)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ResBlock(base_channels * 2 + base_channels, base_channels, activation, norm_type, dropout=dropout)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int = 8):
        h, w = x.shape[2], x.shape[3]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x_pad, h_orig, w_orig = self._pad_to_multiple(input_tensor, multiple=8)

        # Encoder
        x1 = self.inc(x_pad)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x_bn = self.bottleneck(x4)

        # Decoder
        x_up3 = self.up3(x_bn)
        if x_up3.shape[2:] != x3.shape[2:]:
            x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([x_up3, x3], dim=1)
        d3 = self.conv3(d3)

        x_up2 = self.up2(d3)
        if x_up2.shape[2:] != x2.shape[2:]:
            x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([x_up2, x2], dim=1)
        d2 = self.conv2(d2)

        x_up1 = self.up1(d2)
        if x_up1.shape[2:] != x1.shape[2:]:
            x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([x_up1, x1], dim=1)
        d1 = self.conv1(d1)

        residual = self.outc(d1)

        # Global Skip Connection
        out = x_pad + residual
        out = torch.clamp(out, -1.0, 1.0)
        return out[:, :, :h_orig, :w_orig]
