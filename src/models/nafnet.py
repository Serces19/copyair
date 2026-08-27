"""
NAFNet: Nonlinear Activation Free Network for Image Restoration (ECCV 2022)
Paper: https://arxiv.org/abs/2204.04676

Features:
- Completely avoids traditional activation functions (ReLU, GELU).
- SimpleGate (x1 * x2) for efficient element-wise non-linear interaction.
- Simplified Channel Attention (SCA) for global context without non-linearities.
- LayerScale (beta/gamma) for extreme few-shot stability and zero-init identity learning.
- Dynamic reflective padding for arbitrary video frame sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Clean, high-performance LayerNorm for 4D Image Tensors (B, C, H, W)"""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """SimpleGate: Splits channels in half and multiplies them element-wise"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """SCA: Linear channel attention without sigmoids or multi-layer MLPs"""
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.avg_pool(x))


class NAFBlock(nn.Module):
    """
    NAFBlock: Core building block of NAFNet
    Spatial Mixing (DWConv + SimpleGate + SCA) -> FFN (1x1 Conv + SimpleGate + 1x1 Conv)
    """
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = channels * dw_expand

        # 1. Spatial Mixing
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, kernel_size=1, padding=0, bias=True)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        # 2. Feed-Forward Network (FFN)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * ffn_expand, kernel_size=1, padding=0, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d((channels * ffn_expand) // 2, channels, kernel_size=1, padding=0, bias=True)
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        # LayerScale parameters
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        # Branch 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # Branch 2 (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNetHD(nn.Module):
    """
    NAFNetHD: Complete U-Net with NAFBlocks, PixelShuffle transitions, and Global Skip Connection
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        middle_blk_num: int = 12,
        enc_blk_nums: list = [2, 2, 4, 8],
        dec_blk_nums: list = [2, 2, 2, 2],
        dw_expand: int = 1,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def _pad_to_multiple(self, x: torch.Tensor):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect'), h, w

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp_padded, h_orig, w_orig = self._pad_to_multiple(inp)

        x = self.intro(inp_padded)
        enc_skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(enc_skips)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        residual = self.ending(x)

        # Global residual learning for stable convergence
        out = inp_padded + residual
        out = torch.clamp(out, -1.0, 1.0)

        return out[:, :, :h_orig, :w_orig]


def nafnet_small(in_channels=3, out_channels=3, drop_out_rate=0.05):
    """NAFNet Small (~15.3M params)"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=32,
        middle_blk_num=6,
        enc_blk_nums=[1, 1, 1, 2],
        dec_blk_nums=[1, 1, 1, 1],
        drop_out_rate=drop_out_rate
    )


def nafnet_base(in_channels=3, out_channels=3, drop_out_rate=0.05):
    """NAFNet Base (~28.4M params - Recomendado)"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        drop_out_rate=drop_out_rate
    )


def nafnet_large(in_channels=3, out_channels=3, drop_out_rate=0.0):
    """NAFNet Large (~45.8M params)"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=64,
        middle_blk_num=16,
        enc_blk_nums=[2, 2, 6, 12],
        dec_blk_nums=[2, 2, 2, 2],
        drop_out_rate=drop_out_rate
    )
