"""
Modern U-Net para VFX y Restauración de Video (Estable y optimizado para FP16 / AMP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock


class StableAttentionGate(nn.Module):
    """
    Attention Gate numéricamente estable con GroupNorm (seguro para FP16 y batch size 1).
    g: Gate signal (desde el decoder)
    x: Skip connection features (desde el encoder)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        groups_g = min(16, F_int)
        groups_x = min(16, F_int)
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(groups_g, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(groups_x, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ModernUNet(nn.Module):
    """
    Modern U-Net: Arquitectura convolucional profunda para VFX con:
    - Pre-activation ResBlocks con GroupNorm
    - Bottleneck con Spatial Self-Attention estabilizado
    - Attention Gates en todas las conexiones Skip
    - Global Residual Learning (input + residual)
    - 100% libre de NaNs y optimizado para Mixed Precision (AMP FP16)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        norm_type: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
        attention_type: str = 'self',
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.has_global_residual = (in_channels == out_channels)

        # --- ENCODER ---
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Level 0 (Full Res)
        self.enc0 = ResBlock(base_channels, base_channels, activation, norm_type, dropout=dropout)
        self.down0 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1) # -> H/2
        
        # Level 1 (Half Res)
        self.enc1 = ResBlock(base_channels, base_channels * 2, activation, norm_type, dropout=dropout)
        self.down1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1) # -> H/4
        
        # Level 2 (Quarter Res)
        self.enc2 = ResBlock(base_channels * 2, base_channels * 4, activation, norm_type, dropout=dropout)
        self.down2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1) # -> H/8
        
        # Level 3 (Eighth Res)
        self.enc3 = ResBlock(base_channels * 4, base_channels * 8, activation, norm_type, dropout=dropout)
        self.down3 = nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1) # -> H/16

        # --- BOTTLENECK ---
        self.bottleneck = ResBlock(base_channels * 8, base_channels * 8, activation, norm_type, dropout=dropout)
        
        self.use_self_attn = (attention_type == 'self')
        if self.use_self_attn:
            self.attn_norm = nn.GroupNorm(min(32, base_channels * 8), base_channels * 8)
            self.attn = nn.MultiheadAttention(embed_dim=base_channels * 8, num_heads=8, batch_first=True)

        # --- DECODER CON ATTENTION GATES ---
        # Level 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate3 = StableAttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.reduce3 = nn.Conv2d(base_channels * 8 + base_channels * 8, base_channels * 8, 1)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 4, activation, norm_type, dropout=dropout)

        # Level 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate2 = StableAttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.reduce2 = nn.Conv2d(base_channels * 4 + base_channels * 4, base_channels * 4, 1)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, activation, norm_type, dropout=dropout)

        # Level 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate1 = StableAttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.reduce1 = nn.Conv2d(base_channels * 2 + base_channels * 2, base_channels * 2, 1)
        self.dec1 = ResBlock(base_channels * 2, base_channels, activation, norm_type, dropout=dropout)

        # Level 0 (Full Res)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.att_gate0 = StableAttentionGate(base_channels, base_channels, base_channels // 2)
        self.reduce0 = nn.Conv2d(base_channels + base_channels, base_channels, 1)
        self.dec0 = ResBlock(base_channels, base_channels, activation, norm_type, dropout=dropout)

        # --- HEAD ---
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False) -> torch.Tensor:
        input_x = x
        
        # Stem
        x = self.stem(x)
        
        # Encoder
        x0 = self.enc0(x)
        x_d0 = self.down0(x0)
        
        x1 = self.enc1(x_d0)
        x_d1 = self.down1(x1)
        
        x2 = self.enc2(x_d1)
        x_d2 = self.down2(x2)
        
        x3 = self.enc3(x_d2)
        x_d3 = self.down3(x3)
        
        # Bottleneck
        bn = self.bottleneck(x_d3)
        
        if self.use_self_attn:
            b, c, h, w = bn.shape
            bn_norm = self.attn_norm(bn)
            bn_flat = bn_norm.flatten(2).transpose(1, 2)  # [B, H*W, C]
            attn_out, _ = self.attn(bn_flat, bn_flat, bn_flat)
            bn = bn + attn_out.transpose(1, 2).reshape(b, c, h, w)

        # Decoder
        u3 = self.up3(bn)
        s3 = self.att_gate3(u3, x3)
        d3 = self.dec3(self.reduce3(torch.cat([u3, s3], dim=1)))

        u2 = self.up2(d3)
        s2 = self.att_gate2(u2, x2)
        d2 = self.dec2(self.reduce2(torch.cat([u2, s2], dim=1)))

        u1 = self.up1(d2)
        s1 = self.att_gate1(u1, x1)
        d1 = self.dec1(self.reduce1(torch.cat([u1, s1], dim=1)))

        u0 = self.up0(d1)
        s0 = self.att_gate0(u0, x0)
        d0 = self.dec0(self.reduce0(torch.cat([u0, s0], dim=1)))

        # Output con Global Residual Learning
        residual = self.out_conv(d0)
        if self.has_global_residual:
            out = input_x + residual
        else:
            out = residual
            
        out = torch.clamp(out, -1.0, 1.0)
        
        if return_dict:
            return {'rgb': out}
        return out
