"""
Restormer & Modern Vision Transformers for High-Resolution Image Restoration
Inspired by:
- Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022)
- NAFNet / NAF-Transformer: Nonlinear Activation Free Network (ECCV 2022)
- PromptIR: Prompting for All-in-One Image Restoration (NeurIPS 2023)
- HAT: Hybrid Attention Transformer (CVPR 2023)

Features:
- MDTA (Multi-Dconv Head Transposed Attention) with linear complexity.
- GDFN (Gated-Dconv Feed-Forward Network) supporting both GELU and SimpleGate.
- RestormerBlock: Combines MDTA + GDFN + LayerNorm.
- RestormerUNet: Progressive U-Net architecture.
- Global residual connection with range control (clamp/tanh).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """LayerNorm for (B, C, H, W) format without autograd hacks"""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention
    Computes cross-covariance across channels instead of spatial dimension.
    Complexity is linear O(W*H).
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, -1, h, w)

        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network
    Supports Restormer GELU-gate and NAFNet SimpleGate.
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, use_simplegate=False):
        super().__init__()
        self.use_simplegate = use_simplegate
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        if self.use_simplegate:
            x = x1 * x2  # SimpleGate (NAFNet style)
        else:
            x = F.gelu(x1) * x2  # Restormer style
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    """
    Restormer Block: LayerNorm -> MDTA -> LayerNorm -> GDFN
    With learnable LayerScale for stability.
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, use_simplegate=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias, use_simplegate=use_simplegate)

        self.gamma1 = nn.Parameter(1e-4 * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(1e-4 * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class DownSample(nn.Module):
    """PixelUnshuffle downsampling + 1x1 conv"""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    """1x1 conv + PixelShuffle upsampling"""
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class RestormerUNet(nn.Module):
    """
    RestormerUNet: Complete Multi-Scale Architecture
    Also acts as a foundation for PromptIR/HAT extensions by adjusting block types.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        use_simplegate=False,
        activation_out="clamp"
    ):
        super().__init__()
        self.activation_out = activation_out
        self.intro = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Encoder Levels
        self.encoder_level1 = nn.Sequential(*[
            RestormerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = DownSample(dim)

        self.encoder_level2 = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = DownSample(int(dim * 2**1))

        self.encoder_level3 = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = DownSample(int(dim * 2**2))

        # Bottleneck (Latent)
        self.latent = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[3])
        ])

        # Decoder Levels
        self.up4_3 = UpSample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = UpSample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = UpSample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            RestormerBlock(dim=int(dim * 2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, use_simplegate=use_simplegate)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, padding=1, bias=bias)

    def _pad_to_multiple(self, x, multiple=8):
        h, w = x.shape[2], x.shape[3]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

    def forward(self, inp_img):
        # 1. Padding to ensure divisibility by 8 (due to 3 downsamples: 2^3)
        inp_padded, h_orig, w_orig = self._pad_to_multiple(inp_img, multiple=8)

        # 2. Extract shallow features
        inp_enc_level1 = self.intro(inp_padded)
        
        # 3. Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        
        # 4. Bottleneck
        latent = self.latent(inp_enc_level4)

        # 5. Decoder with Skip Connections
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # 6. Refinement and Output
        out_dec_level1 = self.refinement(out_dec_level1)
        residual = self.output(out_dec_level1)
        
        # Global Residual Learning
        out = inp_padded + residual
        
        if self.activation_out == "clamp":
            out = torch.clamp(out, -1.0, 1.0)
        elif self.activation_out == "tanh":
            out = torch.tanh(out)
            
        # 7. Unpad to original size
        return out[:, :, :h_orig, :w_orig]


def restormer_tiny(in_channels=3, out_channels=3, **kwargs):
    """Restormer Tiny (~6.5M params)"""
    return RestormerUNet(
        in_channels=in_channels, out_channels=out_channels,
        dim=24, num_blocks=[2, 3, 3, 4], num_refinement_blocks=2, heads=[1, 2, 4, 8], **kwargs
    )


def restormer_small(in_channels=3, out_channels=3, **kwargs):
    """Restormer Small (~16.1M params)"""
    return RestormerUNet(
        in_channels=in_channels, out_channels=out_channels,
        dim=36, num_blocks=[3, 4, 4, 6], num_refinement_blocks=3, heads=[1, 2, 4, 8], **kwargs
    )


def restormer_base(in_channels=3, out_channels=3, **kwargs):
    """Restormer Base (~26.1M params)"""
    return RestormerUNet(
        in_channels=in_channels, out_channels=out_channels,
        dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8], **kwargs
    )


# Aliases
Restormer = RestormerUNet

__all__ = [
    "Restormer",
    "RestormerUNet",
    "restormer_tiny",
    "restormer_small",
    "restormer_base",
    "MDTA",
    "GDFN",
    "TransformerBlock"
]

