"""
NAFNet (Nonlinear Activation Free Network) para Image Restoration
Paper: "Simple Baselines for Image Restoration" (ECCV 2022)
Optimizado para: Pocas imágenes de alta resolución

Características:
- Sin activaciones no lineales tradicionales (ReLU, GELU)
- SimpleGate: activación eficiente basada en multiplicación
- Simplified Channel Attention (SCA)
- Muy eficiente computacionalmente
- Excelente para few-shot learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """LayerNorm optimizado para canales"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """LayerNorm para imágenes (N, C, H, W)"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """
    SimpleGate: Activación sin no-linealidad explícita
    Divide canales en 2 grupos y multiplica elemento a elemento
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    SCA: Channel Attention simplificado sin no-linealidades
    Usa solo avg pooling + conv 1x1
    """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class NAFBlock(nn.Module):
    """
    Bloque NAF: Building block principal de NAFNet
    
    Flujo:
    Input -> LN -> Conv1x1 -> DWConv3x3 -> SimpleGate -> Conv1x1 -> SCA -> + Input
    """
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = channels * dw_expand
        
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, 1, padding=0, stride=1, groups=1, bias=True)
        
        # SCA
        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        
        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        # FFN (Feed-Forward Network)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * ffn_expand, 1, padding=0, stride=1, groups=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(channels * ffn_expand // 2, channels, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        # Beta y gamma para scaling (importante para few-shot)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        
        # Primera rama
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        # Segunda rama (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class NAFNetHD(nn.Module):
    """
    NAFNet optimizado para alta resolución
    
    Arquitectura:
    - Encoder-Decoder con skip connections
    - NAF blocks en cada nivel
    - Sin downsampling agresivo (mejor para HR)
    - Optimizado para pocas imágenes (dropout, regularización)
    
    Args:
        in_channels: Canales de entrada (3 para RGB)
        out_channels: Canales de salida (3 para RGB)
        width: Ancho base de canales (32 o 64)
        middle_blk_num: Bloques en el bottleneck (4-6)
        enc_blk_nums: Lista de bloques por nivel encoder [1,1,1,28]
        dec_blk_nums: Lista de bloques por nivel decoder [1,1,1,1]
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        dw_expand=1,
        ffn_expand=2,
        drop_out_rate=0.0
    ):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1, stride=1, groups=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(middle_blk_num)]
        )
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, dw_expand, ffn_expand, drop_out_rate) for _ in range(num)]
                )
            )
        
        self.padder_size = 2 ** len(self.encoders)
        
        # Activación final: Tanh para rango [-1, 1]
        self.tanh = nn.Tanh()

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = self.tanh(x)
        
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Padding para que sea divisible por padder_size"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


def nafnet_small(in_channels=3, out_channels=3):
    """NAFNet pequeño para pocas imágenes"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=32,
        middle_blk_num=6,
        enc_blk_nums=[1, 1, 1, 2],
        dec_blk_nums=[1, 1, 1, 1],
        drop_out_rate=0.1  # Dropout para evitar overfitting
    )


def nafnet_base(in_channels=3, out_channels=3):
    """NAFNet base (recomendado para 5-15 imágenes)"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        drop_out_rate=0.05
    )


def nafnet_large(in_channels=3, out_channels=3):
    """NAFNet grande (solo si tienes >10 imágenes)"""
    return NAFNetHD(
        in_channels=in_channels,
        out_channels=out_channels,
        width=64,
        middle_blk_num=16,
        enc_blk_nums=[2, 2, 6, 12],
        dec_blk_nums=[2, 2, 2, 2],
        drop_out_rate=0.0
    )
