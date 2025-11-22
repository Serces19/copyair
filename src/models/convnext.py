"""
ConvNeXt para Image-to-Image Translation
Paper: "A ConvNet for the 2020s" (CVPR 2022)
Optimizado para: Pocas imágenes de alta resolución con few-shot learning

Características:
- Arquitectura CNN moderna inspirada en Vision Transformers
- Depthwise convolutions grandes (7x7)
- LayerNorm + GELU
- Inverted bottleneck design
- Excelente para few-shot learning
- Escalable a alta resolución
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    """LayerNorm que soporta dos formatos de datos: channels_last y channels_first"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """
    Bloque ConvNeXt
    
    Arquitectura (Inverted Bottleneck):
    DWConv 7x7 -> LayerNorm -> 1x1 Conv (expand 4x) -> GELU -> 1x1 Conv -> DropPath
    
    Args:
        dim: Número de canales de entrada
        drop_path: Probabilidad de DropPath (stochastic depth)
        layer_scale_init_value: Valor inicial para layer scale
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt en arquitectura U-Net para Image-to-Image Translation
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        use_transpose=False
    ):
        super().__init__()
        
        # Stem: Patchify con conv 4x4 stride 4 (similar a ViT)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.encoder_stages.append(stage)
            cur += depths[i]
            
            # Downsampling entre stages (excepto el último)
            if i < len(dims) - 1:
                downsample = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample)
        
        # Decoder stages (U-Net style)
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(len(dims) - 1, 0, -1):
            # Upsample
            if use_transpose:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=2, stride=2),
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first")
                )
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(dims[i], dims[i-1], kernel_size=1),
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first")
                )
            self.upsample_layers.append(upsample)
            
            # Decoder stage (con skip connection, así que dims[i-1] * 2)
            stage = nn.Sequential(
                nn.Conv2d(dims[i-1] * 2, dims[i-1], kernel_size=1),  # Fusión de skip
                *[ConvNeXtBlock(dim=dims[i-1], drop_path=0.0,  # Sin dropout en decoder
                                layer_scale_init_value=layer_scale_init_value)
                  for _ in range(depths[i-1])]
            )
            self.decoder_stages.append(stage)
        
        # Head: Upsample final + conv
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], out_channels, kernel_size=1)
        )
        
        # Activación final: Tanh para rango [-1, 1]
        self.tanh = nn.Tanh()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Encoder con skip connections
        skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i < len(self.encoder_stages) - 1:
                skips.append(x)
                x = self.downsample_layers[i](x)
        
        # Decoder con skip connections
        for i, (upsample, stage) in enumerate(zip(self.upsample_layers, self.decoder_stages)):
            x = upsample(x)
            skip = skips[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = stage(x)
        
        # Head
        x = self.head(x)
        x = self.tanh(x)
        
        return x


def convnext_tiny(in_channels=3, out_channels=3, drop_path_rate=0.1, use_transpose=False):
    """
    ConvNeXt-Tiny para few-shot learning (5-15 imágenes)
    Parámetros: ~28M
    """
    return ConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        drop_path_rate=drop_path_rate,
        use_transpose=use_transpose
    )


def convnext_small(in_channels=3, out_channels=3, drop_path_rate=0.15, use_transpose=False):
    """
    ConvNeXt-Small
    Parámetros: ~50M
    Recomendado para 10-15 imágenes
    """
    return ConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 27, 3],
        drop_path_rate=drop_path_rate,
        use_transpose=use_transpose
    )


def convnext_base(in_channels=3, out_channels=3, drop_path_rate=0.2, use_transpose=False):
    """
    ConvNeXt-Base
    Parámetros: ~89M
    Solo si tienes >15 imágenes de muy alta calidad
    """
    return ConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=[128, 256, 512, 1024],
        depths=[3, 3, 27, 3],
        drop_path_rate=drop_path_rate,
        use_transpose=use_transpose
    )


def convnext_nano(in_channels=3, out_channels=3, drop_path_rate=0.05, use_transpose=False):
    """
    ConvNeXt-Nano (ultra ligero)
    Parámetros: ~15M
    Recomendado para 5-8 imágenes
    """
    return ConvNeXtUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        drop_path_rate=drop_path_rate,
        use_transpose=use_transpose
    )
