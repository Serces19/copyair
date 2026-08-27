"""
Factory para creación de modelos y optimizadores en CopyAir.
Catálogo oficial consolidado de arquitecturas SOTA para VFX y Generative AI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

# Core Architectures
from .basic_unet import BasicUNet
from .smartunet import SmartUNet
from .residual_unet import ResidualUNet
from .modern_unet import ModernUNet
from .scope_unet import GatedUNet, gated_unet_small, gated_unet_base, scope_unet_base
from .nafnet import nafnet_small, nafnet_base, nafnet_large, NAFNetHD
from .convnext import convnext_nano, convnext_tiny, convnext_small, convnext_base, ConvNeXtUNet
from .restormer import restormer_tiny, restormer_small, restormer_base, Restormer
from .mambair import mambair_tiny, mambair_base, mambair_large, MambaIRv2
from .swin_unet import SwinV2UNet
from .unet import UNet  # Legacy baseline


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Instancia y retorna el modelo especificado en la configuración.
    """
    arch = config.get('architecture', 'nafnet').lower()

    # Parámetros comunes
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 3)
    base_channels = config.get('base_channels', 64)
    norm_type = config.get('norm_type', 'group')
    activation = config.get('activation', 'mish')
    dropout = config.get('dropout_p', 0.0)
    size = config.get('size', 'base').lower()

    # 1. NAFNet (Nonlinear Activation Free Network - SOTA Few-Shot)
    if arch == 'nafnet':
        if size == 'small':
            return nafnet_small(in_channels, out_channels, drop_out_rate=dropout)
        elif size == 'base':
            return nafnet_base(in_channels, out_channels, drop_out_rate=dropout)
        elif size == 'large':
            return nafnet_large(in_channels, out_channels, drop_out_rate=dropout)
        else:
            return NAFNetHD(in_channels=in_channels, out_channels=out_channels, width=base_channels, drop_out_rate=dropout)

    # 2. ScopeUNet / GatedUNet (PixelUnshuffle + ICNR PixelShuffle)
    elif arch in ['scope_unet', 'gated_unet']:
        if size == 'small':
            return gated_unet_small(in_channels, out_channels)
        elif size == 'base':
            return gated_unet_base(in_channels, out_channels)
        else:
            return GatedUNet(in_channels, out_channels, base_dim=base_channels)

    # 3. Restormer (Multi-Dconv Head Transposed Attention + GDFN)
    elif arch == 'restormer':
        if size == 'tiny':
            return restormer_tiny(in_channels, out_channels)
        elif size == 'small':
            return restormer_small(in_channels, out_channels)
        elif size == 'base':
            return restormer_base(in_channels, out_channels)
        else:
            return Restormer(in_channels=in_channels, out_channels=out_channels, dim=base_channels)

    # 4. ConvNeXt-V2 (Pretrained Backbone + GroupNorm/GRN Decoder)
    elif arch == 'convnext':
        drop_path = config.get('drop_path_rate', 0.10)
        use_transpose = config.get('use_transpose', False)
        if size == 'nano':
            return convnext_nano(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'tiny':
            return convnext_tiny(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'small':
            return convnext_small(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'base':
            return convnext_base(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        else:
            return ConvNeXtUNet('convnext_tiny.fb_in22k', in_channels, out_channels, drop_path_rate=drop_path)

    # 5. MambaIR (2D Cross-Scan State Space Model)
    elif arch in ['mambair', 'vmamba', 'mamba']:
        if size == 'tiny':
            return mambair_tiny(in_channels, out_channels)
        elif size == 'base':
            return mambair_base(in_channels, out_channels)
        elif size == 'large':
            return mambair_large(in_channels, out_channels)
        else:
            return MambaIRv2(in_channels=in_channels, out_channels=out_channels, embed_dim=base_channels)

    # 6. Residual U-Net (Pre-Act Mish + True Identity Shortcuts)
    elif arch == 'residual_unet':
        res_cfg = config.get('residual', {})
        return ResidualUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_dilated_bottleneck=res_cfg.get('use_dilated_bottleneck', True)
        )

    # 7. Modern U-Net (Pre-Act ResBlocks + Bottleneck Self-Attention + Attention Gates)
    elif arch == 'modern_unet':
        modern_cfg = config.get('modern', {})
        return ModernUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            attention_type=modern_cfg.get('attention_type', 'self')
        )

    # 8. Smart U-Net (100% Persistent Skip Connections)
    elif arch == 'smart_unet':
        smart_cfg = config.get('smart', {})
        return SmartUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_attention=smart_cfg.get('use_attention', False),
            use_smart_filter=smart_cfg.get('use_smart_filter', False)
        )


    # 9. SwinV2 U-Net
    elif arch == 'swin_unet':
        swin_type = config.get('swin_type', 'tiny')
        pretrained = config.get('pretrained', True)
        use_global_residual = config.get('use_global_residual', True)
        return SwinV2UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            swin_type=swin_type,
            pretrained=pretrained,
            use_global_residual=use_global_residual
        )

    # 10. Basic U-Net & Legacy UNet
    elif arch == 'basic_unet':
        return BasicUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
    elif arch == 'unet':
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels
        )
    else:
        raise ValueError(f"Arquitectura no soportada: '{arch}'. Opciones: nafnet, scope_unet, restormer, convnext, mambair, residual_unet, modern_unet, smart_unet, basic_unet")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Crea el optimizador basado en la configuración.
    """
    lr = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 0.0001))

    opt_config = config.get('optimizer', {})
    if isinstance(opt_config, str):
        opt_type = opt_config
        opt_params = {}
    else:
        opt_type = opt_config.get('type', 'adamw')
        opt_params = {k: v for k, v in opt_config.items() if k != 'type'}

    opt_type = opt_type.lower()

    if opt_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(opt_params.get('beta1', 0.9), opt_params.get('beta2', 0.999))
        )
    elif opt_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(opt_params.get('beta1', 0.9), opt_params.get('beta2', 0.999))
        )
    elif opt_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=opt_params.get('momentum', 0.9)
        )
    elif opt_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=opt_params.get('momentum', 0)
        )
    else:
        raise ValueError(f"Optimizador no soportado: {opt_type}")
