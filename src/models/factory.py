"""
Factory para creación de modelos y optimizadores.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

# New Models
from .basic_unet import BasicUNet
from .smartunet import SmartUNet
from .residual_unet import ResidualUNet
from .modern_unet import ModernUNet
from .unet import UNet # Legacy

# Other Architectures
from .architectures import UMamba
from .nafnet import nafnet_small, nafnet_base, nafnet_large, NAFNetHD
from .convnext import convnext_nano, convnext_tiny, convnext_small, convnext_base, ConvNeXtUNet
from .mambair import mambair_tiny, mambair_base, mambair_large, MambaIRv2
from .swin_unet import SwinV2UNet

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Crea y retorna el modelo especificado en la configuración.
    """
    arch = config.get('architecture', 'basic_unet').lower()
    
    # Common Parameters
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 3)
    base_channels = config.get('base_channels', 64)
    norm_type = config.get('norm_type', 'batch')
    activation = config.get('activation', 'relu')
    dropout = config.get('dropout_p', 0.0)
    
    # 1. Basic U-Net
    if arch == 'basic_unet':
        return BasicUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
        
    # 2. Smart U-Net (Configurable)
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
            use_smart_filter=smart_cfg.get('use_smart_filter', False),
            drop_skip_prob=float(smart_cfg.get('drop_skip_prob', 0.0))
        )

    # 3. Residual U-Net (Global Residual)
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

    # 4. Modern U-Net (SOTA VFX)
    elif arch == 'modern_unet':
        modern_cfg = config.get('modern', {})
        input_map_ch = config.get('input_map_channels', 0)
        return ModernUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            input_map_channels=input_map_ch,
            base_channels=base_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            use_film=modern_cfg.get('use_film', True),
            use_adain=modern_cfg.get('use_adain', False),
            attention_type=modern_cfg.get('attention_type', 'self'),
            cond_dim=modern_cfg.get('cond_dim', 128)
        )

    # Legacy / Other Models
    elif arch == 'unet': # Legacy
         return UNet(
            in_channels=in_channels, 
            out_channels=out_channels, 
            base_channels=base_channels
        )

    elif arch == 'nafnet':
        size = config.get('size', 'base').lower()
        if size == 'small': return nafnet_small(in_channels, out_channels, drop_out_rate=dropout)
        elif size == 'base': return nafnet_base(in_channels, out_channels, drop_out_rate=dropout)
        elif size == 'large': return nafnet_large(in_channels, out_channels, drop_out_rate=dropout)
        else: raise ValueError(f"NAFNet size unknown: {size}")

    elif arch == 'convnext':
        size = config.get('size', 'base').lower()
        drop_path = config.get('drop_path_rate', 0.1)
        use_transpose = config.get('use_transpose', False)
        if size == 'nano': return convnext_nano(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'tiny': return convnext_tiny(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'small': return convnext_small(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'base': return convnext_base(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        else: raise ValueError(f"ConvNeXt size unknown: {size}")

    elif arch == 'mambair':
        size = config.get('size', 'base').lower()
        if size == 'tiny': return mambair_tiny(in_channels, out_channels)
        elif size == 'base': return mambair_base(in_channels, out_channels)
        elif size == 'large': return mambair_large(in_channels, out_channels)
        else: raise ValueError(f"MambaIR size unknown: {size}")

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
        
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Crea el optimizador basado en la configuración.
    Args:
        model: Modelo cuyos parámetros se optimizarán
        config: Diccionario de configuración (sección 'training')
    """
    # Convertir a float para manejar notación científica de YAML
    lr = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 0.0001))
    
    # Support both old config (flat) and new config (nested optimizer dict)
    opt_config = config.get('optimizer', {})
    if isinstance(opt_config, str): # Handle case where it might be just a string name
        opt_type = opt_config
        opt_params = {}
    else:
        opt_type = opt_config.get('type', 'adam')
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
