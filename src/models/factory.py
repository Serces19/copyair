"""
Factory para creación de modelos y optimizadores.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from .unet import UNet, UNetWithConvNeXt
from .architectures import ConvexNet, MambaIRv2, UMamba, NAFNetHD
from .nafnet import nafnet_small, nafnet_base, nafnet_large
from .convnext import convnext_nano, convnext_tiny, convnext_small, convnext_base

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Crea y retorna el modelo especificado en la configuración.
    
    Args:
        config: Diccionario de configuración (sección 'model')
        
    Returns:
        Modelo instanciado (nn.Module)
        
    Arquitecturas disponibles:
        - unet: U-Net clásica
        - unet_convnext: U-Net con backbone ConvNeXt
        - convexnet: ConvexNet
        - mambairv2: MambaIR v2
        - umamba: U-Mamba
        - nafnet: NAFNet HD (especificar size: small/base/large)
        - convnext: ConvNeXt U-Net (especificar size: nano/tiny/small/base)
    """
    arch = config.get('architecture', 'unet').lower()
    model_type = config.get('type', 'unet').lower() # Fallback for legacy config
    
    # Prioritize 'architecture' but support 'type' for backward compatibility
    if 'architecture' not in config and 'type' in config:
        arch = model_type
        
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 3)
    base_channels = config.get('base_channels', 64)
    activation = config.get('activation', 'relu')
    size = config.get('size', 'base').lower()  # Para NAFNet y ConvNeXt
    
    if arch == 'unet':
        return UNet(in_channels, out_channels, base_channels, activation)
    elif arch == 'unet_convnext':
        return UNetWithConvNeXt(in_channels, out_channels, base_channels, activation)
    elif arch == 'convexnet':
        return ConvexNet(in_channels, out_channels, base_channels, activation)
    elif arch == 'mambairv2':
        return MambaIRv2(in_channels, out_channels, base_channels, activation)
    elif arch == 'umamba':
        return UMamba(in_channels, out_channels, base_channels, activation)
    elif arch == 'nafnet' or arch == 'nafnethd':
        # NAFNet con diferentes tamaños
        if size == 'small':
            return nafnet_small(in_channels, out_channels)
        elif size == 'base':
            return nafnet_base(in_channels, out_channels)
        elif size == 'large':
            return nafnet_large(in_channels, out_channels)
        else:
            raise ValueError(f"NAFNet size no soportado: {size}. Opciones: small, base, large")
    elif arch == 'convnext':
        # ConvNeXt con diferentes tamaños
        drop_path = config.get('drop_path_rate', 0.1)
        if size == 'nano':
            return convnext_nano(in_channels, out_channels, drop_path)
        elif size == 'tiny':
            return convnext_tiny(in_channels, out_channels, drop_path)
        elif size == 'small':
            return convnext_small(in_channels, out_channels, drop_path)
        elif size == 'base':
            return convnext_base(in_channels, out_channels, drop_path)
        else:
            raise ValueError(f"ConvNeXt size no soportado: {size}. Opciones: nano, tiny, small, base")
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Crea el optimizador basado en la configuración.
    
    Args:
        model: Modelo cuyos parámetros se optimizarán
        config: Diccionario de configuración (sección 'training')
    """
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
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
