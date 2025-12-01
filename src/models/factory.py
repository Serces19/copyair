"""
Factory para creación de modelos y optimizadores.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from .unet import UNet
from .modernUnet import ModernUNet
from .architectures import UMamba
from .nafnet import nafnet_small, nafnet_base, nafnet_large, NAFNetHD
from .convnext import convnext_nano, convnext_tiny, convnext_small, convnext_base, ConvNeXtUNet
from .mambair import mambair_tiny, mambair_base, mambair_large, MambaIRv2

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Crea y retorna el modelo especificado en la configuración.
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
    size = config.get('size', 'base').lower()
    
    # Nuevos parámetros de configuración
    use_batchnorm = config.get('use_batchnorm', True)
    use_dropout = config.get('use_dropout', False)
    dropout_p = config.get('dropout_p', 0.0)
    use_transpose = config.get('use_transpose', False)
    
    if arch == 'unet':
        # La implementación de `UNet` espera la firma:
        # UNet(in_channels=..., out_channels=..., base_channels=..., depth=..., norm=..., activation=...)
        # Mapear parámetros antiguos a la nueva API para mantener compatibilidad.
        norm = 'batch' if use_batchnorm else 'group'
        depth = config.get('depth', 5)
        # return UNet(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     base_channels=base_channels,
        #     depth=depth,
        #     norm=norm,
        #     activation=activation
        # )
        return UNet(
            base_channels=64,
            activation=activation,  # o 'gelu'
            dropout=dropout_p,
            output_activation='tanh'  # o 'sigmoid' para segmentación
        )
  
    if arch == 'modernunet':
        # return ModernUNet(
        #     in_channels = in_channels,   # Igual que antes (ej. 3)
        #     out_channels = out_channels, # Igual que antes (ej. 3)
        #     base_dim = base_channels,    # Tu antiguo 'base_channels' (ej. 32 o 64)
        #     num_blocks = [6, 4, 4, 4]    # NUEVO: Controla la profundidad por nivel
        # )
        return ModernUNet(
            in_channels = in_channels,   # Igual que antes (ej. 3)
            out_channels = out_channels, # Igual que antes (ej. 3)
            base_channels = base_channels,    # Tu antiguo 'base_channels' (ej. 32 o 64)
        )

    elif arch == 'umamba':
        return UMamba(in_channels, out_channels, base_channels, activation)
    elif arch == 'nafnet':
        # NAFNet usa dropout_p como drop_out_rate si use_dropout es True
        drop_rate = dropout_p if use_dropout else 0.0
        
        if size == 'small':
            return nafnet_small(in_channels, out_channels, drop_out_rate=drop_rate)
        elif size == 'base':
            return nafnet_base(in_channels, out_channels, drop_out_rate=drop_rate)
        elif size == 'large':
            return nafnet_large(in_channels, out_channels, drop_out_rate=drop_rate)
        else:
            raise ValueError(f"NAFNet size no soportado: {size}. Opciones: small, base, large")
    elif arch == 'convnext':
        # ConvNeXt usa drop_path_rate (stochastic depth)
        drop_path = config.get('drop_path_rate', 0.1)
        
        if size == 'nano':
            return convnext_nano(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'tiny':
            return convnext_tiny(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'small':
            return convnext_small(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        elif size == 'base':
            return convnext_base(in_channels, out_channels, drop_path, use_transpose=use_transpose)
        else:
            raise ValueError(f"ConvNeXt size no soportado: {size}. Opciones: nano, tiny, small, base")
    elif arch == 'mambair':
        if size == 'tiny':
            return mambair_tiny(in_channels, out_channels)
        elif size == 'base':
            return mambair_base(in_channels, out_channels)
        elif size == 'large':
            return mambair_large(in_channels, out_channels)
        else:
            raise ValueError(f"MambaIR size no soportado: {size}. Opciones: tiny, base, large")
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
