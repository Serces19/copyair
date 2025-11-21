"""
Learning Rate Schedulers Factory
Permite configurar diferentes schedulers desde params.yaml
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)
from typing import Dict, Any


class ConstantLR:
    """
    Scheduler que mantiene el learning rate constante.
    Compatible con la interfaz de PyTorch schedulers.
    """
    def __init__(self, optimizer: Optimizer, **kwargs):
        self.optimizer = optimizer
        self._last_lr = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, *args, **kwargs):
        """No hace nada, mantiene LR constante"""
        pass
    
    def get_last_lr(self):
        """Retorna el LR actual"""
        return self._last_lr
    
    def state_dict(self):
        """Retorna estado (vacío para LR constante)"""
        return {}
    
    def load_state_dict(self, state_dict):
        """Carga estado (no hace nada para LR constante)"""
        pass


def get_scheduler(optimizer: Optimizer, config: Dict[str, Any]):
    """
    Factory para crear schedulers basado en configuración.
    
    Args:
        optimizer: Optimizador de PyTorch
        config: Diccionario de configuración con estructura:
            {
                'type': 'constant' | 'cosine' | 'step' | 'exponential' | 'plateau' | 'onecycle' | 'cosine_warmup',
                'params': {...}  # Parámetros específicos del scheduler
            }
    
    Returns:
        Scheduler configurado
    
    Ejemplos de configuración:
        # Constant LR
        scheduler:
            type: constant
        
        # Cosine Annealing
        scheduler:
            type: cosine
            params:
                T_max: 100
                eta_min: 0
        
        # Step LR
        scheduler:
            type: step
            params:
                step_size: 30
                gamma: 0.1
        
        # Exponential LR
        scheduler:
            type: exponential
            params:
                gamma: 0.95
        
        # Reduce on Plateau
        scheduler:
            type: plateau
            params:
                mode: min
                factor: 0.5
                patience: 10
        
        # OneCycle LR
        scheduler:
            type: onecycle
            params:
                max_lr: 0.001
                epochs: 100
                steps_per_epoch: 100
        
        # Cosine Annealing with Warm Restarts
        scheduler:
            type: cosine_warmup
            params:
                T_0: 10
                T_mult: 2
                eta_min: 0
    """
    scheduler_type = config.get('type', 'constant').lower()
    params = config.get('params', {})
    
    if scheduler_type == 'constant':
        return ConstantLR(optimizer)
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', 100),
            eta_min=params.get('eta_min', 0)
        )
    
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=params.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 10),
            verbose=params.get('verbose', True)
        )
    
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=params.get('max_lr', 0.001),
            epochs=params.get('epochs', 100),
            steps_per_epoch=params.get('steps_per_epoch', 100)
        )
    
    elif scheduler_type == 'cosine_warmup':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get('T_0', 10),
            T_mult=params.get('T_mult', 2),
            eta_min=params.get('eta_min', 0)
        )
    
    else:
        raise ValueError(
            f"Scheduler no soportado: {scheduler_type}. "
            f"Opciones: constant, cosine, step, exponential, plateau, onecycle, cosine_warmup"
        )
