"""
Loop de entrenamiento y validación
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Entrena el modelo durante una época
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        optimizer: Optimizador
        loss_fn: Función de pérdida (HybridLoss)
        device: CPU o GPU
        epoch: Número de época
    
    Returns:
        Diccionario con métricas de entrenamiento
    """
    model.train()
    
    # Acumulador de métricas
    accumulated_metrics = {}
    num_batches = 0
    
    logger.info(f"Época {epoch + 1}")
    
    for batch_idx, batch in enumerate(train_loader):
        input_img = batch['input'].to(device)
        target_img = batch['gt'].to(device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_img)
        
        # Calcular pérdida (Siempre es HybridLoss -> retorna dict)
        losses_dict = loss_fn(output, target_img, mask=mask)
        loss = losses_dict['total']
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping (Crucial para NAFNet/MambaIR)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Acumular métricas
        for k, v in losses_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
            
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = accumulated_metrics['total'] / num_batches
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
    
    # Promediar métricas
    final_metrics = {k: v / num_batches for k, v in accumulated_metrics.items()}
    # Renombrar 'total' a 'loss' para compatibilidad
    final_metrics['loss'] = final_metrics['total']
    
    return final_metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn,
    device: torch.device
) -> Dict[str, float]:
    """
    Valida el modelo
    
    Args:
        model: Modelo a validar
        val_loader: DataLoader de validación
        loss_fn: Función de pérdida
        device: CPU o GPU
    
    Returns:
        Diccionario con métricas de validación
    """
    model.eval()
    accumulated_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_img = batch['input'].to(device)
            target_img = batch['gt'].to(device)
            
            output = model(input_img)
            
            # Siempre HybridLoss
            losses_dict = loss_fn(output, target_img)
            
            # Acumular métricas
            for k, v in losses_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
            
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': 0.0}

    # Promediar métricas
    final_metrics = {f"val_{k}": v / num_batches for k, v in accumulated_metrics.items()}
    # Asegurar que existe val_loss
    final_metrics['val_loss'] = final_metrics.get('val_total', 0.0)
    
    return final_metrics


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calcula métricas de evaluación: MSE, PSNR, SSIM
    Asume que output y target están en rango [-1, 1]
    """
    mse = torch.mean((output - target) ** 2)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # max_val = 2.0 para [-1, 1]
    
    # SSIM simplificado
    mean_output = torch.mean(output)
    mean_target = torch.mean(target)
    
    var_output = torch.mean((output - mean_output) ** 2)
    var_target = torch.mean((target - mean_target) ** 2)
    cov = torch.mean((output - mean_output) * (target - mean_target))
    
    ssim = ((2 * mean_output * mean_target + 0.01) * (2 * cov + 0.03)) / \
           ((mean_output ** 2 + mean_target ** 2 + 0.01) * (var_output + var_target + 0.03))
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'ssim': ssim.item()
    }
