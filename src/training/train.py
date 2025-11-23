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
        loss_fn: Función de pérdida
        device: CPU o GPU
        epoch: Número de época
    
    Returns:
        Diccionario con métricas de entrenamiento
    """
    model.train()
    total_loss = 0.0
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
        
        # Calcular pérdida
        if isinstance(loss_fn, nn.Module) and hasattr(loss_fn, '__name__'):
            loss = loss_fn(output, target_img)
        else:
            # Si es HybridLoss que retorna un diccionario
            losses_dict = loss_fn(output, target_img, mask=mask)
            loss = losses_dict['total']
        
        # Backward pass
        loss.backward()
        
        # Gradient Clipping (Crucial para NAFNet/MambaIR)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
    
    return {
        'loss': total_loss / num_batches
    }


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
    total_loss = 0.0
    psnr_values = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_img = batch['input'].to(device)
            target_img = batch['gt'].to(device)
            
            output = model(input_img)
            
            if isinstance(loss_fn, nn.Module) and hasattr(loss_fn, '__name__'):
                loss = loss_fn(output, target_img)
            else:
                losses_dict = loss_fn(output, target_img)
                loss = losses_dict['total']
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calcular PSNR (max_val = 2.0 para rango [-1, 1])
            mse = torch.mean((output - target_img) ** 2)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
            psnr_values.append(psnr.item())
    
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    
    return {
        'val_loss': total_loss / num_batches,
        'psnr': avg_psnr
    }


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
