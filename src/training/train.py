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
    Valida el modelo usando métricas robustas:
    1. HybridLoss (Loss de entrenamiento)
    2. PSNR y SSIM (Sanity Checks)
    3. Crop-LPIPS (Métrica de textura/fidelidad en parches)
    
    Args:
        model: Modelo a validar
        val_loader: DataLoader de validación
        loss_fn: Función de pérdida
        device: CPU o GPU
    
    Returns:
        Diccionario con métricas de validación
    """
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    import random

    model.eval()
    accumulated_metrics = {}
    num_batches = 0
    
    # Inicializar métricas de validación
    # Rango de datos [-1, 1] -> data_range = 2.0
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    # LPIPS espera [-1, 1] si normalize=False
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device)
    
    logger.info("Iniciando validación con métricas extendidas (PSNR, SSIM, Crop-LPIPS)...")
    
    with torch.no_grad():
        for batch in val_loader:
            input_img = batch['input'].to(device)
            target_img = batch['gt'].to(device)
            
            output = model(input_img)
            
            # 1. Pérdida Híbrida (la misma que train)
            losses_dict = loss_fn(output, target_img)
            
            # Acumular métricas de loss
            for k, v in losses_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
            
            # 2. Métricas de Sanidad (PSNR, SSIM)
            # Calculadas sobre la imagen completa
            psnr_val = psnr_metric(output, target_img).item()
            ssim_val = ssim_metric(output, target_img).item()
            
            accumulated_metrics['psnr'] = accumulated_metrics.get('psnr', 0.0) + psnr_val
            accumulated_metrics['ssim'] = accumulated_metrics.get('ssim', 0.0) + ssim_val
            
            # 3. Crop-LPIPS (Validación de Textura)
            # Extraer 5 parches aleatorios de 256x256
            n_crops = 5
            crop_size = 256
            batch_lpips = 0.0
            
            B, C, H, W = output.shape
            
            # Si la imagen es más pequeña que el crop, usar la imagen entera
            if H < crop_size or W < crop_size:
                batch_lpips = lpips_metric(output, target_img).item()
            else:
                for _ in range(n_crops):
                    # Coordenadas aleatorias
                    h_start = random.randint(0, H - crop_size)
                    w_start = random.randint(0, W - crop_size)
                    
                    # Recortar
                    crop_pred = output[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
                    crop_target = target_img[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
                    
                    # Calcular LPIPS del parche
                    batch_lpips += lpips_metric(crop_pred, crop_target).item()
                
                batch_lpips /= n_crops
            
            accumulated_metrics['crop_lpips'] = accumulated_metrics.get('crop_lpips', 0.0) + batch_lpips
            
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': 0.0}

    # Promediar métricas
    final_metrics = {f"val_{k}": v / num_batches for k, v in accumulated_metrics.items()}
    # Asegurar que existe val_loss
    final_metrics['val_loss'] = final_metrics.get('val_total', 0.0)
    
    return final_metrics
