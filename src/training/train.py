"""
Loop de entrenamiento y validación
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Dict, Tuple
import logging
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import random

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
            
            # # 1. Pérdida Híbrida (la misma que train)
            # losses_dict = loss_fn(output, target_img)
            
            # # Acumular métricas de loss
            # for k, v in losses_dict.items():
            #     if isinstance(v, torch.Tensor):
            #         v = v.item()
            #     accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
            
            # 2. Métricas de Sanidad (PSNR, SSIM)
            # Calculadas sobre la imagen completa
            psnr_val = psnr_metric(output, target_img).item()
            ssim_val = ssim_metric(output, target_img).item()
            
            accumulated_metrics['psnr'] = accumulated_metrics.get('psnr', 0.0) + psnr_val
            accumulated_metrics['ssim'] = accumulated_metrics.get('ssim', 0.0) + ssim_val
            
            # 3. Crop-LPIPS (Validación de Textura)
            # Extraer 5 parches aleatorios de 256x256
            # n_crops = 10
            # crop_size = 128
            # batch_lpips = 0.0
            
            # B, C, H, W = output.shape
            
            # # Si la imagen es más pequeña que el crop, usar la imagen entera
            # if H < crop_size or W < crop_size:
            #     batch_lpips = lpips_metric(output, target_img).item()
            # else:
            #     for _ in range(n_crops):
            #         # Coordenadas aleatorias
            #         h_start = random.randint(0, H - crop_size)
            #         w_start = random.randint(0, W - crop_size)
                    
            #         # Recortar
            #         crop_pred = output[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
            #         crop_target = target_img[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
                    
            #         # Calcular LPIPS del parche
            #         batch_lpips += lpips_metric(crop_pred, crop_target).item()
                
            #     batch_lpips /= n_crops
            
            # accumulated_metrics['crop_lpips'] = accumulated_metrics.get('crop_lpips', 0.0) + batch_lpips
            heat, score = lpips_sliding_window(
                lpips_metric,
                output,
                target_img,
                patch_size=128,
                return_heatmap=True,
                score_mode="p95"
            )

            accumulated_metrics["lpips_sliding"] = score
            
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': 0.0}

    # Promediar métricas
    final_metrics = {f"val_{k}": v / num_batches for k, v in accumulated_metrics.items()}
    # Asegurar que existe val_loss
    final_metrics['val_loss'] = final_metrics.get('val_total', 0.0)
    
    return final_metrics



def lpips_sliding_window(
    lpips_metric,
    img_pred,
    img_target,
    patch_size=128,
    stride=2,   # overlap del 50%
    return_heatmap=False,
    score_mode="mean",        # "mean", "max", "p95", None
):
    """
    LPIPS basado en sliding windows. Cubre completamente la imagen y
    permite generar un heatmap perceptual denso.

    Args:
        lpips_metric: instancia de TorchMetrics LPIPS
        img_pred: tensor (B, 3, H, W)
        img_target: tensor (B, 3, H, W)
        patch_size: tamaño de ventana (64, 128, 256…)
        stride: paso del sliding window. Stride = patch_size//2 = overlap 50%
        return_heatmap: True para devolver heatmap perceptual
        score_mode: "mean", "max", "p95", None

    Returns:
        heatmap (opcional), score_global (opcional)
    """
    stride = patch_size // stride

    B, C, H, W = img_pred.shape

    # Extraer patches (tipo unfold)
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)

    pred_patches = unfold(img_pred)  # (B, C*ps*ps, N_patches)
    tgt_patches = unfold(img_target)

    # Reorganizar: (N_patches_total, 3, ps, ps)
    npatches = pred_patches.shape[-1]

    pred_patches = pred_patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)
    tgt_patches = tgt_patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)

    # LPIPS en batch → muy rápido
    with torch.no_grad():
        patch_scores = lpips_metric(pred_patches, tgt_patches)  # shape (npatches,)

    # ---- Opcional: Heatmap ----
    heatmap = None
    if return_heatmap:
        # Reconstruir heatmap a la forma (B, H/stride, W/stride)
        grid_h = (H - patch_size) // stride + 1
        grid_w = (W - patch_size) // stride + 1

        heatmap = patch_scores.reshape(B, grid_h, grid_w)

    # ---- Opcional: Score global ----
    score = None
    if score_mode is not None:
        if score_mode == "mean":
            score = patch_scores.mean().item()
        elif score_mode == "max":
            score = patch_scores.max().item()
        elif score_mode == "p95":
            score = patch_scores.quantile(0.95).item()
        else:
            raise ValueError("score_mode inválido")

    return heatmap, score
