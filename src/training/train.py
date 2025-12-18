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
    epoch: int,
    discriminator: nn.Module = None,
    optimizer_d: Optimizer = None,
    config: Dict = None
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
    
    if discriminator:
        discriminator.train()
        # Inicializar losses GAN si no existen en loss_fn
        if not hasattr(loss_fn, 'gan_loss'):
             from src.models.losses import GANLoss, FeatureMatchingLoss
             loss_fn.gan_loss = GANLoss().to(device)
             loss_fn.feature_matching_loss = FeatureMatchingLoss().to(device)
             
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
        
        # Preparar argumentos extra para modelos como ModernUNet
        kwargs = {}
        arch = config['model'].get('architecture', 'unet').lower()
        if arch == 'modern_unet':
            # Por ahora pasamos un vector de ceros (o aleatorio si se prefiere)
            # Esto activa el flujo de FiLM/AdaIN en el modelo
            cond_dim = config['model'].get('modern', {}).get('cond_dim', 128)
            kwargs['cond_vector'] = torch.zeros(input_img.size(0), cond_dim).to(device)
            # Podríamos pasar ruido aleatorio para forzar robustez:
            # kwargs['cond_vector'] = torch.randn(input_img.size(0), cond_dim).to(device)

        output = model(input_img, **kwargs)
        
        # Manejar salida multi-head (diccionario)
        if isinstance(output, dict):
            output = output['rgb'] # Usamos la salida RGB principal para la loss común
        
        if discriminator is not None and optimizer_d is not None:
            # --- GAN Training ---
            
            # 1. Train Discriminator
            optimizer_d.zero_grad()
            
            # Real
            # Concatenamos input + target para Conditional GAN (cGAN)
            real_ab = torch.cat((input_img, target_img), 1)
            pred_real, _ = discriminator(real_ab)
            loss_d_real = loss_fn.gan_loss(pred_real, True)
            
            # Fake
            # Detach para no backpropagar al generador en este paso
            fake_ab = torch.cat((input_img, output.detach()), 1)
            pred_fake, _ = discriminator(fake_ab) 
            loss_d_fake = loss_fn.gan_loss(pred_fake, False)
            
            # Total Discriminator Loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            accumulated_metrics['d_loss'] = accumulated_metrics.get('d_loss', 0.0) + loss_d.item()
            
            # 2. Train Generator (Adversarial)
            # Ya hemos hecho zero_grad del optimizador del generador arriba? NO, hicimos antes de model(input)
            # Pero necesitamos volver a calcular output para grafos?
            # En Pytorch estándar, si retain_graph=True en backward anterior o re-forward.
            # Aquí, 'output' ya se usó para loss principal (L1/LPIPS). 
            # PERO 'loss.backward()' ya se llamó para la loss "normal" (Pixel/Perceptual).
            # Si queremos sumar la loss GAN al generador, deberíamos haberla sumado ANTES de backward().
            
            # REFACTORING LOOP:
            # Calculamos TODAS las losses del generador y hacemos UN solo backward.
            
            # Reiniciamos el grafo para el paso correcto:
            # A. Discriminator Step (ya hecho arriba con detach)
            # B. Generator Step (incluye G_GAN + L1 + LPIPS...)
            
            # Dado que loss.backward() ya se ejecutó para pixels/lpips en el código original (líneas 62-67),
            # tenemos un problema: los gradientes de G ya se calcularon parcialmente.
            # IDEALMENTE: Sumar todo a 'loss' antes de backward.
            
            # CORRECCIÓN EN EL PROCESO:
            # Moveremos el backward original para después del cálculo de GAN.
            
            pass # Placeholder para indicar que cambiamos la lógica abajo
        
        # --- Cálculo de Loss del Generador (Pixel + Perceptual + GAN) ---
        
        # Reset Grads (si no se hizo antes, pero train_epoch original lo hace al inicio)
        # optimizer.zero_grad() # Ya está en línea 59
        
        # Si ya se hizo output = model(input), lo usamos.
        
        # Losses Estándar
        losses_dict = loss_fn(output, target_img, mask=mask)
        loss_g_total = losses_dict['total']
        
        # Add GAN Loss to Generator
        if discriminator is not None:
             # Adversarial Loss: Queremos que D crea que es Real
             fake_ab_g = torch.cat((input_img, output), 1) # Sin detach!
             pred_fake_g, pred_features = discriminator(fake_ab_g)
             
             loss_g_gan = loss_fn.gan_loss(pred_fake_g, True)
             
             # Feature Matching Loss
             # Necesitamos features reales (sin gradientes)
             with torch.no_grad():
                 real_ab_feat = torch.cat((input_img, target_img), 1)
                 _, real_features = discriminator(real_ab_feat)
                 
             loss_fm = loss_fn.feature_matching_loss(pred_features, real_features)
             
             # Weights from config
             lambda_gan = config['gan'].get('lambda_gan', 0.01)
             lambda_fm = config['gan'].get('lambda_feature_matching', 10.0)
             
             loss_g_total += (loss_g_gan * lambda_gan) + (loss_fm * lambda_fm)
             
             # Log metrics
             losses_dict['g_gan'] = loss_g_gan.item()
             losses_dict['g_fm'] = loss_fm.item()
             accumulated_metrics['g_gan_loss'] = accumulated_metrics.get('g_gan_loss', 0.0) + loss_g_gan.item()
             
        
        # Backward (Generator)
        loss_g_total.backward() # Un solo backward acumulado
        
        # Gradient Clipping
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
    device: torch.device,
    limit_batches: int = None
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
        limit_batches: Límite de batches a procesar (para validación rápida o random sampling)
    
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
        for i, batch in enumerate(val_loader):
            if limit_batches is not None and i >= limit_batches:
                break
            input_img = batch['input'].to(device)
            target_img = batch['gt'].to(device)
            
            # Preparar argumentos extra
            kwargs = {}
            # Necesitamos acceder a la config del modelo. 
            # Como validate no recibe config completa, la inferimos o podríamos pasarla.
            # Pero podemos chequear si el modelo es ModernUNet por sus atributos.
            if hasattr(model, 'use_film') and hasattr(model, 'cond_mlp'):
                from src.models.factory import get_model # Not ideal inside loop, but safe if architecture is check
                # Mejor: Solo chequear si acepta cond_vector
                cond_dim = getattr(model, 'cond_dim', 128) # Asumimos o leemos de atributo si existe
                if not hasattr(model, 'cond_dim'): # Fallback robusto
                     # Intentamos leer de la config si estuviera disponible, sino 128
                     cond_dim = 128 
                kwargs['cond_vector'] = torch.zeros(input_img.size(0), cond_dim).to(device)

            output = model(input_img, **kwargs)
            
            if isinstance(output, dict):
                output = output['rgb']
            
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
            heat, score = lpips_sliding_window(
                lpips_metric,
                output,
                target_img,
                patch_size=256,
                return_heatmap=False,
                score_mode="mean"
            )

            accumulated_metrics["lpips_sliding"] = accumulated_metrics.get("lpips_sliding", 0.0) + score
            lpips_metric.reset() # Resetear métrica para evitar leak
            
            num_batches += 1
    
    if num_batches == 0:
        return {'val_loss': 0.0}

    # Promediar métricas
    final_metrics = {f"val_{k}": v / num_batches for k, v in accumulated_metrics.items()}
    # Asegurar que existe val_loss
    final_metrics['val_loss'] = final_metrics.get('val_total', 0.0)
    
    # Retornar también input y target del último batch para visualización
    return final_metrics, input_img, target_img, output



def lpips_sliding_window(
    lpips_metric,
    img_pred: torch.Tensor,
    img_target: torch.Tensor,
    patch_size: int = 128,
    stride: int = 2,             # si stride=2 → patch_size//2, overlap 50%
    return_heatmap: bool = False,
    score_mode: str = "mean",    # "mean", "max", "p95", None
):
    """
    LPIPS sliding-window seguro y robusto para imágenes de cualquier tamaño.

    - Usa torch.nn.Unfold para extraer todos los parches.
    - Calcula LPIPS por parche (batching masivo, muy eficiente).
    - Puede generar un heatmap perceptual (grid de LPIPS por tile).
    - score_mode permite escoger cómo combinar los parches.

    Args:
        lpips_metric: instancia de TorchMetrics LPIPS ya movida a device
        img_pred: tensor (B, 3, H, W)
        img_target: tensor (B, 3, H, W)
        patch_size: tamaño de ventana (64, 128, 256…)
        stride: define overlap → stride_real = patch_size // stride
        return_heatmap: True para devolver heatmap perceptual
        score_mode: "mean", "max", "p95" o None

    Returns:
        heatmap (o None), score_global (o None)
    """

    # -------- 1. Normalización del stride --------
    # Si stride=2 → stride_real = patch_size//2 (overlap 50%)
    stride_real = patch_size // stride

    B, C, H, W = img_pred.shape

    # -------- 2. Caso especial: imagen más pequeña que el patch --------
    if H < patch_size or W < patch_size:
        # fallback: LPIPS normal en imagen completa
        with torch.no_grad():
            score = lpips_metric(img_pred, img_target).item()

        heatmap = None
        if return_heatmap:
            heatmap = torch.tensor([[score]], device=img_pred.device)

        return heatmap, score

    # -------- 3. Extraer parches usando Unfold --------
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride_real)

    pred_patches = unfold(img_pred)    # (B, C*ps*ps, Npatches)
    tgt_patches  = unfold(img_target)

    npatches = pred_patches.shape[-1]

    # Si no hay parches → fallback (no debería pasar, pero por seguridad)
    if npatches == 0:
        with torch.no_grad():
            score = lpips_metric(img_pred, img_target).item()
        heatmap = None
        return heatmap, score

    # -------- 4. Reorganizar a batch gigante de parches --------
    # (B, C*ps*ps, N) → (B, N, C*ps*ps) → (B*N, C, ps, ps)
    pred_patches = pred_patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)
    tgt_patches  = tgt_patches.permute(0, 2, 1).reshape(-1, C, patch_size, patch_size)

    # -------- 5. LPIPS por parche --------
    with torch.no_grad():
        patch_scores = lpips_metric(pred_patches, tgt_patches)  # (B*Npatches,)
        patch_scores = patch_scores.squeeze()

    # -------- 6. Crear heatmap seguro --------
    heatmap = None
    if return_heatmap:
        # tiles horizontales/verticales
        grid_h = (H - patch_size) // stride_real + 1
        grid_w = (W - patch_size) // stride_real + 1
        expected = grid_h * grid_w * B

        if expected == patch_scores.numel():
            # reshape perfecto
            heatmap = patch_scores.reshape(B, grid_h, grid_w)
        else:
            # fallback: heatmap 1D por patches
            heatmap = patch_scores.reshape(B, -1)

    # -------- 7. Score global --------
    score = None
    if score_mode is not None:

        if patch_scores.numel() == 1:
            # caso seguro para patch único
            score = patch_scores.item()

        else:
            if score_mode == "mean":
                score = patch_scores.mean().item()
            elif score_mode == "max":
                score = patch_scores.max().item()
            elif score_mode == "p95":
                score = torch.quantile(patch_scores, 0.95).item()
            else:
                raise ValueError(f"score_mode inválido: {score_mode}")

    return heatmap, score


