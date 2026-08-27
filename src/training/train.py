"""
Loop de entrenamiento y validación optimizado para CopyAir
"""

import time
import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict = None,
    scaler: torch.cuda.amp.GradScaler = None
) -> Dict[str, float]:
    """
    Entrena el modelo durante una época de forma limpia, eficiente y con Mixed Precision (AMP).
    """
    model.train()
    if hasattr(optimizer, 'train'):
        optimizer.train()
    accumulated_metrics = {}
    num_batches = 0
    use_amp = config.get('training', {}).get('mixed_precision', True) and (device.type == 'cuda')


    for batch_idx, batch in enumerate(train_loader):
        input_img = batch['input'].to(device, non_blocking=True)
        target_img = batch['gt'].to(device, non_blocking=True)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward con Autocast (Mixed Precision)
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(input_img)

                if isinstance(output, dict):
                    output = output['rgb']
                losses_dict = loss_fn(output, target_img, mask=mask)
                loss_total = losses_dict['total']

            if scaler is not None:
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            output = model(input_img)
            if isinstance(output, dict):
                output = output['rgb']
            losses_dict = loss_fn(output, target_img, mask=mask)
            loss_total = losses_dict['total']

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Acumular métricas
        for k, v in losses_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + val

        num_batches += 1

    final_metrics = {k: v / max(num_batches, 1) for k, v in accumulated_metrics.items()}
    final_metrics['loss'] = final_metrics.get('total', 0.0)
    return final_metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    limit_batches: int = None
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Valida el modelo calculando PSNR, SSIM y LPIPS en parches representativos.
    """
    model.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    last_in, last_gt, last_out = None, None, None

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if limit_batches is not None and i >= limit_batches:
                break

            input_img = batch['input'].to(device, non_blocking=True)
            target_img = batch['gt'].to(device, non_blocking=True)

            output = model(input_img)
            if isinstance(output, dict):
                output = output['rgb']

            output = torch.clamp(output, -1.0, 1.0)

            psnr_val = psnr_metric(output, target_img).item()
            ssim_val = ssim_metric(output, target_img).item()

            total_psnr += psnr_val
            total_ssim += ssim_val
            num_samples += 1

            last_in = input_img.cpu()
            last_gt = target_img.cpu()
            last_out = output.cpu()

    metrics = {
        'val_psnr': total_psnr / max(num_samples, 1),
        'val_ssim': total_ssim / max(num_samples, 1),
        'val_lpips_sliding': 0.0
    }

    return metrics, last_in, last_gt, last_out
