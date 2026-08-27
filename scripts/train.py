"""
Script principal de entrenamiento para CopyAir
Uso: python scripts/train.py --config configs/params.yaml
"""
import tempfile
from PIL import Image
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
import time
import sys
import os
import mlflow
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PairedImageDataset, get_transforms
from src.models import HybridLoss
from src.models.factory import get_model, get_optimizer
from src.training.train import train_epoch, validate
from src.training.schedulers import get_scheduler
from src.utils.mlflow_utils import MLflowLogger
from src.utils.common_utils import GracefulKiller, load_config, setup_device, tensor_to_numpy

# Configurar logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

logger = logging.getLogger(__name__)


def setup_data(config: dict, device: torch.device):
    """Configura datasets y dataloaders"""
    logger.info("Cargando datos...")
    
    train_transform = get_transforms(
        img_size=config['augmentation']['img_size'],
        augment=config['augmentation']['enabled'],
        aug_config=config['augmentation']
    )
    val_transform = get_transforms(
        img_size=config['augmentation']['img_size'],
        augment=False,
        aug_config=config['augmentation']
    )
    
    mask_config = config.get('masked_loss', {'enabled': False})
    
    if config['training'].get('val_split', 0) == 0:
        logger.info("Modo: Entrenar con TODO el dataset (sin split de validación estático)")
        
        train_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=train_transform,
            mask_config=mask_config
        )
        
        native_val_transform = A.Compose([
            A.LongestMaxSize(max_size=1080),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={'image1': 'image'})
        
        full_val_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=native_val_transform,
            mask_config={'enabled': False}
        )
        
        val_samples = config['training'].get('val_samples', 4)
        val_sampler = RandomSampler(full_val_dataset, replacement=True, num_samples=val_samples)
        
        val_loader = DataLoader(
            full_val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', True)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', True)
        )
        
        logger.info(f"Dataset Total: {len(train_dataset)} imágenes")
        logger.info(f"Validación: {val_samples} muestras aleatorias")
        
    else:
        base_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=None,
            mask_config=mask_config
        )
        
        val_size = int(len(base_dataset) * config['training']['val_split'])
        train_size = len(base_dataset) - val_size
        
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size])
        
        full_train_ds = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=train_transform,
            mask_config=mask_config
        )
        
        full_val_ds = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=val_transform,
            mask_config=mask_config
        )
        
        train_dataset = torch.utils.data.Subset(full_train_ds, train_subset.indices)
        val_dataset = torch.utils.data.Subset(full_val_ds, val_subset.indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', True)
        )
        
        logger.info(f"Dataset Total: {len(base_dataset)} imágenes (Train: {train_size}, Val: {val_size})")
    
    return train_loader, val_loader


def setup_model_and_optimizer(config: dict, device: torch.device, train_loader=None):
    """Configura modelo, optimizador y función de pérdida"""
    arch = config['model'].get('architecture', 'nafnet')
    logger.info(f"Inicializando modelo: {arch} (size: {config['model'].get('size', 'base')})")
    
    model = get_model(config['model'])
    model = model.to(device)
    
    optimizer = get_optimizer(model, config['training'])
    scheduler_config = config['training'].get('scheduler', {'type': 'cosine'})
    
    if scheduler_config.get('type') == 'onecycle' and train_loader is not None:
        if 'params' not in scheduler_config:
            scheduler_config['params'] = {}
        scheduler_config['params']['steps_per_epoch'] = len(train_loader)
        scheduler_config['params']['epochs'] = config['training']['epochs']
        scheduler_config['params']['max_lr'] = config['training']['learning_rate']
    
    scheduler = get_scheduler(optimizer, scheduler_config)
    
    # Pérdida Híbrida
    loss_cfg = config.get('loss', {})
    loss_fn = HybridLoss(
        lambda_charbonnier=float(loss_cfg.get('lambda_charbonnier', 1.0)),
        lambda_perceptual=float(loss_cfg.get('lambda_perceptual', 0.8)),
        lambda_dino=float(loss_cfg.get('lambda_dino', 0.1)),
        lambda_ssim=float(loss_cfg.get('lambda_ssim', 0.0)),
        device=str(device)
    ).to(device)
    
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    return model, optimizer, scheduler, loss_fn


def train(config: dict, device: torch.device):
    """Loop principal de entrenamiento"""
    killer = GracefulKiller()
    mlflow_logger = MLflowLogger(config)

    train_loader, val_loader = setup_data(config, device)
    model, optimizer, scheduler, loss_fn = setup_model_and_optimizer(config, device, train_loader)

    scaler = torch.amp.GradScaler('cuda') if config.get('training', {}).get('mixed_precision', True) and device.type == 'cuda' else None


    # Directorio de modelos
    models_dir = Path(config['data'].get('models_dir', 'models'))
    models_dir.mkdir(parents=True, exist_ok=True)

    best_train_loss = float('inf')
    logger.info(f"Iniciando entrenamiento por {config['training']['epochs']} épocas...")

    with mlflow_logger.start_run():
        mlflow_logger.log_params(config)

    
        for epoch in range(config['training']['epochs']):
            if killer.kill_now:
                logger.info(f"Entrenamiento interrumpido por el usuario en la época {epoch}. Guardando checkpoint...")
                ckpt_path = models_dir / f'interrupted_checkpoint_epoch_{epoch}.pth'
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ckpt_path)
                mlflow_logger.log_artifact(str(ckpt_path), artifact_path='checkpoints')
                break

            epoch_start = time.time()

            # Entrenamiento
            train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, scaler=scaler)
            for m_key, m_val in train_metrics.items():
                if m_key in ('loss', 'total'):
                    mlflow_logger.log_metric('train/loss', m_val, step=epoch)
                elif m_key == 'perceptual':
                    mlflow_logger.log_metric('train/lpips', m_val, step=epoch)
                    mlflow_logger.log_metric('train/perceptual', m_val, step=epoch)
                else:
                    mlflow_logger.log_metric(f'train/{m_key}', m_val, step=epoch)


            # Validación
            val_interval = config['training'].get('val_interval', 50)
            if epoch == 0 or (epoch + 1) % val_interval == 0:
                if hasattr(optimizer, 'eval'):
                    optimizer.eval()
                limit_batches = config['training'].get('val_samples', 4) if config['training'].get('val_split', 0) == 0 else None
                val_metrics, val_in, val_gt, val_out = validate(model, val_loader, loss_fn, device, limit_batches=limit_batches)
                if hasattr(optimizer, 'train'):
                    optimizer.train()
                
                mlflow_logger.log_metric('val/psnr', val_metrics['val_psnr'], step=epoch)
                mlflow_logger.log_metric('val/ssim', val_metrics['val_ssim'], step=epoch)
                logger.info(f"[Época {epoch + 1}/{config['training']['epochs']}] Loss: {train_metrics['loss']:.4f} | PSNR: {val_metrics['val_psnr']:.2f} | SSIM: {val_metrics['val_ssim']:.3f}")

            # Guardar mejor modelo
            if train_metrics['loss'] < best_train_loss:
                best_train_loss = train_metrics['loss']
                arch_name = config['model'].get('architecture', 'model')
                best_path = models_dir / f'best_model_{arch_name}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': train_metrics['loss'],
                    'config': config
                }, best_path)

            if scheduler is not None:
                scheduler.step()

        logger.info("¡Entrenamiento completado exitosamente!")
        mlflow_logger.log_model(model)



def main():
    parser = argparse.ArgumentParser(description="Entrenar U-Net para Image-to-Image Translation")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Ruta a archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=None, help='Override número de épocas')
    parser.add_argument('--batch-size', type=int, default=None, help='Override tamaño de batch')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--arch', '--architecture', type=str, default=None, help='Override arquitectura del modelo')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['device'] = args.device
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.arch is not None:
        config['model']['architecture'] = args.arch
    
    device = setup_device(config['device'])
    train(config, device)


if __name__ == '__main__':
    main()
