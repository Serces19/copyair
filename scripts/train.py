"""
Script principal de entrenamiento
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
    
    # Transformaciones
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
    
    # Configuración de máscara
    mask_config = config.get('masked_loss', {'enabled': False})
    
    if config['training']['val_split'] == 0:
        logger.info("Modo: Entrenar con TODO el dataset (sin split de validación estático)")
        
        train_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=train_transform,
            mask_config=mask_config
        )
        
        # Validación: Resolución nativa (con límite para evitar OOM)
        native_val_transform = A.Compose([
            A.LongestMaxSize(max_size=1920),
            A.PadIfNeeded(min_height=1, min_width=1, pad_height_divisor=32, pad_width_divisor=32),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ])
        
        full_val_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=native_val_transform,
            mask_config={'enabled': False}
        )
        
        val_sampler = RandomSampler(full_val_dataset, replacement=True, num_samples=1)
        
        val_loader = DataLoader(
            full_val_dataset,
            batch_size=1,
            sampler=val_sampler,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        logger.info(f"Dataset Total: {len(train_dataset)} imágenes")
        
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
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )
        
        logger.info(f"Dataset Total: {len(base_dataset)} imágenes")
        logger.info(f"Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader


def setup_model_and_optimizer(config: dict, device: torch.device, train_loader=None):
    """Configura modelo, optimizador y scheduler"""
    logger.info(f"Inicializando modelo: {config['model'].get('architecture', 'unet')}")
    
    model = get_model(config['model'])
    model = model.to(device)
    
    optimizer = get_optimizer(model, config['training'])
    
    scheduler_config = config['training'].get('scheduler', {'type': 'constant'})
    
    if scheduler_config.get('type') == 'onecycle' and train_loader is not None:
        if 'params' not in scheduler_config:
            scheduler_config['params'] = {}
        scheduler_config['params']['steps_per_epoch'] = len(train_loader)
        scheduler_config['params']['epochs'] = config['training']['epochs']
        scheduler_config['params']['max_lr'] = config['training']['learning_rate']
    
    scheduler = get_scheduler(optimizer, scheduler_config)
    logger.info(f"Scheduler: {scheduler_config.get('type', 'constant')}")
    
    loss_fn = HybridLoss(
        lambda_l1=config['loss']['lambda_l1'],
        lambda_ssim=config['loss']['lambda_ssim'],
        lambda_perceptual=config['loss']['lambda_perceptual'],
        lambda_laplacian=config['loss'].get('lambda_laplacian', 0.05),
        lambda_ffl=config['loss'].get('lambda_ffl', 0.0),
        lambda_dreamsim=config['loss'].get('lambda_dreamsim', 0.0),
        lambda_charbonnier=config['loss'].get('lambda_charbonnier', 0.0),
        lambda_sobel=config['loss'].get('lambda_sobel', 0.0),
        device=str(device)
    )
    loss_fn = loss_fn.to(device)
    
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, scheduler, loss_fn


def log_training_sample(train_loader, mlflow_logger, device):
    """Loguea una muestra de los datos de entrenamiento para visualizar aumentaciones"""
    try:
        logger.info("Logueando muestra de datos de entrenamiento...")
        batch = next(iter(train_loader))
        # Tomar hasta 4 imágenes
        n_samples = min(4, batch['input'].size(0))
        
        inputs = batch['input'][:n_samples].to(device)
        gts = batch['gt'][:n_samples].to(device)
        
        vis_list = []
        for i in range(n_samples):
            inp = tensor_to_numpy(inputs[i])
            gt = tensor_to_numpy(gts[i])
            # Concatenar verticalmente input y gt
            pair = np.concatenate([inp, gt], axis=0)
            vis_list.append(pair)
            
        # Concatenar pares horizontalmente
        grid = np.concatenate(vis_list, axis=1)
        grid = (grid * 255).astype(np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            Image.fromarray(grid).save(f.name, quality=90)
            mlflow_logger.log_artifact(f.name, artifact_path='data_samples')
            os.unlink(f.name)
            
    except Exception as e:
        logger.warning(f"No se pudo loggear muestra de entrenamiento: {e}")


def train(config: dict, device: torch.device):
    """Loop principal de entrenamiento"""
    
    killer = GracefulKiller()
    mlflow_logger = MLflowLogger(config)

    # Datos
    train_loader, val_loader = setup_data(config, device)

    # Modelo
    model, optimizer, scheduler, loss_fn = setup_model_and_optimizer(config, device, train_loader)

    # Directorio de modelos
    Path(config['data']['models_dir']).mkdir(parents=True, exist_ok=True)

    best_train_loss = float('inf')
    last_saved_epoch = -1
    patience_counter = 0
    
    # Variables para reutilizar datos de validación
    latest_val_input = None
    latest_val_target = None
    latest_val_pred = None

    logger.info("Iniciando entrenamiento...")

    # Iniciar MLflow run
    mlflow_logger.start_run()
    
    try:
        mlflow_logger.log_params(config)

        min_delta = 0.02
        epsilon = 1e-8

        for epoch in range(config['training']['epochs']):
            if killer.kill_now:
                logger.info("🛑 Deteniendo entrenamiento por señal de usuario.")
                # Checkpoint de emergencia
                ckpt_path = Path(config['data']['models_dir']) / f'interrupted_checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                }, ckpt_path)
                mlflow_logger.log_artifact(str(ckpt_path), artifact_path='checkpoints')
                break

            epoch_start = time.time()

            # --- Entrenamiento ---
            train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
            mlflow_logger.log_metric('train/loss', train_metrics['loss'], step=epoch)

            # Loguear muestra de entrenamiento
            #log_training_sample(train_loader, mlflow_logger, device)

            # --- Validación ---
            val_metrics = None
            val_interval = config['training'].get('val_interval', 50)
            
            if epoch == 0 or epoch % val_interval == 0:
                t_val_start = time.time()
                # Ahora validate retorna también las imágenes del último batch
                val_metrics, val_in, val_gt, val_out = validate(model, val_loader, loss_fn, device)
                
                # Guardar para visualización posterior
                latest_val_input = val_in
                latest_val_target = val_gt
                latest_val_pred = val_out
                
                val_time = time.time() - t_val_start
                
                mlflow_logger.log_metric('val/psnr', val_metrics['val_psnr'], step=epoch)
                mlflow_logger.log_metric('val/ssim', val_metrics['val_ssim'], step=epoch)
                mlflow_logger.log_metric('val/crop_lpips', val_metrics['val_lpips_sliding'], step=epoch)
                mlflow_logger.log_metric('time/val_duration', val_time, step=epoch)
                
                logger.info(f"[Validación] PSNR: {val_metrics['val_psnr']:.2f} | SSIM: {val_metrics['val_ssim']:.3f} | LPIPS: {val_metrics['val_lpips_sliding']:.4f}")

            # --- Visualización ---
            if (epoch + 1) % config['training'].get('viz_interval', 250) == 0:
                try:
                    viz_start = time.time()
                    
                    # Usar datos cacheados si existen, si no, intentar obtener del loader (fallback)
                    if latest_val_input is not None:
                        input_vis_t = latest_val_input[0]
                        gt_vis_t = latest_val_target[0]
                        pred_vis_t = latest_val_pred[0]
                    else:
                        # Fallback si viz_interval no coincide con val_interval y no hay cache
                        logger.info("Generando visualización (sin cache de validación)...")
                        batch = next(iter(val_loader))
                        input_vis_t = batch['input'][0].to(device)
                        gt_vis_t = batch['gt'][0].to(device)
                        model.eval()
                        with torch.no_grad():
                            pred_vis_t = model(input_vis_t.unsqueeze(0)).squeeze(0)

                    # Procesar imágenes
                    input_vis = tensor_to_numpy(input_vis_t)
                    gt_vis = tensor_to_numpy(gt_vis_t)
                    pred_vis = tensor_to_numpy(pred_vis_t)
                    
                    comparison = np.concatenate([input_vis, gt_vis, pred_vis], axis=1)
                    comparison = (comparison * 255).astype(np.uint8)
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        Image.fromarray(comparison).save(f.name)
                        mlflow_logger.log_artifact(f.name, artifact_path=f'predictions/epoch_{epoch+1}')
                        os.unlink(f.name)
                    
                    mlflow_logger.log_metric('time/viz_duration', time.time() - viz_start, step=epoch)
                    logger.info(f"✓ Imagen de validación guardada (época {epoch+1})")
                    
                except Exception as e:
                    logger.warning(f"No se pudo guardar imagen de validación: {e}")

            # --- Scheduler ---
            scheduler.step()
            mlflow_logger.log_metric('train/lr', scheduler.get_last_lr()[0], step=epoch)

            # --- Logging Consola ---
            if val_metrics:
                logger.info(
                    f"Época {epoch + 1}/{config['training']['epochs']} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"PSNR: {val_metrics['val_psnr']:.2f} | "
                    f"LPIPS: {val_metrics['val_lpips_sliding']:.4f}"
                )
            else:
                logger.info(f"Época {epoch + 1}/{config['training']['epochs']} | Train Loss: {train_metrics['loss']:.4f}")

            # --- Guardado de Modelos ---
            # Guardar mejor modelo (basado en train loss por few-shot strategy)
            if epoch > 250 and (train_metrics['loss'] < best_train_loss * (1 - min_delta) - epsilon or epoch - last_saved_epoch >= config['training']['early_stopping_patience']):
                last_saved_epoch = epoch
                best_train_loss = train_metrics['loss']
                patience_counter = 0

                arch_name = config['model'].get('architecture', 'unet')
                best_path = Path(config['data']['models_dir']) / f'best_model_{arch_name}.pth'
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': config['model'],
                    'architecture': arch_name,
                    'epoch': epoch,
                    'train_loss': train_metrics['loss']
                }, best_path)
                
                logger.info(f"✓ Mejor modelo guardado (Train Loss: {train_metrics['loss']:.4f})")
                mlflow_logger.log_artifact(str(best_path), artifact_path='checkpoints')
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping después de {epoch + 1} épocas")
                    break

            # Checkpoint periódico
            if (epoch + 1) % config['training']['save_interval'] == 0:
                last_saved_epoch = epoch
                arch_name = config['model'].get('architecture', 'unet')
                ckpt_path = Path(config['data']['models_dir']) / f'checkpoint_{arch_name}_epoch_{epoch + 1}.pth'
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': config['model'],
                    'architecture': arch_name,
                    'epoch': epoch,
                    'train_loss': train_metrics['loss']
                }, ckpt_path)
                mlflow_logger.log_artifact(str(ckpt_path), artifact_path='checkpoints')

            # Tiempo total
            epoch_time = time.time() - epoch_start
            mlflow_logger.log_metric('time/epoch_duration', epoch_time, step=epoch)

        logger.info("¡Entrenamiento completado!")
        mlflow_logger.log_model(model)
        
    finally:
        mlflow_logger.end_run()


def main():
    parser = argparse.ArgumentParser(description="Entrenar U-Net para Image-to-Image Translation")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Ruta a archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo (cuda/cpu)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['device'] = args.device
    
    device = setup_device(config['device'])
    
    train(config, device)


if __name__ == '__main__':
    main()
