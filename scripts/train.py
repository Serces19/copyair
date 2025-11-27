"""
Script principal de entrenamiento
Uso: python scripts/train.py --config configs/params.yaml
"""
import tempfile
from PIL import Image
import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
import time
import signal
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PairedImageDataset, get_transforms
from src.models import HybridLoss
from src.models.factory import get_model, get_optimizer
from src.training.train import train_epoch, validate
from src.training.schedulers import get_scheduler
import mlflow
import mlflow.pytorch
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import RandomSampler


# Configurar logging: asegurarse de que haya un handler que imprima INFO
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_name: str) -> torch.device:
    """Configura el dispositivo (CPU/GPU)"""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Usando CPU")
    return device


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
    
    # Si val_split es 0, usamos TODO para entrenar y validamos con una muestra aleatoria del mismo set (sin augs)
    if config['training']['val_split'] == 0:
        logger.info("Modo: Entrenar con TODO el dataset (sin split de validación estático)")
        
        # Dataset de entrenamiento (Todo el data, con aumentaciones)
        train_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=train_transform,
            mask_config=mask_config
        )
        
        # Dataset de validación (Resolución NATIVA, SIN crop, SIN resize, solo normalize)
        # Para few-shot: validamos en resolución original para ver calidad real
        native_val_transform = A.Compose([
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ])
        
        full_val_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=native_val_transform,
            mask_config={'enabled': False}  # No mask for validation
        )
        
        # Para cumplir "un frame aleatorio por epoch", usamos un Sampler o simplemente shuffle=True
        # y limitamos el loop de validación, o creamos un loader que solo devuelva 1 batch.
        # Aquí usamos RandomSampler con replacement=True y num_samples=1 para sacar 1 imagen aleatoria.
        val_sampler = RandomSampler(full_val_dataset, replacement=True, num_samples=1)
        
        val_loader = DataLoader(
            full_val_dataset,
            batch_size=1, # 1 frame como pidió el usuario
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
        logger.info("Validación: 1 imagen aleatoria del set de entrenamiento (sin augs) por época.")
        
    else:
        # Modo clásico: Split Train/Val
        # Cargamos dataset base para obtener índices
        base_dataset = PairedImageDataset(
            input_dir=config['data']['input_dir'],
            gt_dir=config['data']['gt_dir'],
            transform=None, # No transform yet
            mask_config=mask_config
        )
        
        val_size = int(len(base_dataset) * config['training']['val_split'])
        train_size = len(base_dataset) - val_size
        
        # Split de índices
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size])
        
        # Crear datasets finales con las transformaciones correctas
        # Nota: PairedImageDataset carga archivos basado en directorio, no índices directos fácilmente si no modificamos la clase.
        # Workaround: Instanciar dos datasets completos y usar Subset con los índices generados.
        
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
        
        # Aplicar los índices del split
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
    """Configura modelo, optimizador y scheduler usando factories"""
    logger.info(f"Inicializando modelo: {config['model'].get('architecture', 'unet')}")
    
    # Modelo usando Factory
    model = get_model(config['model'])
    model = model.to(device)
    
    # Optimizador usando Factory
    optimizer = get_optimizer(model, config['training'])
    
    # Scheduler usando Factory (nuevo!)
    scheduler_config = config['training'].get('scheduler', {'type': 'constant'})
    
    # Para OneCycleLR, necesitamos steps_per_epoch
    if scheduler_config.get('type') == 'onecycle' and train_loader is not None:
        if 'params' not in scheduler_config:
            scheduler_config['params'] = {}
        scheduler_config['params']['steps_per_epoch'] = len(train_loader)
        scheduler_config['params']['epochs'] = config['training']['epochs']
        scheduler_config['params']['max_lr'] = config['training']['learning_rate']
    
    scheduler = get_scheduler(optimizer, scheduler_config)
    logger.info(f"Scheduler: {scheduler_config.get('type', 'constant')}")
    
    # Pérdida
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


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        logger.info("\n🛑 Señal de interrupción recibida. Terminando después de esta época...")


def train(config: dict, device: torch.device):
    """Loop principal de entrenamiento"""
    
    # Inicializar killer para manejo de señales
    killer = GracefulKiller()

    # Datos
    train_loader, val_loader = setup_data(config, device)

    # Modelo, optimizador y scheduler
    model, optimizer, scheduler, loss_fn = setup_model_and_optimizer(config, device, train_loader)

    # Crear directorio de modelos
    Path(config['data']['models_dir']).mkdir(parents=True, exist_ok=True)

    # Configurar MLflow si está habilitado
    mlflow_enabled = config.get('mlflow', {}).get('enabled', False)
    if mlflow_enabled:
        tracking_uri = config.get('mlflow', {}).get('tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = config.get('mlflow', {}).get('experiment_name', 'copyair')
        mlflow.set_experiment(experiment_name)

    # Logging
    best_train_loss = float('inf')  # Cambiado a train_loss para few-shot
    last_saved_epoch = -1
    patience_counter = 0

    logger.info("Iniciando entrenamiento...")

    if mlflow_enabled:
        # Iniciar run de MLflow y registrar parámetros
        run_name = f"{config['model'].get('architecture', 'unet')}_{config['training'].get('optimizer', {}).get('type', 'adam')}"
        with mlflow.start_run(run_name=run_name):
            # Log all static params from the loaded config (flatten nested dicts)
            def _flatten(d: dict, parent_key: str = '', sep: str = '.') -> dict:
                items = {}
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.update(_flatten(v, new_key, sep=sep))
                    else:
                        # Convert non-primitive values to JSON strings so MLflow can store them
                        if isinstance(v, (list, tuple, dict)):
                            try:
                                items[new_key] = json.dumps(v)
                            except Exception:
                                items[new_key] = str(v)
                        else:
                            items[new_key] = v
                return items

            try:
                params_to_log = _flatten(config)
                # mlflow.log_params expects string/int/float values; convert None to 'null'
                safe_params = {k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else (v if v is not None else 'null')) for k, v in params_to_log.items()}
                mlflow.log_params(safe_params)
            except Exception as e:
                logger.warning(f"No se pudieron registrar todos los parámetros en MLflow: {e}")

            min_delta = 0.02
            epsilon = 1e-8

            for epoch in range(config['training']['epochs']):
                # Verificar si debemos detenernos ANTES de empezar la época
                if killer.kill_now:
                    logger.info("🛑 Deteniendo entrenamiento por señal de usuario (Ctrl+C).")
                    # Guardar checkpoint de emergencia
                    ckpt_path = Path(config['data']['models_dir']) / f'interrupted_checkpoint_epoch_{epoch}.pth'
                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'model_config': config['model'],
                        'architecture': config['model'].get('architecture', 'unet'),
                        'epoch': epoch,
                    }
                    torch.save(checkpoint_data, ckpt_path)
                    mlflow.log_artifact(str(ckpt_path), artifact_path='checkpoints')
                    break

                # Medir tiempo total de la época y tiempo de entrenamiento por separado
                epoch_start = time.time()

                # Entrenamiento (medimos solo el tiempo de la función train_epoch)
                t_train_start = time.time()
                train_metrics = train_epoch(
                    model, train_loader, optimizer, loss_fn, device, epoch
                )

                # Registrar métricas de entrenamiento en MLflow
                # Solo registramos la pérdida total para evitar ruido
                mlflow.log_metric('train/loss', train_metrics['loss'], step=epoch)

                # Validación (solo cada val_interval épocas para eficiencia)
                val_interval = config['training'].get('val_interval', 50)
                if epoch == 0 or epoch % val_interval == 0:
                    t_val_start = time.time()
                    val_metrics = validate(model, val_loader, loss_fn, device)
                    val_time = time.time() - t_val_start
                    # Registramos todas las métricas de validación
                    mlflow.log_metric('val/loss', val_metrics['val_loss'], step=epoch)
                    mlflow.log_metric('val/psnr', val_metrics['val_psnr'], step=epoch)
                    mlflow.log_metric('val/ssim', val_metrics['val_ssim'], step=epoch)
                    mlflow.log_metric('val/crop_lpips', val_metrics['val_crop_lpips'], step=epoch)
                    mlflow.log_metric('time/val_duration', val_time, step=epoch)
                    logger.info(f"[Validación] Loss: {val_metrics['val_loss']:.4f} | PSNR: {val_metrics['val_psnr']:.2f} | SSIM: {val_metrics['val_ssim']:.3f} | LPIPS(Crop): {val_metrics['val_crop_lpips']:.4f}")
                else:
                    val_metrics = None

                # Visualización de validación cada 200 épocas
                if (epoch + 1) % 200 == 0:
                    try:
                        viz_start = time.time()
                        # Obtener un batch de validación
                        val_batch = next(iter(val_loader))
                        input_img = val_batch['input'][:1].to(device)  # Solo primera imagen
                        gt_img = val_batch['gt'][:1].to(device)
                        
                        # Predicción
                        model.eval()
                        with torch.no_grad():
                            pred_img = model(input_img)
                        
                        # Denormalizar de [-1, 1] a [0, 1] para visualización
                        def denormalize(tensor):
                            """Denormaliza tensor de [-1, 1] a [0, 1]"""
                            return torch.clamp((tensor + 1.0) / 2.0, 0, 1)
                        
                        # Preparar imágenes para logging
                        input_vis = denormalize(input_img[0].cpu()).numpy()
                        gt_vis = denormalize(gt_img[0].cpu()).numpy()  # GT también está en [-1, 1]
                        pred_vis = denormalize(pred_img[0].cpu()).numpy()
                        
                        # Convertir CHW a HWC para guardar
                        import numpy as np
                        input_vis = np.transpose(input_vis, (1, 2, 0))
                        gt_vis = np.transpose(gt_vis, (1, 2, 0))
                        pred_vis = np.transpose(pred_vis, (1, 2, 0))
                        
                        # Concatenar horizontalmente
                        comparison = np.concatenate([input_vis, gt_vis, pred_vis], axis=1)
                        comparison = (comparison * 255).astype(np.uint8)
                        
                        # Guardar temp y loggear
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                            Image.fromarray(comparison).save(f.name)
                            mlflow.log_artifact(f.name, artifact_path=f'predictions/epoch_{epoch+1}')
                            import os
                            os.unlink(f.name)
                        viz_time = time.time() - viz_start
                        mlflow.log_metric('time/viz_duration', viz_time, step=epoch)
                        
                        logger.info(f"✓ Imagen de validación guardada (época {epoch+1})")
                    except Exception as e:
                        logger.warning(f"No se pudo guardar imagen de validación: {e}")

                # Scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                mlflow.log_metric('train/lr', current_lr, step=epoch)

                # Logging
                if val_metrics:
                    logger.info(
                        f"Época {epoch + 1}/{config['training']['epochs']} | "
                        f"Train Loss: {train_metrics['loss']:.4f} | "
                        f"Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"PSNR: {val_metrics['val_psnr']:.2f} | "
                        f"LPIPS: {val_metrics['val_crop_lpips']:.4f}"
                    )
                else:
                    logger.info(
                        f"Época {epoch + 1}/{config['training']['epochs']} | "
                        f"Train Loss: {train_metrics['loss']:.4f}"
                    )

                # Guardar mejor modelo basado en TRAIN_LOSS (few-shot strategy)
                if epoch > 250 and (train_metrics['loss'] < best_train_loss * (1 - min_delta) - epsilon or epoch - last_saved_epoch >= config['training']['early_stopping_patience']):
                    last_saved_epoch = epoch
                    best_train_loss = train_metrics['loss']
                    patience_counter = 0

                    # Guardar mejor modelo con metadatos
                    arch_name = config['model'].get('architecture', 'unet')
                    best_path = Path(config['data']['models_dir']) / f'best_model_{arch_name}.pth'
                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'model_config': config['model'],
                        'architecture': arch_name,
                        'epoch': epoch,
                        'train_loss': train_metrics['loss']
                    }
                    torch.save(checkpoint_data, best_path)
                    logger.info(f"✓ Mejor modelo guardado (Train Loss: {train_metrics['loss']:.4f})")

                    # Registrar checkpoint como artefacto
                    mlflow.log_artifact(str(best_path), artifact_path='checkpoints')
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
                    checkpoint_data = {
                        'model_state_dict': model.state_dict(),
                        'model_config': config['model'],
                        'architecture': arch_name,
                        'epoch': epoch,
                        'train_loss': train_metrics['loss']
                    }
                    torch.save(checkpoint_data, ckpt_path)
                    mlflow.log_artifact(str(ckpt_path), artifact_path='checkpoints')

                # Final de epoch: acumular tiempo total
                epoch_time = time.time() - epoch_start
                mlflow.log_metric('time/epoch_duration', epoch_time, step=epoch)
                
                # Reportar tiempos en log (opcional, para debug en consola)
                logger.info(f"Tiempos: Epoch={epoch_time:.2f}s")
                
                # Verificar killer al FINAL de la época también
                if killer.kill_now:
                    logger.info("🛑 Deteniendo entrenamiento por señal de usuario (Ctrl+C).")
                    break

            logger.info("¡Entrenamiento completado!")

            # Log final model to MLflow
            try:
                mlflow.pytorch.log_model(model, artifact_path='model')
            except Exception as e:
                logger.warning(f"No se pudo loggear el modelo en MLflow: {e}")

    else:
        # Si MLflow está deshabilitado, solo corre el loop normal
        # También agregamos soporte para killer aquí
        killer = GracefulKiller()
        
        for epoch in range(config['training']['epochs']):
            if killer.kill_now:
                logger.info("🛑 Deteniendo entrenamiento por señal de usuario (Ctrl+C).")
                break
                
            train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()
            logger.info(
                f"Época {epoch + 1}/{config['training']['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"PSNR: {val_metrics['val_psnr']:.2f} | "
                f"SSIM: {val_metrics['val_ssim']:.3f} | "
                f"LPIPS: {val_metrics['val_crop_lpips']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Entrenar U-Net para Image-to-Image Translation")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Ruta a archivo de configuración')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    config['device'] = args.device
    
    # Configurar dispositivo
    device = setup_device(config['device'])
    
    # Entrenar
    train(config, device)


if __name__ == '__main__':
    main()
