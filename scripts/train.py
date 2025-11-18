"""
Script principal de entrenamiento
Uso: python scripts/train.py --config configs/params.yaml
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PairedImageDataset, get_transforms
from src.models import HybridLoss
from src.models.factory import get_model, get_optimizer
from src.training.train import train_epoch, validate
import mlflow
import mlflow.pytorch

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
        augment=config['augmentation']['enabled']
    )
    val_transform = get_transforms(
        img_size=config['augmentation']['img_size'],
        augment=False
    )
    
    # Dataset
    dataset = PairedImageDataset(
        input_dir=config['data']['input_dir'],
        gt_dir=config['data']['gt_dir'],
        transform=train_transform
    )
    
    # Split train/val
    val_size = int(len(dataset) * config['training']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Dataloaders
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
    
    logger.info(f"Dataset: {len(dataset)} imágenes")
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    return train_loader, val_loader


def setup_model_and_optimizer(config: dict, device: torch.device):
    """Configura modelo y optimizador usando el factory"""
    logger.info(f"Inicializando modelo: {config['model'].get('architecture', 'unet')}")
    
    # Modelo usando Factory
    model = get_model(config['model'])
    model = model.to(device)
    
    # Optimizador usando Factory
    optimizer = get_optimizer(model, config['training'])
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=0
    )
    
    # Pérdida
    loss_fn = HybridLoss(
        lambda_l1=config['loss']['lambda_l1'],
        lambda_ssim=config['loss']['lambda_ssim'],
        lambda_perceptual=config['loss']['lambda_perceptual']
    )
    loss_fn = loss_fn.to(device)
    
    logger.info(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, optimizer, scheduler, loss_fn


def train(config: dict, device: torch.device):
    """Loop principal de entrenamiento"""

    # Datos
    train_loader, val_loader = setup_data(config, device)

    # Modelo y optimizador
    model, optimizer, scheduler, loss_fn = setup_model_and_optimizer(config, device)

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
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Iniciando entrenamiento...")

    if mlflow_enabled:
        # Iniciar run de MLflow y registrar parámetros
        run_name = f"{config['model'].get('architecture', 'unet')}_{config['training'].get('optimizer', {}).get('type', 'adam')}"
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_param('model.architecture', config['model'].get('architecture', 'unet'))
            mlflow.log_param('model.activation', config['model'].get('activation', 'relu'))
            mlflow.log_param('model.base_channels', config['model'].get('base_channels'))
            
            mlflow.log_param('training.batch_size', config['training'].get('batch_size'))
            mlflow.log_param('training.epochs', config['training'].get('epochs'))
            mlflow.log_param('training.learning_rate', config['training'].get('learning_rate'))
            mlflow.log_param('training.optimizer', config['training'].get('optimizer', {}).get('type', 'adam'))
            
            mlflow.log_param('loss.type', config['loss'].get('type'))
            mlflow.log_param('data.input_dir', config['data'].get('input_dir'))

            for epoch in range(config['training']['epochs']):
                # Entrenamiento
                train_metrics = train_epoch(
                    model, train_loader, optimizer, loss_fn, device, epoch
                )

                # Registrar métricas de entrenamiento en MLflow
                mlflow.log_metric('train/loss', train_metrics['loss'], step=epoch)

                # Validación
                val_metrics = validate(model, val_loader, loss_fn, device)

                # Registrar métricas de validación en MLflow
                mlflow.log_metric('val/loss', val_metrics['val_loss'], step=epoch)
                mlflow.log_metric('val/psnr', val_metrics['psnr'], step=epoch)

                # Scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                mlflow.log_metric('train/lr', current_lr, step=epoch)

                logger.info(
                    f"Época {epoch + 1}/{config['training']['epochs']} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"PSNR: {val_metrics['psnr']:.2f}"
                )

                # Early stopping
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0

                    # Guardar mejor modelo
                    best_path = Path(config['data']['models_dir']) / 'best_model.pth'
                    torch.save(model.state_dict(), best_path)
                    logger.info("✓ Mejor modelo guardado")

                    # Registrar checkpoint como artefacto
                    mlflow.log_artifact(str(best_path), artifact_path='checkpoints')
                else:
                    patience_counter += 1
                    if patience_counter >= config['training']['early_stopping_patience']:
                        logger.info(f"Early stopping después de {epoch + 1} épocas")
                        break

                # Checkpoint periódico
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    ckpt_path = Path(config['data']['models_dir']) / f'checkpoint_epoch_{epoch + 1}.pth'
                    torch.save(model.state_dict(), ckpt_path)
                    mlflow.log_artifact(str(ckpt_path), artifact_path='checkpoints')

            logger.info("¡Entrenamiento completado!")

            # Log final model to MLflow
            try:
                mlflow.pytorch.log_model(model, artifact_path='model')
            except Exception as e:
                logger.warning(f"No se pudo loggear el modelo en MLflow: {e}")

    else:
        # Si MLflow está deshabilitado, solo corre el loop normal
        for epoch in range(config['training']['epochs']):
            train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()
            logger.info(
                f"Época {epoch + 1}/{config['training']['epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"PSNR: {val_metrics['psnr']:.2f}"
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
