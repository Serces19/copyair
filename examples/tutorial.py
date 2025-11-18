"""
Script de ejemplo: Entrenar modelo en modo notebook (sin notebook)
Para ejecutar: python examples/tutorial.py
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from src.data import PairedImageDataset, get_transforms
from src.models import UNet, HybridLoss
from src.training.train import train_epoch, validate

print("=" * 70)
print("CopyAir - Tutorial: Entrenar U-Net desde Python")
print("=" * 70)

# ============================================================================
# 1. CONFIGURACIÓN INICIAL
# ============================================================================

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'input_dir': 'data/03_processed/input',
    'gt_dir': 'data/03_processed/gt',
    'models_dir': 'models',
    'epochs': 5,  # Para tutorial, usamos pocas épocas
    'batch_size': 4,
    'learning_rate': 0.001,
    'img_size': 256,
}

print(f"\n✓ Dispositivo: {CONFIG['device']}")
print(f"✓ Batch size: {CONFIG['batch_size']}")
print(f"✓ Épocas: {CONFIG['epochs']}")

# ============================================================================
# 2. CARGAR DATOS
# ============================================================================

print("\n📁 Cargando datos...")

try:
    # Transformaciones
    train_transform = get_transforms(img_size=CONFIG['img_size'], augment=True)
    val_transform = get_transforms(img_size=CONFIG['img_size'], augment=False)
    
    # Dataset
    dataset = PairedImageDataset(
        input_dir=CONFIG['input_dir'],
        gt_dir=CONFIG['gt_dir'],
        transform=train_transform
    )
    
    print(f"✓ Dataset cargado: {len(dataset)} pares de imágenes")
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Train: {train_size} | Val: {val_size}")
    
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
    print(f"   Verifica que existan archivos en:")
    print(f"   - {CONFIG['input_dir']}")
    print(f"   - {CONFIG['gt_dir']}")
    exit(1)

# ============================================================================
# 3. CREAR MODELO
# ============================================================================

print("\n🧠 Inicializando modelo U-Net...")

model = UNet(in_channels=3, out_channels=3, base_channels=32)  # base_channels bajo para tutorial
model = model.to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Parámetros totales: {total_params:,}")

# ============================================================================
# 4. CONFIGURAR OPTIMIZADOR Y PÉRDIDA
# ============================================================================

print("\n⚙️  Configurando optimizador...")

optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
loss_fn = HybridLoss(lambda_l1=0.6, lambda_ssim=0.2, lambda_perceptual=0.2)
loss_fn = loss_fn.to(CONFIG['device'])

print("✓ Optimizador: Adam")
print("✓ Scheduler: CosineAnnealingLR")
print("✓ Pérdida: Hybrid (L1 + SSIM + Perceptual)")

# ============================================================================
# 5. LOOP DE ENTRENAMIENTO
# ============================================================================

print("\n" + "=" * 70)
print("🚀 INICIANDO ENTRENAMIENTO")
print("=" * 70)

Path(CONFIG['models_dir']).mkdir(parents=True, exist_ok=True)

best_val_loss = float('inf')

for epoch in range(CONFIG['epochs']):
    
    # Entrenamiento
    print(f"\n[Época {epoch + 1}/{CONFIG['epochs']}]")
    train_metrics = train_epoch(
        model, train_loader, optimizer, loss_fn, CONFIG['device'], epoch
    )
    
    # Validación
    val_metrics = validate(model, val_loader, loss_fn, CONFIG['device'])
    
    # Scheduler
    scheduler.step()
    
    # Mostrar resultados
    print(f"  Train Loss: {train_metrics['loss']:.4f}")
    print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
    print(f"  PSNR:       {val_metrics['psnr']:.2f} dB")
    
    # Guardar mejor modelo
    if val_metrics['val_loss'] < best_val_loss:
        best_val_loss = val_metrics['val_loss']
        torch.save(
            model.state_dict(),
            Path(CONFIG['models_dir']) / 'best_model.pth'
        )
        print("  ✓ Mejor modelo guardado")

# ============================================================================
# 6. RESUMEN
# ============================================================================

print("\n" + "=" * 70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 70)

print("\n📊 Resultados:")
print(f"  - Best Val Loss: {best_val_loss:.4f}")
print(f"  - Modelo guardado: {Path(CONFIG['models_dir']) / 'best_model.pth'}")

print("\n🎯 Próximos pasos:")
print("  1. Exporta el modelo:")
print("     torch.onnx.export(model, dummy_input, 'model.onnx')")
print("")
print("  2. Utiliza para inferencia:")
print("     python scripts/predict.py --model models/best_model.pth \\")
print("                                --video input.mp4 --output output.mp4")
print("")
print("  3. Personaliza configuración en: configs/params.yaml")

print("\n📖 Para más información: README.md y DEVELOPMENT.md\n")
