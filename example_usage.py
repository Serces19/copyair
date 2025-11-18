"""Ejemplo de uso del pipeline completo"""

from pathlib import Path
import torch
from src.data import PairedImageDataset, get_transforms
from src.models import UNet, HybridLoss
from src.training.train import train_epoch, validate
from torch.utils.data import DataLoader
from torch.optim import Adam

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
BATCH_SIZE = 8

# 1. Cargar datos
print("📁 Cargando datos...")
transform = get_transforms(img_size=IMG_SIZE, augment=True)

dataset = PairedImageDataset(
    input_dir='data/03_processed/input',
    gt_dir='data/03_processed/ground_truth',
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 2. Crear modelo
print("🧠 Inicializando modelo U-Net...")
model = UNet(in_channels=3, out_channels=3, base_channels=64).to(DEVICE)
print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")

# 3. Optimizador y pérdida
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = HybridLoss().to(DEVICE)

# 4. Entrenamiento (1 época como ejemplo)
print("\n🚀 Iniciando entrenamiento...")
for epoch in range(1):
    metrics = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch)
    print(f"   Época {epoch + 1} | Loss: {metrics['loss']:.4f}")

print("\n✅ ¡Listo! Ahora puedes entrenar el modelo completo con:")
print("   python scripts/train.py --config configs/params.yaml")
