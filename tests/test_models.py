"""
Pruebas para los modelos
"""

import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet, HybridLoss


@pytest.fixture
def device():
    return torch.device('cpu')


def test_unet_creation(device):
    """Prueba creación del modelo U-Net"""
    model = UNet(in_channels=3, out_channels=3, base_channels=64)
    model = model.to(device)
    
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0


def test_unet_forward(device):
    """Prueba forward pass del modelo"""
    model = UNet(in_channels=3, out_channels=3, base_channels=32)
    model = model.to(device)
    model.eval()
    
    # Entrada: batch de 1 imagen 256x256 RGB
    x = torch.randn(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (1, 3, 256, 256)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_unet_different_sizes(device):
    """Prueba U-Net con diferentes tamaños de imagen"""
    model = UNet(in_channels=3, out_channels=3, base_channels=32)
    model = model.to(device)
    model.eval()
    
    for size in [128, 256, 512]:
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 3, size, size)


def test_hybrid_loss(device):
    """Prueba función de pérdida híbrida"""
    loss_fn = HybridLoss()
    loss_fn = loss_fn.to(device)
    
    pred = torch.rand(2, 3, 256, 256, device=device)
    target = torch.rand(2, 3, 256, 256, device=device)
    
    losses = loss_fn(pred, target)
    
    assert 'total' in losses
    assert 'l1' in losses
    assert 'ssim' in losses
    assert 'perceptual' in losses
    assert losses['total'].item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
