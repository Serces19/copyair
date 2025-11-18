import torch
from src.models.unet import UNet


def test_unet_forward():
    # Crear modelo con base_channels pequeño para prueba rápida
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    model.eval()

    # Batch size 2, 3 channels, 128x128
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    # Salida debe tener mismo HxW y canales de salida
    assert out.shape == (2, 3, 128, 128)
