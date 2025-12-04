import sys
import os
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.swin_unet import SwinV2UNet

def test_swin_unet():
    print("Initializing SwinV2UNet...")
    try:
        model = SwinV2UNet(img_size=256, in_channels=3, out_channels=3, swin_type='tiny', pretrained=True)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        
        assert y.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {y.shape}"
        print("Forward pass successful! Output shape matches input.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_swin_unet()
