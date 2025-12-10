import sys
from pathlib import Path
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import get_model

def load_config(path='configs/params.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def verify_model(config, arch_name, custom_cfg=None):
    print(f"--- Verifying {arch_name} ---")
    
    # Update config for this architecture
    config['model']['architecture'] = arch_name
    if custom_cfg:
        for k, v in custom_cfg.items():
            config['model'][k] = v
            
    try:
        model = get_model(config['model'])
        model.eval()
        print(f"✅ Instantiated {arch_name}")
        
        # Determine input channels
        in_ch = config['model']['in_channels']
        if arch_name == 'modern_unet':
            in_ch += config['model'].get('input_map_channels', 0)
        
        # Forward Pass
        x = torch.randn(1, in_ch, 256, 256)
        
        # For ModernUNet, we might need cond_vector if FiLM is enabled
        kwargs = {}
        if arch_name == 'modern_unet':
            if config['model']['modern'].get('use_film', False) or config['model']['modern'].get('use_adain', False):
                 cond_dim = config['model']['modern'].get('cond_dim', 128)
                 kwargs['cond_vector'] = torch.randn(1, cond_dim)
        
        with torch.no_grad():
            y = model(x, **kwargs)
        
        # Handle dict output from ModernUNet if configured
        if isinstance(y, dict):
            print(f"✅ Forward Pass successful. Output keys: {y.keys()}")
            y_rgb = y['rgb']
        else:
            y_rgb = y
            print(f"✅ Forward Pass successful. Output shape: {y_rgb.shape}")
            
        assert y_rgb.shape == (1, 3, 256, 256)
        print("✅ Shape Check Passed\n")
        return True
    
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    config = load_config()
    
    results = []
    
    # 1. Basic UNet
    results.append(verify_model(config, 'basic_unet'))
    
    # 2. Smart UNet
    results.append(verify_model(config, 'smart_unet', {}))
    
    # 3. Residual UNet
    results.append(verify_model(config, 'residual_unet'))
    
    # 4. Modern UNet
    results.append(verify_model(config, 'modern_unet'))
    
    if all(results):
        print("🎉 ALL MODELS VERIFIED SUCCESSFULLY")
    else:
        print("⚠️ SOME MODELS FAILED VERIFICATION")

if __name__ == '__main__':
    main()
