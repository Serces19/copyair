"""
Script de verificación y testing de todas las arquitecturas de CopyAir.
"""

import sys
from pathlib import Path
import torch
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import get_model


def load_config(path='configs/params.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def verify_model(config, arch_name, custom_cfg=None):
    print(f"\n--- Verificando: {arch_name} ---")


    cfg = config['model'].copy()
    cfg['architecture'] = arch_name
    if custom_cfg:
        cfg.update(custom_cfg)

    try:
        model = get_model(cfg)
        model.eval()

        in_ch = cfg.get('in_channels', 3)
        if arch_name == 'modern_unet':
            in_ch += cfg.get('input_map_channels', 0)

        # Probar con tamaño estándar y tamaño impar/arbitrario
        test_shapes = [(1, in_ch, 256, 256), (1, in_ch, 137, 219)]

        for shape in test_shapes:
            x = torch.randn(*shape)
            kwargs = {}
            if arch_name == 'modern_unet':
                if cfg.get('modern', {}).get('use_film', False) or cfg.get('modern', {}).get('use_adain', False):
                    cond_dim = cfg.get('modern', {}).get('cond_dim', 128)
                    kwargs['cond_vector'] = torch.randn(1, cond_dim)

            with torch.no_grad():
                y = model(x, **kwargs)

            if isinstance(y, dict):
                y_rgb = y['rgb']
            else:
                y_rgb = y

            expected_shape = (shape[0], cfg.get('out_channels', 3), shape[2], shape[3])
            assert y_rgb.shape == expected_shape, f"Shape mismatch: {y_rgb.shape} != {expected_shape}"
            assert not torch.isnan(y_rgb).any(), f"NaNs detected in {arch_name}"

        params_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Instanciado {arch_name} | Parametros: {params_count:,} | Forward Pass OK (Standard & Arbitrary resolution)")
        return True

    except Exception as e:
        print(f"[ERROR] Fallo {arch_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    config = load_config()
    results = []

    models_to_test = [
        ('nafnet', {'size': 'small'}),
        ('nafnet', {'size': 'base'}),
        ('scope_unet', {'size': 'small'}),
        ('scope_unet', {'size': 'base'}),
        ('restormer', {'size': 'tiny'}),
        ('restormer', {'size': 'small'}),
        ('mambair', {'size': 'tiny'}),
        ('mambair', {'size': 'base'}),
        ('convnext', {'size': 'nano'}),
        ('residual_unet', {}),
        ('modern_unet', {}),
        ('smart_unet', {}),
        ('basic_unet', {})
    ]

    for arch, custom_cfg in models_to_test:
        results.append(verify_model(config, arch, custom_cfg))

    print("\n" + "=" * 60)
    if all(results):
        print("[SUCCESS] TODOS LOS MODELOS (13 CONFIGURACIONES) FUERON VERIFICADOS CON EXITO!")
    else:
        print("[WARNING] ALGUNOS MODELOS FALLARON LA VERIFICACION")
    print("=" * 60)


if __name__ == '__main__':
    main()

