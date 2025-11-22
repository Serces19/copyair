"""
Script de prueba: Verificar que un checkpoint tiene metadatos
Uso: python scripts/check_checkpoint.py --model models/best_model.pth
"""

import argparse
import torch
from pathlib import Path


def check_checkpoint(model_path: str):
    """Verifica el formato de un checkpoint"""
    
    print(f"\n{'='*60}")
    print(f"Analizando checkpoint: {model_path}")
    print(f"{'='*60}\n")
    
    if not Path(model_path).exists():
        print(f"❌ Error: El archivo no existe: {model_path}")
        return
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Verificar formato
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("✓ Formato: NUEVO (con metadatos)")
        print(f"\n📋 Metadatos encontrados:")
        print(f"  • Arquitectura: {checkpoint.get('architecture', 'N/A')}")
        print(f"  • Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"  • Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        if 'model_config' in checkpoint:
            print(f"\n⚙️  Configuración del modelo:")
            for key, value in checkpoint['model_config'].items():
                print(f"  • {key}: {value}")
        
        print(f"\n📊 Tamaño del state_dict: {len(checkpoint['model_state_dict'])} parámetros")
        
    else:
        print("⚠ Formato: ANTIGUO (solo state_dict)")
        print(f"\n📊 Tamaño del state_dict: {len(checkpoint)} parámetros")
        print(f"\n💡 Recomendación: Convierte este checkpoint al nuevo formato usando:")
        print(f"   python scripts/convert_checkpoint.py --model {model_path} --architecture <tu_arquitectura>")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Verificar formato de checkpoint")
    parser.add_argument('--model', type=str, required=True, help='Ruta al checkpoint')
    
    args = parser.parse_args()
    check_checkpoint(args.model)


if __name__ == '__main__':
    main()
