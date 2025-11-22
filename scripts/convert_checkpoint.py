"""
Script de utilidad: Convertir modelos antiguos (solo pesos) al nuevo formato (con metadatos)
Uso: python scripts/convert_checkpoint.py --model models/old_model.pth --architecture nafnet --output models/new_model.pth
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def convert_checkpoint(model_path: str, architecture: str, config_path: str, output_path: str):
    """Convierte checkpoint antiguo a nuevo formato con metadatos"""
    
    print(f"Cargando checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Verificar si ya tiene metadatos
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("⚠ Este checkpoint ya tiene metadatos. No es necesario convertir.")
        print(f"  Arquitectura: {checkpoint['architecture']}")
        return
    
    # Cargar configuración
    config = load_config(config_path)
    
    # Actualizar arquitectura si se especificó
    if architecture:
        config['model']['architecture'] = architecture
    
    print(f"Arquitectura: {config['model']['architecture']}")
    
    # Crear nuevo formato
    new_checkpoint = {
        'model_state_dict': checkpoint,
        'model_config': config['model'],
        'architecture': config['model']['architecture'],
        'epoch': 0,  # Desconocido para modelos antiguos
        'val_loss': None  # Desconocido para modelos antiguos
    }
    
    # Guardar
    torch.save(new_checkpoint, output_path)
    print(f"✓ Checkpoint convertido guardado en: {output_path}")
    print(f"  Formato: Nuevo (con metadatos)")
    print(f"  Arquitectura: {new_checkpoint['architecture']}")


def main():
    parser = argparse.ArgumentParser(description="Convertir checkpoint antiguo a nuevo formato")
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo antiguo')
    parser.add_argument('--architecture', type=str, help='Arquitectura del modelo (nafnet, unet, convnext, etc.)')
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Configuración')
    parser.add_argument('--output', type=str, help='Ruta de salida (por defecto: sobrescribe el original)')
    
    args = parser.parse_args()
    
    # Si no se especifica output, usar el mismo archivo
    output_path = args.output if args.output else args.model
    
    convert_checkpoint(args.model, args.architecture, args.config, output_path)


if __name__ == '__main__':
    main()
