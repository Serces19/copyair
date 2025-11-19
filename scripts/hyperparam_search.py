"""
Hyperparameter Search Script
Prueba automáticamente diferentes combinaciones de hiperparámetros.

Uso:
    python scripts/hyperparam_search.py --search-type grid --max-trials 10
    python scripts/hyperparam_search.py --search-type random --max-trials 50
"""

import argparse
import yaml
import logging
from pathlib import Path
import itertools
import random
from copy import deepcopy
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import train, setup_device, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define el espacio de búsqueda
SEARCH_SPACE = {
    'model': {
        'architecture': ['unet'],  # Puedes agregar: 'convexnet', 'mambairv2', etc.
        'activation': ['relu', 'gelu', 'silu', 'mish'],
        'base_channels': [32, 64],
    },
    'training': {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [4, 8],
        'optimizer': {
            'type': ['adamw'],
        }
    },
    'loss': {
        'lambda_l1': [0.5, 0.6, 0.7],
        'lambda_ssim': [0.1, 0.2, 0.3],
        'lambda_perceptual': [0.1, 0.15, 0.2],
        'lambda_laplacian': [0.0, 0.05, 0.1],
    }
}


def generate_grid_configs(base_config: dict, search_space: dict, max_trials: int = None):
    """
    Genera todas las combinaciones posibles (Grid Search)
    
    Args:
        base_config: Configuración base
        search_space: Espacio de búsqueda
        max_trials: Máximo número de combinaciones a generar
        
    Returns:
        Lista de configuraciones a probar
    """
    # Extraer todas las combinaciones de parámetros
    param_keys = []
    param_values = []
    
    for section, params in search_space.items():
        for param_name, values in params.items():
            if isinstance(values, dict):
                # Nested params (e.g., optimizer.type)
                for nested_key, nested_values in values.items():
                    param_keys.append((section, param_name, nested_key))
                    param_values.append(nested_values)
            else:
                param_keys.append((section, param_name))
                param_values.append(values)
    
    # Generar todas las combinaciones
    all_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Total de combinaciones posibles: {len(all_combinations)}")
    
    # Limitar si es necesario
    if max_trials and len(all_combinations) > max_trials:
        all_combinations = random.sample(all_combinations, max_trials)
        logger.info(f"Limitado a {max_trials} combinaciones aleatorias")
    
    # Crear configs
    configs = []
    for combination in all_combinations:
        config = deepcopy(base_config)
        
        for i, param_key in enumerate(param_keys):
            value = combination[i]
            
            if len(param_key) == 2:
                section, param = param_key
                config[section][param] = value
            elif len(param_key) == 3:
                section, param, nested = param_key
                if param not in config[section]:
                    config[section][param] = {}
                config[section][param][nested] = value
        
        configs.append(config)
    
    return configs


def generate_random_configs(base_config: dict, search_space: dict, max_trials: int = 50):
    """
    Genera combinaciones aleatorias (Random Search)
    
    Args:
        base_config: Configuración base
        search_space: Espacio de búsqueda
        max_trials: Número de combinaciones a generar
        
    Returns:
        Lista de configuraciones a probar
    """
    configs = []
    
    for _ in range(max_trials):
        config = deepcopy(base_config)
        
        for section, params in search_space.items():
            for param_name, values in params.items():
                if isinstance(values, dict):
                    # Nested params
                    for nested_key, nested_values in values.items():
                        value = random.choice(nested_values)
                        if param_name not in config[section]:
                            config[section][param_name] = {}
                        config[section][param_name][nested_key] = value
                else:
                    value = random.choice(values)
                    config[section][param_name] = value
        
        configs.append(config)
    
    return configs


def run_search(base_config_path: str, search_type: str = 'random', max_trials: int = 10, device_name: str = 'cuda'):
    """
    Ejecuta la búsqueda de hiperparámetros
    
    Args:
        base_config_path: Ruta al archivo de configuración base
        search_type: 'grid' o 'random'
        max_trials: Número máximo de pruebas
        device_name: Dispositivo (cuda/cpu)
    """
    # Cargar config base
    base_config = load_config(base_config_path)
    
    # Generar configuraciones a probar
    if search_type == 'grid':
        configs = generate_grid_configs(base_config, SEARCH_SPACE, max_trials)
    elif search_type == 'random':
        configs = generate_random_configs(base_config, SEARCH_SPACE, max_trials)
    else:
        raise ValueError(f"search_type debe ser 'grid' o 'random', got: {search_type}")
    
    logger.info(f"Iniciando búsqueda de hiperparámetros: {len(configs)} configuraciones")
    logger.info(f"Tipo de búsqueda: {search_type}")
    
    # Setup device
    device = setup_device(device_name)
    
    # Ejecutar cada configuración
    for i, config in enumerate(configs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Experimento {i+1}/{len(configs)}")
        logger.info(f"{'='*80}")
        
        # Log config highlights
        logger.info(f"Architecture: {config['model']['architecture']}")
        logger.info(f"Activation: {config['model']['activation']}")
        logger.info(f"LR: {config['training']['learning_rate']}")
        logger.info(f"Batch Size: {config['training']['batch_size']}")
        logger.info(f"Optimizer: {config['training']['optimizer']['type']}")
        
        try:
            # Entrenar (MLflow trackeará automáticamente)
            train(config, device)
            logger.info(f"✓ Experimento {i+1} completado")
        except Exception as e:
            logger.error(f"✗ Experimento {i+1} falló: {e}")
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info("Búsqueda de hiperparámetros completada!")
    logger.info(f"Revisa los resultados en MLflow UI: mlflow ui")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Búsqueda automática de hiperparámetros")
    parser.add_argument('--config', type=str, default='configs/params.yaml', help='Config base')
    parser.add_argument('--search-type', type=str, default='random', choices=['grid', 'random'], 
                        help='Tipo de búsqueda')
    parser.add_argument('--max-trials', type=int, default=10, 
                        help='Número máximo de experimentos')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_search(
        base_config_path=args.config,
        search_type=args.search_type,
        max_trials=args.max_trials,
        device_name=args.device
    )


if __name__ == '__main__':
    main()
