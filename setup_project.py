"""
Script para inicializar directorios y crear estructura del proyecto
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directorios a crear
DIRS_TO_CREATE = [
    'data/01_raw',
    'data/02_interim',
    'data/03_processed/input',
    'data/03_processed/ground_truth',
    'models',
    'output_inference',
    'logs',
]

def setup_project():
    """Inicializa la estructura del proyecto"""
    
    logger.info("🚀 Inicializando proyecto CopyAir...")
    
    # Crear directorios
    for dir_path in DIRS_TO_CREATE:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✓ Creado: {dir_path}/")
    
    # Crear archivos .gitkeep para mantener directorios vacíos
    for dir_path in DIRS_TO_CREATE:
        gitkeep = Path(dir_path) / '.gitkeep'
        gitkeep.touch()
    
    logger.info("\n✅ Proyecto configurado exitosamente!")
    logger.info("\nPróximos pasos:")
    logger.info("1. Copia tus imágenes a: data/03_processed/input/ y ground_truth/")
    logger.info("2. Edita configuración: configs/params.yaml")
    logger.info("3. Entrena: python scripts/train.py")
    logger.info("\nPara más información, lee README.md y DEVELOPMENT.md")

if __name__ == '__main__':
    setup_project()
