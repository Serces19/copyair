import signal
import logging
import yaml
import torch
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        logger.info("\n🛑 Señal de interrupción recibida. Terminando después de esta época...")

def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(device_name: str) -> torch.device:
    """Configura el dispositivo (CPU/GPU)"""
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Usando CPU")
    return device

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormaliza tensor de [-1, 1] a [0, 1]"""
    return torch.clamp((tensor + 1.0) / 2.0, 0, 1)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convierte tensor (C, H, W) a numpy (H, W, C) para visualización"""
    img = denormalize(tensor.cpu()).numpy()
    img = np.transpose(img, (1, 2, 0))
    return img
