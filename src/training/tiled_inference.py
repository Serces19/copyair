import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def get_gaussian_weight_map(tile_size: int, device: torch.device) -> torch.Tensor:
    """
    Genera un mapa de pesos Gaussiano 2D para suavizar los bordes de los tiles.
    El centro tiene peso 1.0 y los bordes caen suavemente hacia 0.
    """
    # Crear vector 1D Gaussiano
    sigma = tile_size / 4.0 # Sigma controla la caída. tile_size/4 asegura caída suave en los bordes.
    x = torch.arange(tile_size, device=device).float()
    center = tile_size / 2.0 - 0.5 # Centro ajustado
    
    gauss_1d = torch.exp(-((x - center)**2) / (2 * sigma**2))
    
    # Crear mapa 2D (producto exterior)
    gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    
    # Normalizar para que el máximo sea 1.0
    gauss_2d = gauss_2d / gauss_2d.max()
    
    # Añadir dimensiones de canal y batch: (1, 1, H, W)
    return gauss_2d.unsqueeze(0).unsqueeze(0)

def tiled_inference(
    model: nn.Module,
    img_tensor: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Realiza inferencia usando tiles solapados y mezcla Gaussiana.
    
    Args:
        model: Modelo entrenado.
        img_tensor: Tensor de entrada (1, C, H, W) normalizado a [-1, 1].
        tile_size: Tamaño del tile (debe coincidir con entrenamiento o ser potencia de 2).
        overlap: Tamaño del solapamiento entre tiles.
        device: Dispositivo.
        
    Returns:
        Tensor de salida reconstruido (1, C, H, W) en rango [-1, 1].
    """
    B, C, H, W = img_tensor.shape
    
    # Stride (paso) = tamaño - overlap
    stride = tile_size - overlap
    
    # Calcular padding necesario para cubrir toda la imagen
    # Queremos que el último tile cubra el borde derecho/inferior
    pad_h = (tile_size - (H % stride)) % stride
    pad_w = (tile_size - (W % stride)) % stride
    
    # Añadir padding extra si la imagen es menor que el tile
    if H < tile_size: pad_h += tile_size - H
    if W < tile_size: pad_w += tile_size - W
    
    # Aplicar padding (reflect para continuidad visual)
    # F.pad order: (left, right, top, bottom)
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    H_pad, W_pad = img_padded.shape[2], img_padded.shape[3]
    
    # Tensores para acumular resultados y pesos
    output_sum = torch.zeros_like(img_padded)
    weight_sum = torch.zeros_like(img_padded)
    
    # Mapa de pesos Gaussiano
    weight_map = get_gaussian_weight_map(tile_size, device)
    # Expandir a canales: (1, C, H, W)
    weight_map = weight_map.repeat(1, C, 1, 1)
    
    model.eval()
    
    with torch.no_grad():
        # Iterar sobre tiles
        for y in range(0, H_pad - tile_size + 1, stride):
            for x in range(0, W_pad - tile_size + 1, stride):
                
                # Extraer tile
                tile = img_padded[:, :, y:y+tile_size, x:x+tile_size]
                tile = tile.to(device)
                
                # Inferencia
                pred_tile = model(tile)
                
                # Acumular predicción ponderada
                output_sum[:, :, y:y+tile_size, x:x+tile_size] += pred_tile * weight_map
                weight_sum[:, :, y:y+tile_size, x:x+tile_size] += weight_map
                
    # Normalizar dividiendo por la suma de pesos
    # Evitar división por cero (aunque con overlap suficiente no debería pasar)
    output = output_sum / (weight_sum + 1e-8)
    
    # Recortar al tamaño original
    output = output[:, :, :H, :W]
    
    return output
