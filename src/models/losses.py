"""
Funciones de pérdida para Image-to-Image Translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """Pérdida L1 (Mean Absolute Error)"""
    
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM)
    Mide la similitud estructural entre imágenes
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
    
    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)
    
    def _ssim(self, img1, img2):
        """Calcula SSIM entre dos imágenes"""
        # Implementación simplificada de SSIM
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mean1 = F.avg_pool2d(img1, self.window_size, 1)
        mean2 = F.avg_pool2d(img2, self.window_size, 1)
        
        mean1_sq = F.avg_pool2d(img1 ** 2, self.window_size, 1)
        mean2_sq = F.avg_pool2d(img2 ** 2, self.window_size, 1)
        mean1_2 = F.avg_pool2d(img1 * img2, self.window_size, 1)
        
        var1 = mean1_sq - mean1 ** 2
        var2 = mean2_sq - mean2 ** 2
        cov = mean1_2 - mean1 * mean2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        return ssim.mean()


class PSNRLoss(nn.Module):
    """Peak Signal-to-Noise Ratio (PSNR) - Para evaluación"""
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class PerceptualLoss(nn.Module):
    """
    Pérdida perceptual usando características de una red preentrenada
    (VGG16 típicamente)
    """
    
    def __init__(self):
        super().__init__()
        # En una implementación completa, cargaríamos VGG16
        # y extraeríamos características intermedias
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Versión simplificada: usar L1 en las características
        return self.l1(pred, target)


class HybridLoss(nn.Module):
    """
    Combinación de múltiples pérdidas para mejores resultados
    
    Loss = λ1 * L1 + λ2 * SSIM + λ3 * Perceptual
    """
    
    def __init__(
        self,
        lambda_l1: float = 0.6,
        lambda_ssim: float = 0.2,
        lambda_perceptual: float = 0.2
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (
            self.lambda_l1 * l1 +
            self.lambda_ssim * ssim +
            self.lambda_perceptual * perceptual
        )
        
        return {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim,
            'perceptual': perceptual
        }
