"""
Funciones de pérdida para Image-to-Image Translation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
try:
    import dreamsim
except ImportError:
    dreamsim = None


class L1Loss(nn.Module):
    """Pérdida L1 (Mean Absolute Error)"""
    
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


# --- Pérdida perceptual con LPIPS ---
class PerceptualLoss(nn.Module):
    def __init__(self, net='vgg', device='cuda'):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)

    def forward(self, pred, target):
        return self.lpips(pred, target)


# --- Pérdida Laplaciana para bordes ---
class LaplacianPyramidLoss(nn.Module):
    """Calcula la pérdida de la pirámide laplaciana para preservar bordes nítidos."""
    def __init__(self, max_levels=5, channels=3, device='cuda'):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.loss_fn = nn.L1Loss()
        kernel = self._build_gaussian_kernel(channels=channels, device=device)
        self.gaussian_conv = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False, groups=channels)
        self.gaussian_conv.weight.data = kernel
        self.gaussian_conv.weight.requires_grad = False

    def _build_gaussian_kernel(self, channels, device):
        ax = torch.arange(-2, 3, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * 1.0**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel.repeat(channels, 1, 1, 1).to(device)

    def laplacian_pyramid(self, img):
        pyramid = []
        current_img = img
        for _ in range(self.max_levels):
            blurred = self.gaussian_conv(current_img)
            laplacian = current_img - blurred
            pyramid.append(laplacian)
            current_img = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        pyramid.append(current_img)
        return pyramid

    def forward(self, prediction, target):
        # Nuestros tensores están en [-1, 1], las pérdidas funcionan correctamente en este rango
        pred_pyramid = self.laplacian_pyramid(prediction)
        target_pyramid = self.laplacian_pyramid(target)
        loss = 0
        for pred_level, target_level in zip(pred_pyramid, target_pyramid):
            loss += self.loss_fn(pred_level, target_level)
        return loss / self.max_levels


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
    
    def __init__(self, max_val: float = 2.0):  # Cambiado a 2.0 para rango [-1, 1]
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class FocalFrequencyLoss(nn.Module):
    """
    Convierte imágenes a frecuencia y compara sus espectros.
    Ideal para recuperar texturas finas y poros.
    """
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha # Factor de enfoque (cuánto castigar las frec. difíciles)

    def forward(self, pred, target):
        # 1. Transformada de Fourier 2D (FFT)
        # RFFT2 es para entradas reales (imágenes)
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')

        # 2. Extraer Magnitud (Amplitud del espectro)
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        # 3. Calcular la diferencia logarítmica (mejor estabilidad numérica)
        # Se suma un epsilon pequeño para evitar log(0)
        diff = torch.abs(pred_mag - target_mag) ** 2
        
        # 4. Focal Weighting (Matriz dinámica de pesos)
        # Las frecuencias donde el error es grande reciben más peso
        weight = diff / (diff.mean() + 1e-8) # Normalización
        weight = weight ** self.alpha # Enfoque

        # 5. Pérdida final ponderada
        loss = (diff * weight).mean()
        return loss * self.loss_weight


class HybridLoss(nn.Module):
    """
    Combinación de múltiples pérdidas para mejores resultados
    
    Loss = λ1 * L1 + λ2 * SSIM + λ3 * Perceptual + λ4 * Laplacian
    """
    
    def __init__(
        self,
        lambda_l1: float = 0.6,
        lambda_ssim: float = 0.2,
        lambda_perceptual: float = 0.15,
        lambda_laplacian: float = 0.05,
        lambda_ffl: float = 0.0,
        lambda_dreamsim: float = 0.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_laplacian = lambda_laplacian
        self.lambda_ffl = lambda_ffl
        self.lambda_dreamsim = lambda_dreamsim
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss(device=device)
        self.laplacian_loss = LaplacianPyramidLoss(device=device)
        
        # Inicializar FFL si se usa
        if self.lambda_ffl > 0:
            self.ffl_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0).to(device)
        else:
            self.ffl_loss = None
            
        # Inicializar DreamSim si se usa
        if self.lambda_dreamsim > 0:
            if dreamsim is None:
                print("⚠ DreamSim no está instalado. Ignorando pérdida DreamSim.")
                self.dreamsim_loss = None
                self.lambda_dreamsim = 0
            else:
                print("Cargando DreamSim (OpenCLIP-ViT)...")
                # La API correcta es dreamsim(pretrained=True)
                self.dreamsim_loss, _ = dreamsim.dreamsim(pretrained=True, dreamsim_type='open_clip_vitb32')
                self.dreamsim_loss = self.dreamsim_loss.to(device).eval()
        else:
            self.dreamsim_loss = None
    
    def forward(self, pred, target, mask=None):
        # L1 Loss (Masked or Standard)
        if mask is not None:
            # Masked L1: Enfocarse en áreas de interés
            l1_map = torch.abs(pred - target)
            masked_l1 = l1_map * mask
            # Normalizar por la suma de la máscara para mantener la escala
            l1 = masked_l1.sum() / (mask.sum() + 1e-6)
        else:
            l1 = self.l1_loss(pred, target)
            
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        laplacian = self.laplacian_loss(pred, target)
        
        total_loss = (
            self.lambda_l1 * l1 +
            self.lambda_ssim * ssim +
            self.lambda_perceptual * perceptual +
            self.lambda_laplacian * laplacian
        )
        
        metrics = {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim,
            'perceptual': perceptual,
            'laplacian': laplacian
        }
        
        # Agregar FFL si está habilitado
        if self.ffl_loss is not None:
            ffl = self.ffl_loss(pred, target)
            total_loss += self.lambda_ffl * ffl
            metrics['ffl'] = ffl
            
        # Agregar DreamSim si está habilitado
        if self.dreamsim_loss is not None:
            # DreamSim espera RGB estándar, nuestros tensores están en [-1, 1]
            # DreamSim maneja normalización interna, pero aseguramos float
            dream = self.dreamsim_loss(pred, target)
            if dream.ndim > 0:
                dream = dream.mean()
            total_loss += self.lambda_dreamsim * dream
            metrics['dreamsim'] = dream
            metrics['total'] = total_loss # Actualizar total
            
        return metrics
