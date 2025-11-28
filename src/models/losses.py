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
# class PerceptualLoss(nn.Module):
#     """
#     Pérdida perceptual usando LPIPS.
    
#     Input:
#         - Imágenes en rango [-1, 1] (comportamiento por defecto de torchmetrics con normalize=False).
#         - Shape: (N, 3, H, W)
#     """
#     def __init__(self, net='alex', device='cuda'):
#         super().__init__()
#         self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=False).to(device)

#     def forward(self, pred, target):
#         return self.lpips(pred, target)

class PerceptualLoss(nn.Module):
    """
    LPIPS por ventanas deslizantes (tiles),
    con score_mode: ["mean", "sum", "p95"].
    Optimizado para VFX (detalles finos, 8K seguro).
    """

    def __init__(
        self,
        tile_size=128,
        stride=None,
        score_mode="mean",   # 'mean', 'sum', 'p95'
        device="cuda"
    ):
        super().__init__()

        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size // 2
        self.score_mode = score_mode

        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex',
            normalize=False
        ).to(device)

    def forward(self, pred, target):
        N, C, H, W = pred.shape
        T = self.tile_size

        lpips_values = []

        # Sliding window completo
        for y in range(0, H - T + 1, self.stride):
            for x in range(0, W - T + 1, self.stride):

                p = pred[:, :, y:y+T, x:x+T]
                t = target[:, :, y:y+T, x:x+T]

                lp = self.lpips(p, t)     # LPIPS para este tile
                self.lpips.reset()        # IMPORTANTE: Resetear para evitar acumulación de memoria
                lpips_values.append(lp)

        # Convertir a tensor
        lpips_values = torch.stack(lpips_values, dim=0)  # shape: (#tiles,)
        lpips_values = lpips_values.squeeze()

        # -------- SCORE MODES --------
        if self.score_mode == "mean":
            return lpips_values.mean()

        elif self.score_mode == "sum":
            return lpips_values.sum()

        elif self.score_mode == "p95":
            # percentil 95 para capturar errores importantes
            return torch.quantile(lpips_values, 0.95)

        else:
            raise ValueError(f"score_mode inválido: {self.score_mode}")


# --- Pérdida DreamSim ---
class DreamSimLoss(nn.Module):
    """
    Pérdida perceptual usando DreamSim (OpenCLIP-ViT).
    
    Maneja internamente:
    1. Conversión de rango [-1, 1] -> [0, 1]
    2. Resize a 224x224 (requerido por ViT)
    3. Normalización específica de DreamSim (si es necesaria, aunque el modelo suele manejarla)
    """
    def __init__(self, device='cuda'):
        super().__init__()
        if dreamsim is None:
            raise ImportError("La librería 'dreamsim' no está instalada.")
        
        print("Cargando DreamSim (OpenCLIP-ViT)...")
        # dreamsim(pretrained=True) devuelve (model, preprocess)
        # Usamos solo el modelo y hacemos el pre-procesamiento manualmente para tener control
        self.model, _ = dreamsim.dreamsim(pretrained=True, dreamsim_type='open_clip_vitb32')
        self.model = self.model.to(device).eval()
        
        # Normalización estándar de ImageNet (usada por muchos modelos ViT/CLIP)
        # DreamSim suele esperar [0, 1], pero internamente puede normalizar.
        # Para seguridad, nos aseguramos de pasar [0, 1] limpio.

    def forward(self, pred, target):
        """
        Args:
            pred: Tensor (N, 3, H, W) en rango [-1, 1]
            target: Tensor (N, 3, H, W) en rango [-1, 1]
        """
        # 1. Denormalizar [-1, 1] -> [0, 1]
        pred_01 = (pred + 1.0) * 0.5
        target_01 = (target + 1.0) * 0.5
        
        # Clamp para evitar valores fuera de rango por errores numéricos
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        target_01 = torch.clamp(target_01, 0.0, 1.0)
        
        # 2. Resize a 224x224 (Requisito de OpenCLIP-ViT)
        # Usamos interpolación bilineal con antialias
        if pred_01.shape[-1] != 224 or pred_01.shape[-2] != 224:
            pred_224 = F.interpolate(pred_01, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
            target_224 = F.interpolate(target_01, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        else:
            pred_224 = pred_01
            target_224 = target_01

        # 3. Calcular pérdida
        # DreamSim devuelve la distancia.
        loss = self.model(pred_224, target_224)
        
        # Si devuelve un tensor por imagen, promediamos
        if loss.ndim > 0:
            return loss.mean()
        return loss


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
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor

    def forward(self, pred, target):
        # 1. Normalización de seguridad [-1, 1] -> [0, 1]
        pred = (pred + 1) * 0.5
        target = (target + 1) * 0.5

        # 2. Transformada de Fourier (RFFT2)
        # norm='ortho' es importante para conservar la energía unitaria
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')

        # 3. Extraer Magnitud
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        # --- CORRECCIÓN ---
        # Usamos magnitud lineal en lugar de logarítmica para evitar gradientes explosivos
        # y artefactos de color (solarización).
        
        # 4. Calcular diferencia (MSE en espacio de frecuencias lineal)
        diff = (pred_mag - target_mag) ** 2

        # 5. Focal Weighting (Matriz dinámica de pesos)
        # Definimos cuán difícil es cada frecuencia basado en el error relativo
        weight = diff / (diff.mean() + 1e-8) 
        
        # Clampear el peso para estabilidad numérica
        weight = torch.clamp(weight, min=0, max=50) 
        
        weight = weight ** self.alpha

        # 6. Pérdida final
        loss = (diff * weight).mean()
        
        return loss * self.loss_weight


# --- Charbonnier Loss (Robust L1) ---
class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 pseudo-Huber).
    Más robusta que L1 y L2, converge mejor.
    Formula: sqrt((pred - target)^2 + eps^2)
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


# --- Sobel Loss (Edge Loss) ---
class SobelLoss(nn.Module):
    """
    Calcula la pérdida en el dominio de los gradientes (bordes) usando filtros Sobel.
    Ayuda a recuperar detalles de alta frecuencia.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Asegurar que los filtros estén en el mismo device que la entrada
        if pred.device != self.sobel_x.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)

        # Calcular gradientes por canal y sumarlos
        # (N, 3, H, W) -> (N*3, 1, H, W) para conv2d
        b, c, h, w = pred.shape
        pred_flat = pred.view(-1, 1, h, w)
        target_flat = target.view(-1, 1, h, w)

        pred_gx = F.conv2d(pred_flat, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred_flat, self.sobel_y, padding=1)
        
        target_gx = F.conv2d(target_flat, self.sobel_x, padding=1)
        target_gy = F.conv2d(target_flat, self.sobel_y, padding=1)

        # Pérdida L1 sobre los gradientes
        loss_x = self.l1(pred_gx, target_gx)
        loss_y = self.l1(pred_gy, target_gy)
        
        return loss_x + loss_y


class MultiScaleLoss(nn.Module):
    """
    Calcula la pérdida en múltiples escalas (Pirámide).
    Ayuda al modelo a aprender estructura global (bajas frecuencias) y detalles (altas frecuencias).
    """
    def __init__(self, loss_fn, scales=[1.0, 0.5, 0.25], weights=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.scales = scales
        self.weights = weights if weights else [1.0] * len(scales)
        
    def forward(self, pred, target):
        total_loss = 0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1.0:
                p, t = pred, target
            else:
                p = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)
                t = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)
                
            total_loss += self.loss_fn(p, t) * weight
            
        return total_loss


class HybridLoss(nn.Module):
    """
    Combinación de múltiples pérdidas para mejores resultados
    
    Loss = λ1 * L1 + λ2 * SSIM + λ3 * Perceptual + λ4 * Laplacian + ...
    """
    
    def __init__(
        self,
        lambda_l1: float = 0.6,
        lambda_ssim: float = 0.2,
        lambda_perceptual: float = 0.15,
        lambda_laplacian: float = 0.05,
        lambda_ffl: float = 0.0,
        lambda_dreamsim: float = 0.0,
        lambda_charbonnier: float = 0.0,
        lambda_sobel: float = 0.0,
        lambda_multiscale: float = 0.0,
        device: str = 'cuda'
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_laplacian = lambda_laplacian
        self.lambda_ffl = lambda_ffl
        self.lambda_dreamsim = lambda_dreamsim
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_sobel = lambda_sobel
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss(tile_size=256, score_mode="p95", device=device)
        self.laplacian_loss = LaplacianPyramidLoss(device=device)
        
        # Nuevas pérdidas
        if self.lambda_charbonnier > 0:
            self.charbonnier_loss = CharbonnierLoss()
        else:
            self.charbonnier_loss = None
            
        if self.lambda_sobel > 0:
            self.sobel_loss = SobelLoss(device=device)
        else:
            self.sobel_loss = None
        
        # Inicializar FFL si se usa
        if self.lambda_ffl > 0:
            self.ffl_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0).to(device)
        else:
            self.ffl_loss = None
            
        # Inicializar DreamSim si se usa
        if self.lambda_dreamsim > 0:
            try:
                self.dreamsim_loss = DreamSimLoss(device=device)
            except ImportError:
                print("⚠ DreamSim no está instalado. Ignorando pérdida DreamSim.")
                self.dreamsim_loss = None
                self.lambda_dreamsim = 0
        else:
            self.dreamsim_loss = None

        self.lambda_multiscale = lambda_multiscale
        if self.lambda_multiscale > 0:
            # Usamos L1 para la pirámide multi-escala (simple y efectivo para estructura)
            base_loss = L1Loss()
            self.multiscale_loss = MultiScaleLoss(base_loss, scales=[1.0, 0.5, 0.25, 0.125])
        else:
            self.multiscale_loss = None
    
    def forward(self, pred, target, mask=None):
        total_loss = 0.0
        metrics = {}

        # L1 Loss (Masked or Standard)
        if self.lambda_l1 > 0:
            if mask is not None:
                # Masked L1: Enfocarse en áreas de interés
                l1_map = torch.abs(pred - target)
                masked_l1 = l1_map * mask
                # Normalizar por la suma de la máscara para mantener la escala
                l1 = masked_l1.sum() / (mask.sum() + 1e-6)
            else:
                l1 = self.l1_loss(pred, target)
            total_loss += self.lambda_l1 * l1
            metrics['l1'] = l1
            
        if self.lambda_ssim > 0:
            ssim = self.ssim_loss(pred, target)
            total_loss += self.lambda_ssim * ssim
            metrics['ssim'] = ssim

        if self.lambda_perceptual > 0:
            perceptual = self.perceptual_loss(pred, target)
            total_loss += self.lambda_perceptual * perceptual
            metrics['perceptual'] = perceptual

        if self.lambda_laplacian > 0:
            laplacian = self.laplacian_loss(pred, target)
            total_loss += self.lambda_laplacian * laplacian
            metrics['laplacian'] = laplacian
            
        # Charbonnier
        if self.lambda_charbonnier > 0 and self.charbonnier_loss is not None:
            charb = self.charbonnier_loss(pred, target)
            total_loss += self.lambda_charbonnier * charb
            metrics['charbonnier'] = charb
            
        # Sobel
        if self.lambda_sobel > 0 and self.sobel_loss is not None:
            sobel = self.sobel_loss(pred, target)
            total_loss += self.lambda_sobel * sobel
            metrics['sobel'] = sobel
        
        # Agregar FFL si está habilitado
        if self.lambda_ffl > 0 and self.ffl_loss is not None:
            ffl = self.ffl_loss(pred, target)
            total_loss += self.lambda_ffl * ffl
            metrics['ffl'] = ffl
            
        # Agregar DreamSim si está habilitado
        if self.lambda_dreamsim > 0 and self.dreamsim_loss is not None:
            dream = self.dreamsim_loss(pred, target)
            total_loss += self.lambda_dreamsim * dream
            metrics['dreamsim'] = dream
            
        if self.lambda_multiscale > 0 and self.multiscale_loss is not None:
            ms_loss = self.multiscale_loss(pred, target)
            total_loss += self.lambda_multiscale * ms_loss
            metrics['multiscale'] = ms_loss
            
        metrics['total'] = total_loss
            
        return metrics
