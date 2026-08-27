"""
Funciones de pérdida optimizadas para Image-to-Image Translation y VFX (CopyAir)
Pérdidas Core:
- CharbonnierLoss: L1 robusto y suavizado (fidelidad píxel a píxel sin desenfoque).
- PerceptualLoss: LPIPS perceptual (VGG/AlexNet para calidad visual humana).
- DinoLoss: DINOv2 Vision Transformer (fidelidad semántica y estructura anatómica 100% GPU diferenciable).
- LaplacianPyramidLoss: Pirámide Laplaciana (nitidez de micro-texturas y frecuencias altas).
- SSIMLoss: Similitud estructural.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import AutoModel


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 pseudo-Huber): sqrt((pred - target)^2 + eps^2)
    Más suave y numéricamente estable que L1 y L2, preserva bordes sin desenfoque.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    Pérdida perceptual LPIPS (Learned Perceptual Image Patch Similarity).
    Entrada: tensores en rango [-1, 1], shape (B, 3, H, W).
    """
    def __init__(self, net_type: str = 'alex', device: str = 'cuda'):
        super().__init__()
        target_device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=False).to(target_device)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Asegurar tensor en device correcto
        pred_clamped = torch.clamp(pred, -1.0, 1.0)
        target_clamped = torch.clamp(target, -1.0, 1.0)
        distance = self.lpips(pred_clamped, target_clamped)
        self.lpips.reset()
        return distance


class DinoLoss(nn.Module):
    """
    DINOv2 Perceptual Loss (Meta AI Vision Transformer).
    100% Diferenciable en GPU, nativo en PyTorch sin llamadas CPU/NumPy.
    Captura anatomía profunda, geometría 3D y coherencia semántica facial.
    """
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda", use_fp16: bool = True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')

        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        except Exception:
            # Fallback a dinov2-base si small falla
            self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

        if self.use_fp16:
            self.model = self.model.half()

        for p in self.model.parameters():
            p.requires_grad = False

        # Buffers de normalización ImageNet (Diferenciables en GPU)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> [0, 1] -> ImageNet Norm -> 224x224 (Diferenciable)"""
        x_01 = (x + 1.0) * 0.5
        x_01 = torch.clamp(x_01, 0.0, 1.0)
        x_norm = (x_01 - self.mean.to(x.device)) / self.std.to(x.device)
        # Interpolación antialiased a 224x224 (múltiplo del patch 14x14 de DINOv2)
        x_224 = F.interpolate(x_norm, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        if self.use_fp16:
            x_224 = x_224.half()
        return x_224

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_in = self._preprocess_gpu(pred)
        target_in = self._preprocess_gpu(target)

        with torch.no_grad():
            target_out = self.model(pixel_values=target_in)
            target_tokens = F.normalize(target_out.last_hidden_state, dim=-1)

        pred_out = self.model(pixel_values=pred_in)
        pred_tokens = F.normalize(pred_out.last_hidden_state, dim=-1)

        # L1 sobre los tokens normalizados de DINOv2
        return F.l1_loss(pred_tokens, target_tokens)


class LaplacianPyramidLoss(nn.Module):
    """
    Pirámide Laplaciana para restaurar micro-texturas y altas frecuencias.
    """
    def __init__(self, max_levels: int = 4, device: str = 'cuda'):
        super().__init__()
        self.max_levels = max_levels
        self.l1 = nn.L1Loss()
        
        ax = torch.arange(-2, 3, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * 1.0**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def _laplacian_pyramid(self, img: torch.Tensor):
        pyramid = []
        curr = img
        k = self.kernel.to(img.device)
        for _ in range(self.max_levels):
            blurred = F.conv2d(curr, k, padding=2, groups=3)
            lap = curr - blurred
            pyramid.append(lap)
            curr = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        pyramid.append(curr)
        return pyramid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_pyr = self._laplacian_pyramid(pred)
        target_pyr = self._laplacian_pyramid(target)
        loss = sum(self.l1(p, t) for p, t in zip(pred_pyr, target_pyr))
        return loss / (self.max_levels + 1)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM)"""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_01 = (pred + 1.0) * 0.5
        t_01 = (target + 1.0) * 0.5
        
        mu1 = F.avg_pool2d(p_01, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(t_01, self.window_size, stride=1, padding=self.window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(p_01 * p_01, self.window_size, stride=1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(t_01 * t_01, self.window_size, stride=1, padding=self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(p_01 * t_01, self.window_size, stride=1, padding=self.window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / \
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2) + 1e-8)
        return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    """
    Pérdida Híbrida Limpia y SOTA para CopyAir:
    Total = lambda_charbonnier * L_Charb + lambda_perceptual * L_LPIPS + lambda_dino * L_DINO + lambda_laplacian * L_Lap + lambda_ssim * L_SSIM
    """
    def __init__(
        self,
        lambda_charbonnier: float = 0.2,
        lambda_perceptual: float = 0.8,
        lambda_dino: float = 0.2,
        lambda_laplacian: float = 0.1,
        lambda_ssim: float = 0.0,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__()
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_perceptual = lambda_perceptual
        self.lambda_dino = lambda_dino
        self.lambda_laplacian = lambda_laplacian
        self.lambda_ssim = lambda_ssim

        self.charbonnier_loss = CharbonnierLoss()
        self.perceptual_loss = PerceptualLoss(net_type='alex', device=device) if lambda_perceptual > 0 else None
        self.dino_loss = DinoLoss(device=device) if lambda_dino > 0 else None
        self.laplacian_loss = LaplacianPyramidLoss(device=device) if lambda_laplacian > 0 else None
        self.ssim_loss = SSIMLoss() if lambda_ssim > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.lambda_charbonnier > 0:
            l_charb = self.charbonnier_loss(pred, target, mask=mask)
            losses['charbonnier'] = l_charb
            total = total + self.lambda_charbonnier * l_charb

        if self.perceptual_loss is not None and self.lambda_perceptual > 0:
            l_perc = self.perceptual_loss(pred, target)
            losses['perceptual'] = l_perc
            total = total + self.lambda_perceptual * l_perc

        if self.dino_loss is not None and self.lambda_dino > 0:
            l_dino = self.dino_loss(pred, target)
            losses['dino'] = l_dino
            total = total + self.lambda_dino * l_dino

        if self.laplacian_loss is not None and self.lambda_laplacian > 0:
            l_lap = self.laplacian_loss(pred, target)
            losses['laplacian'] = l_lap
            total = total + self.lambda_laplacian * l_lap

        if self.ssim_loss is not None and self.lambda_ssim > 0:
            l_ssim = self.ssim_loss(pred, target)
            losses['ssim'] = l_ssim
            total = total + self.lambda_ssim * l_ssim

        losses['total'] = total
        return losses
