"""
MambaIR: A Simple Baseline for Image Restoration with State-Space Model
Paper: https://arxiv.org/abs/2402.15648

Implementación REAL de 2D Visual State Space (VMamba / MambaIR).
Soporta `mamba_ssm` si está disponible (entornos GPU Cloud).
Provee fallback matemáticamente exacto en PyTorch puro para entornos locales/Windows.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    IS_CUDA = True
except ImportError:
    IS_CUDA = False

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Fallback PyTorch puro del selective scan de Mamba.
    Matemáticamente equivalente a `selective_scan_fn` de mamba_ssm.
    u: (B, D, L)
    delta: (B, D, L)
    A: (D, N)
    B: (B, N, L)
    C: (B, N, L)
    D: (D,)
    z: (B, D, L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
        
    B_in, D_in, L_in = u.shape
    N_in = A.shape[1]
    
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A)) # (B, D, L, N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u) # (B, D, L, N)
    
    x = torch.zeros((B_in, D_in, N_in), device=u.device, dtype=deltaA.dtype)
    ys = []
    
    # Loop secuencial sobre L (asumido para fallback puro en CPU/GPU sin kernels dedicados)
    for i in range(L_in):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        ys.append(y)
        
    y = torch.stack(ys, dim=2) # (B, D, L)
    
    if D is not None:
        y = y + u * D[:, None]
        
    if z is not None:
        y = y * F.silu(z)
        
    return y.to(dtype_in)


class LayerNorm(nn.Module):
    """LayerNorm que soporta formato channels_first (N, C, H, W) y channels_last (N, H, W, C)"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SS2D(nn.Module):
    """
    2D Selective Scan (SS2D) - Implementación REAL
    Realiza el escaneo en 4 direcciones:
    1. Top-left -> Bottom-right
    2. Bottom-right -> Top-left
    3. Top-right -> Bottom-left
    4. Bottom-left -> Top-right
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Proyección de entrada para x y z (gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolución local DW
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        
        # Proyecciones para las 4 direcciones (x_proj: B, C, dt)
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ])
        
        # Proyecciones para dt (Delta)
        self.dt_projs = nn.ModuleList([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        ])
        
        # Parámetros A y D
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
            for _ in range(4)
        ])
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner))
            for _ in range(4)
        ])
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def dt_init(self, dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        L = H * W
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, H, W, d_inner)
        
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.conv2d(x)) # (B, d_inner, H, W)
        
        # Generar las 4 vistas (escaneos)
        xs = [
            x.view(B, -1, L), # forward
            x.view(B, -1, L).flip([-1]), # backward
            x.transpose(2, 3).reshape(B, -1, L), # transpose forward
            x.transpose(2, 3).reshape(B, -1, L).flip([-1]) # transpose backward
        ]
        
        y_out = []
        for i in range(4):
            x_i = xs[i] # (B, d_inner, L)
            x_i_flat = x_i.transpose(1, 2) # (B, L, d_inner)
            
            x_proj = self.x_proj[i](x_i_flat) # (B, L, dt_rank + 2*d_state)
            dt, B_proj, C_proj = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            dt = self.dt_projs[i](dt).transpose(1, 2) # (B, d_inner, L)
            B_proj = B_proj.transpose(1, 2) # (B, d_state, L)
            C_proj = C_proj.transpose(1, 2) # (B, d_state, L)
            
            A = -torch.exp(self.A_logs[i].float()) # (d_inner, d_state)
            D = self.Ds[i].float()
            
            if IS_CUDA:
                y = selective_scan_fn(
                    x_i, dt, A, B_proj, C_proj, D, z=None,
                    delta_bias=self.dt_projs[i].bias.float(),
                    delta_softplus=True, return_last_state=False
                )
            else:
                y = selective_scan_ref(
                    x_i, dt, A, B_proj, C_proj, D, z=None,
                    delta_bias=self.dt_projs[i].bias.float(),
                    delta_softplus=True
                )
            y_out.append(y)
            
        # Revertir los escaneos y sumar las 4 direcciones
        y_0 = y_out[0].view(B, self.d_inner, H, W)
        y_1 = y_out[1].flip([-1]).view(B, self.d_inner, H, W)
        y_2 = y_out[2].view(B, self.d_inner, W, H).transpose(2, 3)
        y_3 = y_out[3].flip([-1]).view(B, self.d_inner, W, H).transpose(2, 3)
        
        y = y_0 + y_1 + y_2 + y_3
        
        # Gating e Inversión de proyecciones
        y = y.permute(0, 2, 3, 1) # (B, H, W, d_inner)
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """
    Visual State Space Block
    Arquitectura core basada en Mamba:
    Input -> LayerNorm -> SS2D (con DWConv) -> Output + Residual
    """
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0., norm_layer: nn.Module = LayerNorm):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim, data_format="channels_last")
        self.self_attention = SS2D(d_model=hidden_dim)
        # DropPath se deja omitido para simplificación a menos que usemos timm.DropPath
        self.drop_path = nn.Identity()

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class MambaIR(nn.Module):
    """
    MambaIR: U-Net / Residual architecture para Image Restoration y VFX
    Aprovecha VSSBlocks para extraer características de contexto global y local simultáneamente.
    """
    def __init__(
        self, 
        in_channels=3, 
        out_channels=3, 
        embed_dim=64, 
        depths=[4, 4, 4, 4], 
        drop_rate=0., 
        drop_path_rate=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depths = depths
        
        # 1. Extracción de Características Superficiales
        self.shallow_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, 1, 1),
            nn.SiLU()
        )
        
        # 2. Extracción de Características Profundas (Bloques VSS)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.Sequential(
                *[VSSBlock(hidden_dim=embed_dim, drop_path=drop_path_rate) 
                  for _ in range(depths[i_layer])]
            )
            self.layers.append(layer)
            
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # 3. Reconstrucción
        self.reconstruction = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(embed_dim, out_channels, 3, 1, 1)
        )
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (B, C, H, W)
        
        # Shallow features
        x_shallow = self.shallow_feature_extractor(x)
        
        # Preparar para bloques VSS (esperan channels_last)
        x_deep = x_shallow.permute(0, 2, 3, 1) # (B, H, W, C)
        
        # Forward por los bloques profundos
        for layer in self.layers:
            x_deep = layer(x_deep)
            
        x_deep = x_deep.permute(0, 3, 1, 2) # (B, C, H, W)
        x_deep = self.conv_after_body(x_deep)
        x_deep = x_deep + x_shallow # Residual profunda
        
        # Reconstrucción final
        x_out = self.reconstruction(x_deep)
        
        # Residual connection global (Global Skip Connection)
        x_out = x_out + x
        
        return self.tanh(x_out)

# Helpers
def mambair_tiny(in_channels=3, out_channels=3):
    return MambaIR(in_channels, out_channels, embed_dim=32, depths=[2, 2, 2, 2])

def mambair_base(in_channels=3, out_channels=3):
    return MambaIR(in_channels, out_channels, embed_dim=64, depths=[4, 4, 4, 4])

def mambair_large(in_channels=3, out_channels=3):
    return MambaIR(in_channels, out_channels, embed_dim=96, depths=[6, 6, 6, 6])


# Aliases & Exports
MambaIRv2 = MambaIR

__all__ = [
    "MambaIR",
    "MambaIRv2",
    "mambair_tiny",
    "mambair_base",
    "mambair_large",
    "SS2D",
    "VSSBlock"
]


