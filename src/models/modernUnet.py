# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ------------------------------------------------------------------------------
# # 1. Componentes "State-of-the-Art" (Building Blocks)
# # ------------------------------------------------------------------------------

# class LayerNorm2d(nn.Module):
#     """
#     LayerNorm optimizado para formato (B, C, H, W).
#     Usado en ConvNeXt, Restormer, NAFNet. 
#     Más estable que BatchNorm para restauración.
#     """
#     def __init__(self, channels, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(channels))
#         self.bias = nn.Parameter(torch.zeros(channels))
#         self.eps = eps

#     def forward(self, x):
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x

# class SimpleGate(nn.Module):
#     """
#     Mecanismo de atención simplificado (NAFNet).
#     Divide los canales en 2 y los multiplica.
#     Reemplaza a ReLU/GELU y aprende relaciones no lineales complejas.
#     """
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         return x1 * x2

# class GatedBlock(nn.Module):
#     """
#     Bloque principal inspirado en NAFNet y Restormer.
#     Combina la eficiencia de CNN con el 'Gating' de Transformers.
#     """
#     def __init__(self, c, ffn_expansion_factor=2):
#         super().__init__()
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)

#         # 1. Spatial Mixing (Contexto local)
#         self.conv1 = nn.Conv2d(c, c, kernel_size=1)
#         self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c) # Depthwise
        
#         # 2. Channel Mixing (Gating)
#         # Expansion factor para el Gate
#         hidden_dim = int(c * ffn_expansion_factor)
#         self.conv2 = nn.Conv2d(c, hidden_dim * 2, kernel_size=1) # *2 para el split
#         self.sg = SimpleGate()
#         self.conv3 = nn.Conv2d(hidden_dim, c, kernel_size=1)

#         # SCA (Simplified Channel Attention) - Opcional pero recomendada
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, 1),
#         )
        
#         # Escala aprendible para estabilidad (Layer Scale)
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

#     def forward(self, x):
#         input = x
        
#         # Parte 1: Spatial & Gating
#         x = self.norm1(x)
#         x = self.conv1(x)
#         x = self.dwconv(x)
#         x = self.conv2(x)
#         x = self.sg(x) # Aquí ocurre la magia no lineal
#         x = self.conv3(x)
        
#         # Parte 2: Channel Attention simple
#         x = x * self.sca(x)
        
#         # Residual Connection con escala
#         return input + x * self.beta

# # class Downsample(nn.Module):
# #     """Bajada de resolución sin perder información (PixelUnshuffle invertido o Strided Conv)"""
# #     def __init__(self, in_c, out_c):
# #         super().__init__()
# #         self.conv = nn.Conv2d(in_c, out_c, kernel_size=2, stride=2)
# #     def forward(self, x):
# #         return self.conv(x)


# class Downsample(nn.Module):
#     """
#     Bajada de resolución SIN PÉRDIDA de información.
#     Usa PixelUnshuffle: (B, C, H, W) -> (B, 4C, H/2, W/2)
#     Luego reduce canales con conv 1x1.
#     """
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         # PixelUnshuffle aumenta canales x4, así que la conv reduce de in_c*4 a out_c
#         self.body = nn.Sequential(
#             nn.PixelUnshuffle(2),
#             nn.Conv2d(in_c * 4, out_c, kernel_size=1, bias=False) 
#         )

#     def forward(self, x):
#         return self.body(x)

# class UpsampleICNR(nn.Module):
#     """Subida de resolución libre de ajedrez (PixelShuffle + ICNR)"""
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=1)
#         self.ps = nn.PixelShuffle(2)
#         self._icnr_init()

#     def _icnr_init(self):
#         # Inicialización ICNR para evitar checkerboard artifacts
#         weight = nn.init.kaiming_normal_(
#             torch.zeros_like(self.conv.weight[:self.conv.out_channels // 4])
#         )
#         weight = weight.transpose(0, 1).repeat(1, 4, 1, 1).transpose(0, 1)
#         self.conv.weight.data.copy_(weight)
#         if self.conv.bias is not None: self.conv.bias.data.fill_(0)

#     def forward(self, x):
#         return self.ps(self.conv(x))

# # ------------------------------------------------------------------------------
# # 2. La Red Principal (ModernUNet)
# # ------------------------------------------------------------------------------

# class ModernUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_dim=32, num_blocks=[2, 2, 4, 6]):
#         super().__init__()
        
#         self.base_dim = base_dim
        
#         # --- Input ---
#         self.intro = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

#         # --- Encoder ---
#         # Level 1
#         self.enc1 = nn.Sequential(*[GatedBlock(base_dim) for _ in range(num_blocks[0])])
#         self.down1 = Downsample(base_dim, base_dim * 2)
        
#         # Level 2
#         self.enc2 = nn.Sequential(*[GatedBlock(base_dim * 2) for _ in range(num_blocks[1])])
#         self.down2 = Downsample(base_dim * 2, base_dim * 4)
        
#         # Level 3
#         self.enc3 = nn.Sequential(*[GatedBlock(base_dim * 4) for _ in range(num_blocks[2])])
#         self.down3 = Downsample(base_dim * 4, base_dim * 8)
        
#         # --- Bottleneck ---
#         # Aquí es donde aprende el contexto global más abstracto
#         self.bottleneck = nn.Sequential(*[GatedBlock(base_dim * 8) for _ in range(num_blocks[3])])
        
#         # --- Decoder ---
#         # Level 3
#         self.up3 = UpsampleICNR(base_dim * 8, base_dim * 4)
#         self.reduce3 = nn.Conv2d(base_dim * 8, base_dim * 4, 1) # Fusion channels
#         self.dec3 = nn.Sequential(*[GatedBlock(base_dim * 4) for _ in range(num_blocks[2])])
        
#         # Level 2
#         self.up2 = UpsampleICNR(base_dim * 4, base_dim * 2)
#         self.reduce2 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)
#         self.dec2 = nn.Sequential(*[GatedBlock(base_dim * 2) for _ in range(num_blocks[1])])
        
#         # Level 1
#         self.up1 = UpsampleICNR(base_dim * 2, base_dim)
#         self.reduce1 = nn.Conv2d(base_dim * 2, base_dim, 1)
#         self.dec1 = nn.Sequential(*[GatedBlock(base_dim) for _ in range(num_blocks[0])])
        
#         # --- Output ---
#         self.final = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)
#         self.tanh = nn.Tanh()

#     def _pad_to_multiple(self, x, multiple=8):
#         h, w = x.shape[2], x.shape[3]
#         ph = (multiple - h % multiple) % multiple
#         pw = (multiple - w % multiple) % multiple
#         return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w

#     def forward(self, x):
#         # Manejo automático de tamaños arbitrarios
#         x, h_orig, w_orig = self._pad_to_multiple(x, multiple=8)
        
#         # Intro
#         x1 = self.intro(x) # (B, C, H, W)
        
#         # Encoder
#         x1 = self.enc1(x1)
#         x2 = self.down1(x1) # (B, 2C, H/2, W/2)
#         x2 = self.enc2(x2)
#         x3 = self.down2(x2) # (B, 4C, H/4, W/4)
#         x3 = self.enc3(x3)
#         x4 = self.down3(x3) # (B, 8C, H/8, W/8)
        
#         # Bottleneck
#         x4 = self.bottleneck(x4)
        
#         # Decoder con Skip Connections inteligentes
#         # Up 3
#         up3 = self.up3(x4)
#         cat3 = torch.cat([up3, x3], dim=1) # Concatenación
#         out3 = self.dec3(self.reduce3(cat3))
        
#         # Up 2
#         up2 = self.up2(out3)
#         cat2 = torch.cat([up2, x2], dim=1)
#         out2 = self.dec2(self.reduce2(cat2))
        
#         # Up 1
#         up1 = self.up1(out2)
#         cat1 = torch.cat([up1, x1], dim=1)
#         out1 = self.dec1(self.reduce1(cat1))
        
#         # Salida
#         out = self.final(out1)
#         out = self.tanh(out)
        
#         # Recortar padding si fue necesario
#         return out[:, :, :h_orig, :w_orig]




import torch
import torch.nn as nn
import torch.nn.functional as F

# --- COMPONENTES MODIFICADOS (AGRESIVOS) ---

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class StrictGatedBlock(nn.Module):
    """
    Versión 'Hardcore' del GatedBlock.
    - Sin LayerScale (beta).
    - Sin Residual Connection (input + x) -> Obliga a aprender la transformación completa.
    - Usa GroupNorm en lugar de LayerNorm (Mejor para batch pequeño/few-shot).
    """
    def __init__(self, c, ffn_expansion_factor=2, use_residual=False): # <--- Flag importante
        super().__init__()
        self.use_residual = use_residual
        
        # Volvemos a GroupNorm (lo que funcionó en tu Vanilla UNet)
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=c)
        self.norm2 = nn.GroupNorm(num_groups=16, num_channels=c)

        # Spatial Mixing
        self.conv1 = nn.Conv2d(c, c, kernel_size=1)
        self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c) 
        
        # Channel Mixing (Gating)
        hidden_dim = int(c * ffn_expansion_factor)
        self.conv2 = nn.Conv2d(c, hidden_dim * 2, kernel_size=1)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(hidden_dim, c, kernel_size=1)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid() # Sigmoid es vital aquí
        )

    def forward(self, x):
        input = x
        
        # Proceso
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = x * self.sca(x)
        
        # DECISIÓN CRÍTICA:
        # Si use_residual es False, la red NO puede copiar la entrada.
        # Tiene que generar la salida desde cero.
        if self.use_residual:
            return input + x
        else:
            return x # <-- Aquí forzamos el aprendizaje estructural

# --- ARQUITECTURA PRINCIPAL ---

class ModernUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        # 1. ENCODER (Sin residuos = Aprendizaje forzado)
        self.inc = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Usamos use_residual=False en el Encoder para forzar abstracción
        self.block1 = StrictGatedBlock(base_channels, use_residual=False)
        
        self.down1 = nn.MaxPool2d(2) # MaxPool (Destructivo, bueno para limpiar)
        self.expand1 = nn.Conv2d(base_channels, base_channels*2, 1)
        self.block2 = StrictGatedBlock(base_channels*2, use_residual=False)
        
        self.down2 = nn.MaxPool2d(2)
        self.expand2 = nn.Conv2d(base_channels*2, base_channels*4, 1)
        self.block3 = StrictGatedBlock(base_channels*4, use_residual=False)
        
        # Bottleneck (Aquí permitimos residuales para profundidad)
        self.down3 = nn.MaxPool2d(2)
        self.expand3 = nn.Conv2d(base_channels*4, base_channels*8, 1)
        self.bottleneck = nn.Sequential(
            StrictGatedBlock(base_channels*8, use_residual=True), 
            StrictGatedBlock(base_channels*8, use_residual=True)
        )

        # 2. DECODER (Bilinear para video)
        
        # Up 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce3 = nn.Conv2d(base_channels*8 + base_channels*4, base_channels*4, 1)
        # En el decoder activamos residuales para refinar
        self.dec3 = StrictGatedBlock(base_channels*4, use_residual=True)
        
        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce2 = nn.Conv2d(base_channels*4 + base_channels*2, base_channels*2, 1)
        self.dec2 = StrictGatedBlock(base_channels*2, use_residual=True)
        
        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce1 = nn.Conv2d(base_channels*2 + base_channels, base_channels, 1)
        self.dec1 = StrictGatedBlock(base_channels, use_residual=True)
        
        # 3. OUTPUT
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1 = self.block1(x1) # Sin residual, transforma agresivamente
        
        x2 = self.down1(x1)
        x2 = self.expand1(x2)
        x2 = self.block2(x2)
        
        x3 = self.down2(x2)
        x3 = self.expand2(x3)
        x3 = self.block3(x3)
        
        x4 = self.down3(x3)
        x4 = self.expand3(x4)
        x4 = self.bottleneck(x4)
        
        # Decoder
        x_up3 = self.up3(x4)
        if x_up3.shape != x3.shape:
             x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up3, x3], dim=1)
        x = self.reduce3(x)
        x = self.dec3(x)
        
        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape:
             x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up2, x2], dim=1)
        x = self.reduce2(x)
        x = self.dec2(x)
        
        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape:
             x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up1, x1], dim=1)
        x = self.reduce1(x)
        x = self.dec1(x)
        
        return self.tanh(self.outc(x))