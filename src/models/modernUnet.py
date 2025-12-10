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