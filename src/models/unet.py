import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. BLOQUES BÁSICOS (Robustos y Modernos)
# -----------------------------------------------------------------------------

class ResDoubleConv(nn.Module):
    """
    Bloque Residual con GroupNorm y SiLU.
    Soporta Dilatación para aumentar el campo receptivo sin perder resolución.
    """
    def __init__(self, in_ch, out_ch, norm='group', activation='silu', dilation=1):
        super().__init__()
        
        # Adaptador para el residual si cambian los canales
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # Conv 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = self._get_norm(norm, out_ch)
        self.act1 = self._get_act(activation)

        # Conv 2
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = self._get_norm(norm, out_ch)
        self.act2 = self._get_act(activation)

    def _get_norm(self, norm, channels):
        if norm == 'batch': return nn.BatchNorm2d(channels)
        if norm == 'instance': return nn.InstanceNorm2d(channels, affine=True)
        # GroupNorm inteligente: evita errores con pocos canales
        num_groups = 32 if channels % 32 == 0 else (channels // 8 if channels >= 8 else 1)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def _get_act(self, name):
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'leaky': return nn.LeakyReLU(0.2, inplace=True)
        return nn.SiLU(inplace=True) 

    def forward(self, x):
        identity = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        return out + identity  # Residual Connection local

class Down(nn.Module):
    """Bajada con Strided Convolution (Conserva textura mejor que MaxPool)"""
    def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
        super().__init__()
        self.body = nn.Sequential(
            # Stride 2 reduce dimensión espacial
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False), 
            ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)
        )

    def forward(self, x):
        return self.body(x)

class AttentionBlock(nn.Module):
    """Attention Gate estándar (Oktay et al.)"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Manejo robusto de dimensiones
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Up(nn.Module):
    """Subida con Upsample Bilineal + Conv (Evita Checkerboard Artifacts)"""
    def __init__(self, in_ch, out_ch, norm='group', activation='silu', use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        
        if use_attention:
            self.att = AttentionBlock(F_g=in_ch // 2, F_l=out_ch, F_int=out_ch // 2)
            
        self.conv = ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)

    def forward(self, x1, x2):
        # x1: input desde abajo (bottleneck), x2: skip connection
        x1 = self.up(x1)
        x1 = self.conv_reduce(x1) 
        
        # Padding dinámico para dimensiones impares
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Atención opcional
        if self.use_attention:
            x2 = self.att(g=x1, x=x2)
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# -----------------------------------------------------------------------------
# 2. U-NET MODERNA (High Fidelity Overfitting)
# -----------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64, 
        depth=4,          
        norm='group',
        activation='silu',
        use_attention=False, # <--- Recomendado False para inpainting puro, True si quieres probar
        use_global_residual=True # <--- CRÍTICO: Activar para aprender solo el cambio
    ):
        super().__init__()
        self.use_global_residual = use_global_residual
        
        # Capa inicial
        self.inc = ResDoubleConv(in_channels, base_channels, norm=norm, activation=activation)
        
        # Encoder
        self.downs = nn.ModuleList()
        filters = [base_channels * (2**i) for i in range(depth)]
        
        for i in range(depth - 1):
            self.downs.append(Down(filters[i], filters[i+1], norm=norm, activation=activation))
            
        # Bottleneck DILATADO (DeepMind Fix #3: Contexto Global)
        # Usamos dilation=2 para ver más contexto (arrugas grandes) sin perder resolución
        self.bottleneck = ResDoubleConv(
            filters[-1], filters[-1], 
            norm=norm, activation=activation, 
            dilation=2 
        )
        
        # Decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.ups.append(Up(
                filters[i+1], filters[i], 
                norm=norm, activation=activation, 
                use_attention=use_attention
            ))
            
        # Output Final
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Inicializamos la última capa cerca de cero para facilitar el aprendizaje residual
        nn.init.normal_(self.outc.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.outc.bias, 0.0)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Guardamos input para el Global Skip
        x_input = x
        
        # Encoder
        x1 = self.inc(x)
        skips = [x1]
        
        xi = x1
        for down in self.downs:
            xi = down(xi)
            skips.append(xi)
            
        # Bottleneck (Dilatado)
        x_bot = self.bottleneck(skips[-1])
        
        # Decoder
        x = x_bot
        skips = skips[:-1][::-1]
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[i])
            
        # Predicción del Residual (La máscara de cambios)
        out = self.outc(x)
        residual = self.tanh(out)
        
        # GLOBAL SKIP CONNECTION (DeepMind Fix #2)
        # Si True: Output = Input Original + Residual Predicho
        # Esto hace que la red solo tenga que aprender "cómo borrar la arruga",
        # en lugar de "cómo dibujar una cara entera".
        if self.use_global_residual:
            final_out = x_input + residual
        else:
            final_out = residual
            
        return torch.clamp(final_out, -1, 1)