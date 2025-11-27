import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDoubleConv(nn.Module):
    """
    Bloque Clásico Mejorado:
    1. Usa GroupNorm (mejor para pocos datos que BatchNorm).
    2. Agrega Residual Connection (Input + Output) para estabilidad.
    """
    def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
        super().__init__()
        # Si cambiamos de canales, necesitamos adaptar el input para la suma residual
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = self._get_norm(norm, out_ch)
        self.act1 = self._get_act(activation)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = self._get_norm(norm, out_ch)
        self.act2 = self._get_act(activation)

    def _get_norm(self, norm, channels):
        if norm == 'batch': return nn.BatchNorm2d(channels)
        if norm == 'instance': return nn.InstanceNorm2d(channels, affine=True)
        # GroupNorm estable: 32 grupos por defecto, o 1 grupo si hay pocos canales
        num_groups = 32 if channels % 32 == 0 else (channels // 8 if channels >= 8 else 1)
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def _get_act(self, name):
        if name == 'relu': return nn.ReLU(inplace=True)
        if name == 'leaky': return nn.LeakyReLU(0.2, inplace=True)
        return nn.SiLU(inplace=True) # SiLU (Swish) es mejor hoy en día

    def forward(self, x):
        identity = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        return out + identity  # <--- MAGIA: Residual Connection

class Down(nn.Module):
    """Downscaling con Strided Convolution (Mejor que MaxPool para texturas)"""
    def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
        super().__init__()
        self.body = nn.Sequential(
            # Stride 2 reduce la dimensión a la mitad aprendiendo features
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False), 
            ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)
        )

    def forward(self, x):
        return self.body(x)

# class Up(nn.Module):
#     """Upscaling 'Anti-Artifacts' usando Bilinear + Conv"""
#     def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
#         super().__init__()
#         # 1. Upsample "tonto" pero suave
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         # 2. Convolución para reducir canales antes de concatenar (ahorra memoria)
#         self.conv_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
#         # 3. Bloque procesador
#         self.conv = ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)

#     def forward(self, x1, x2):
#         # x1: input desde abajo (bottleneck), x2: skip connection
#         x1 = self.up(x1)
#         x1 = self.conv_reduce(x1) # Reducimos canales para igualar dimensiones usuales
        
#         # Padding dinámico por si las dimensiones no cuadran perfecto
#         diffY = x2.size(2) - x1.size(2)
#         diffX = x2.size(3) - x1.size(3)
#         if diffY != 0 or diffX != 0:
#             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
#         # Concatenación
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group', activation='silu', use_attention=True): # <--- Nuevo flag
        super().__init__()
        self.use_attention = use_attention
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        
        # Atención: Filtra el skip connection (x2) usando la señal de abajo (x1)
        if use_attention:
            # in_ch//2 es lo que viene de abajo reduccido, out_ch es lo que viene del skip (usualmente)
            self.att = AttentionBlock(F_g=in_ch // 2, F_l=out_ch, F_int=out_ch // 2)
            
        self.conv = ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_reduce(x1) 
        
        # Ajuste de tamaño por padding (igual que antes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # APLICAR ATENCIÓN AL SKIP CONNECTION
        if self.use_attention:
            x2 = self.att(g=x1, x=x2) # <--- Aquí filtramos la arruga del skip
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    Filtra las características que vienen del Skip Connection.
    Ayuda a la red a ignorar información irrelevante (como la arruga que quieres borrar)
    antes de fusionarla con el decoder.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # F_g: Canales viniendo desde abajo (Gating signal)
        # F_l: Canales del skip connection (Local features)
        # F_int: Canales intermedios (menor costo computacional)
        
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
        # g: input del decoder (gating)
        # x: input del encoder (skip connection)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Redimensionar g1 para que coincida con x1 si es necesario (por strides)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64, # Mantén esto en 64 para buena calidad
        depth=4,          # 4 es el estándar equilibrado
        norm='group',
        activation='silu'
    ):
        super().__init__()
        
        self.inc = ResDoubleConv(in_channels, base_channels, norm=norm, activation=activation)
        
        # Construcción dinámica del Encoder
        self.downs = nn.ModuleList()
        current_ch = base_channels
        
        # Encoder (depth-1 bajadas)
        # Canales: 64 -> 128 -> 256 -> 512
        filters = [base_channels * (2**i) for i in range(depth)]
        
        for i in range(depth - 1):
            self.downs.append(Down(filters[i], filters[i+1], norm=norm, activation=activation))
            
        # Bottleneck (sin cambiar tamaño espacial, solo procesar)
        # Canales: 512 -> 512
        self.bottleneck = ResDoubleConv(filters[-1], filters[-1], norm=norm, activation=activation)
        
        # Decoder
        self.ups = nn.ModuleList()
        # Invertimos filtros para subir
        # 512 -> 256 -> 128 -> 64
        for i in reversed(range(depth - 1)):
            self.ups.append(Up(filters[i+1], filters[i], norm=norm, activation=activation))
            
        # Output Final
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh() # Asumiendo target en [-1, 1]

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        skips = [x1]
        
        xi = x1
        for down in self.downs:
            xi = down(xi)
            skips.append(xi)
            
        # Bottleneck
        x_bot = self.bottleneck(skips[-1])
        
        # Decoder
        x = x_bot
        # Iteramos ups y skips en reverso (excluyendo el último skip que es el input del bottleneck)
        skips = skips[:-1][::-1] # Invertir lista de skips restantes
        
        for i, up in enumerate(self.ups):
            x = up(x, skips[i])
            
        out = self.outc(x)
        return self.tanh(out)