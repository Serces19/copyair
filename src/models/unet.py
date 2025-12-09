# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -----------------------------------------------------------------------------
# # 1. BLOQUES BÁSICOS (Robustos y Modernos)
# # -----------------------------------------------------------------------------

# class ResDoubleConv(nn.Module):
#     """
#     Bloque Residual con GroupNorm y SiLU.
#     Soporta Dilatación para aumentar el campo receptivo sin perder resolución.
#     """
#     def __init__(self, in_ch, out_ch, norm='group', activation='silu', dilation=1):
#         super().__init__()
        
#         # Adaptador para el residual si cambian los canales
#         self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
#         # Conv 1
#         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
#         self.norm1 = self._get_norm(norm, out_ch)
#         self.act1 = self._get_act(activation)

#         # Conv 2
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
#         self.norm2 = self._get_norm(norm, out_ch)
#         self.act2 = self._get_act(activation)

#     def _get_norm(self, norm, channels):
#         if norm == 'batch': return nn.BatchNorm2d(channels)
#         if norm == 'instance': return nn.InstanceNorm2d(channels, affine=True)
#         # GroupNorm inteligente: evita errores con pocos canales
#         num_groups = 32 if channels % 32 == 0 else (channels // 8 if channels >= 8 else 1)
#         return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

#     def _get_act(self, name):
#         if name == 'relu': return nn.ReLU(inplace=True)
#         if name == 'leaky': return nn.LeakyReLU(0.2, inplace=True)
#         return nn.SiLU(inplace=True) 

#     def forward(self, x):
#         identity = self.residual_conv(x)
        
#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.act1(out)
        
#         out = self.conv2(out)
#         out = self.norm2(out)
#         out = self.act2(out)
        
#         return out + identity  # Residual Connection local

# class Down(nn.Module):
#     """Bajada con Strided Convolution (Conserva textura mejor que MaxPool)"""
#     def __init__(self, in_ch, out_ch, norm='group', activation='silu'):
#         super().__init__()
#         self.body = nn.Sequential(
#             # Stride 2 reduce dimensión espacial
#             nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False), 
#             ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)
#         )

#     def forward(self, x):
#         return self.body(x)

# class AttentionBlock(nn.Module):
#     """Attention Gate estándar (Oktay et al.)"""
#     def __init__(self, F_g, F_l, F_int):
#         super().__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
        
#         # Manejo robusto de dimensiones
#         if g1.shape[2:] != x1.shape[2:]:
#             g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)

#         return x * psi

# class Up(nn.Module):
#     """Subida con Upsample Bilineal + Conv (Evita Checkerboard Artifacts)"""
#     def __init__(self, in_ch, out_ch, norm='group', activation='silu', use_attention=False):
#         super().__init__()
#         self.use_attention = use_attention
        
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        
#         if use_attention:
#             self.att = AttentionBlock(F_g=in_ch // 2, F_l=out_ch, F_int=out_ch // 2)
            
#         self.conv = ResDoubleConv(in_ch, out_ch, norm=norm, activation=activation)

#     def forward(self, x1, x2):
#         # x1: input desde abajo (bottleneck), x2: skip connection
#         x1 = self.up(x1)
#         x1 = self.conv_reduce(x1) 
        
#         # Padding dinámico para dimensiones impares
#         diffY = x2.size(2) - x1.size(2)
#         diffX = x2.size(3) - x1.size(3)
#         if diffY != 0 or diffX != 0:
#             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
#         # Atención opcional
#         if self.use_attention:
#             x2 = self.att(g=x1, x=x2)
            
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

# # -----------------------------------------------------------------------------
# # 2. U-NET MODERNA (High Fidelity Overfitting)
# # -----------------------------------------------------------------------------

# class UNet(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         out_channels=3,
#         base_channels=64, 
#         depth=4,          
#         norm='group',
#         activation='silu',
#         use_attention=False, # <--- Recomendado False para inpainting puro, True si quieres probar
#         use_global_residual=True # <--- CRÍTICO: Activar para aprender solo el cambio
#     ):
#         super().__init__()
#         self.use_global_residual = use_global_residual
        
#         # Capa inicial
#         self.inc = ResDoubleConv(in_channels, base_channels, norm=norm, activation=activation)
        
#         # Encoder
#         self.downs = nn.ModuleList()
#         filters = [base_channels * (2**i) for i in range(depth)]
        
#         for i in range(depth - 1):
#             self.downs.append(Down(filters[i], filters[i+1], norm=norm, activation=activation))
            
#         # Bottleneck DILATADO (DeepMind Fix #3: Contexto Global)
#         # Usamos dilation=2 para ver más contexto (arrugas grandes) sin perder resolución
#         self.bottleneck = ResDoubleConv(
#             filters[-1], filters[-1], 
#             norm=norm, activation=activation, 
#             dilation=2 
#         )
        
#         # Decoder
#         self.ups = nn.ModuleList()
#         for i in reversed(range(depth - 1)):
#             self.ups.append(Up(
#                 filters[i+1], filters[i], 
#                 norm=norm, activation=activation, 
#                 use_attention=use_attention
#             ))
            
#         # Output Final
#         self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
#         # Inicializamos la última capa cerca de cero para facilitar el aprendizaje residual
#         nn.init.normal_(self.outc.weight, mean=0.0, std=1e-5)
#         nn.init.constant_(self.outc.bias, 0.0)

#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         # Guardamos input para el Global Skip
#         x_input = x
        
#         # Encoder
#         x1 = self.inc(x)
#         skips = [x1]
        
#         xi = x1
#         for down in self.downs:
#             xi = down(xi)
#             skips.append(xi)
            
#         # Bottleneck (Dilatado)
#         x_bot = self.bottleneck(skips[-1])
        
#         # Decoder
#         x = x_bot
#         skips = skips[:-1][::-1]
        
#         for i, up in enumerate(self.ups):
#             x = up(x, skips[i])
            
#         # Predicción del Residual (La máscara de cambios)
#         out = self.outc(x)
#         residual = self.tanh(out)
        
#         # GLOBAL SKIP CONNECTION (DeepMind Fix #2)
#         # Si True: Output = Input Original + Residual Predicho
#         # Esto hace que la red solo tenga que aprender "cómo borrar la arruga",
#         # en lugar de "cómo dibujar una cara entera".
#         if self.use_global_residual:
#             final_out = x_input + residual
#         else:
#             final_out = residual
            
#         return torch.clamp(final_out, -1, 1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleConvBlock(nn.Module):
#     """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_channels=64):
#         super().__init__()
        
#         # --- ENCODER ---
#         self.inc = SimpleConvBlock(in_channels, base_channels)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels, base_channels*2))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*2, base_channels*4))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*4, base_channels*8))

#         # --- DECODER ---
#         # Up3
#         self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv3 = SimpleConvBlock(base_channels*8 + base_channels*4, base_channels*4)
        
#         # Up2
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv2 = SimpleConvBlock(base_channels*4 + base_channels*2, base_channels*2)
        
#         # Up1
#         self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv1 = SimpleConvBlock(base_channels*2 + base_channels, base_channels)

#         # --- OUTPUT ---
#         self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
#         self.tanh = nn.Tanh() # Asumiendo que tus datos van de [-1, 1]

#     def forward(self, x):
#         # Encoder
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3) # Bottleneck

#         # Decoder
        
#         # Bloque 3
#         x_up3 = self.up3(x4)
#         # Padding por si las dimensiones no cuadran (impar)
#         if x_up3.shape != x3.shape:
#             x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x_up3, x3], dim=1)
#         x = self.conv3(x)

#         # Bloque 2
#         x_up2 = self.up2(x)
#         if x_up2.shape != x2.shape:
#             x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x_up2, x2], dim=1)
#         x = self.conv2(x)

#         # Bloque 1
#         x_up1 = self.up1(x)
#         if x_up1.shape != x1.shape:
#             x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
#         x = torch.cat([x_up1, x1], dim=1)
#         x = self.conv1(x)

#         # Salida
#         return self.tanh(self.outc(x))






import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # F_g: Canales de la señal de compuerta (viene del decoder/up)
        # F_l: Canales de la señal a filtrar (viene del encoder/skip)
        # F_int: Canales intermedios (generalmente F_l / 2)
        
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
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SimpleConvBlock(nn.Module):
    """Conv -> Norm -> Activation -> Conv -> Norm -> Activation"""
    def __init__(self, in_ch, out_ch, activation='relu', norm_type='batch', groups=32, dropout=0.0):
        super().__init__()
        
        # Función de activación
        activation_fn = self._get_activation(activation)
        
        # Capa de normalización
        norm1 = self._get_norm(norm_type, out_ch, groups)
        norm2 = self._get_norm(norm_type, out_ch, groups)
        
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            norm1,
            activation_fn,
        ]
        
        # Añadir dropout si es mayor que 0
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            norm2,
            activation_fn,
        ])
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.conv = nn.Sequential(*layers)

    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
            'gelu': nn.GELU(),
            'mish': nn.Mish(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.1, inplace=True)
        }
        return activations.get(activation.lower(), nn.ReLU(inplace=True))

    def _get_norm(self, norm_type, channels, groups=32):
        if norm_type.lower() == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type.lower() == 'group':
            # Asegurar que groups no sea mayor que channels
            actual_groups = min(groups, channels)
            # Asegurar que channels sea divisible por groups
            if channels % actual_groups != 0:
                actual_groups = 1
            return nn.GroupNorm(actual_groups, channels)
        elif norm_type.lower() == 'instance':
            return nn.InstanceNorm2d(channels)
        else:
            return nn.BatchNorm2d(channels)  # default

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=64,
                 activation='silu',
                 norm_type='group',
                 groups=16,
                 dropout=0.0,
                 use_attention=False,
                 output_activation='tanh'):
        super().__init__()
        
        self.use_attention = use_attention
        
        # --- ENCODER ---
        self.inc = SimpleConvBlock(in_channels, base_channels, activation, norm_type, groups, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels, base_channels*2, activation, norm_type, groups, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*2, base_channels*4, activation, norm_type, groups, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*4, base_channels*8, activation, norm_type, groups, dropout))

        # --- ESTRATEGIA B: SMART FILTER (1x1 CONVS) ---
        # Descomenta esto para usar filtros 1x1 en las conexiones
        # Nota: Mantenemos los canales iguales para no romper las dimensiones del decoder, 
        # pero añadimos no-linealidad para "romper" la copia directa.
        
        # self.skip_conv3 = nn.Sequential(nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1), nn.Mish())
        # self.skip_conv2 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels*2, kernel_size=1), nn.Mish())
        # self.skip_conv1 = nn.Sequential(nn.Conv2d(base_channels, base_channels, kernel_size=1), nn.Mish())


        # --- ESTRATEGIA C: ATTENTION GATES ---
        # Descomenta esto para usar Attention Gates reales
        # F_g (gate) = canales del upsample, F_l (local) = canales del skip
        
        # self.att_gate3 = AttentionGate(F_g=base_channels*8, F_l=base_channels*4, F_int=base_channels*2)
        # self.att_gate2 = AttentionGate(F_g=base_channels*4, F_l=base_channels*2, F_int=base_channels)
        # self.att_gate1 = AttentionGate(F_g=base_channels*2, F_l=base_channels, F_int=base_channels//2)


        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = SimpleConvBlock(base_channels*8 + base_channels*4, base_channels*4, activation, norm_type, groups, dropout)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = SimpleConvBlock(base_channels*4 + base_channels*2, base_channels*2, activation, norm_type, groups, dropout)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = SimpleConvBlock(base_channels*2 + base_channels, base_channels, activation, norm_type, groups, dropout)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.output_activation = self._get_output_activation(output_activation)

    # ... (Tus métodos auxiliares _get_output_activation, etc. van aquí igual que antes) ...
    def _get_output_activation(self, activation):
        activations = {'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(inplace=True), 'none': nn.Identity(), 'leakyrelu': nn.LeakyReLU(0.1, inplace=True)}
        return activations.get(activation.lower(), nn.Tanh())

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # Skip connection 1 (base_channels)
        x2 = self.down1(x1)      # Skip connection 2 (base_channels*2)
        x3 = self.down2(x2)      # Skip connection 3 (base_channels*4)
        x4 = self.down3(x3)      # Bottleneck

        # --- AQUÍ ESTÁN LAS MODIFICACIONES ---

        # ---------------------------------------------------------
        # BLOQUE 3 (Profundo)
        # ---------------------------------------------------------
        x_up3 = self.up3(x4)
        if x_up3.shape != x3.shape: x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)

        # Copia de la skip connection original
        skip3 = x3 
        
        # [ESTRATEGIA A: CORTAR CABLE]
        skip3 = skip3 * 0.0

        # [ESTRATEGIA B: SMART FILTER] (Descomentar)
        # skip3 = self.skip_conv3(skip3)

        # [ESTRATEGIA C: ATTENTION GATE] (Descomentar)
        # skip3 = self.att_gate3(g=x_up3, x=skip3)

        # [ESTRATEGIA D: DROP-SKIP] (Descomentar)
        # if self.training and torch.rand(1).item() < 0.5: # 50% probabilidad de cortar conexión
        #     skip3 = skip3 * 0.0

        # Concatenación (Esta es la skip connection actual)
        x = torch.cat([x_up3, skip3], dim=1) 
        x = self.conv3(x)

        # ---------------------------------------------------------
        # BLOQUE 2 (Medio)
        # ---------------------------------------------------------
        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape: x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        skip2 = x2

        # [ESTRATEGIA A: CORTAR CABLE]
        #skip2 = skip2 * 0.0

        # [ESTRATEGIA B: SMART FILTER] (Descomentar)
        # skip2 = self.skip_conv2(skip2)

        # [ESTRATEGIA C: ATTENTION GATE] (Descomentar)
        # skip2 = self.att_gate2(g=x_up2, x=skip2)

        # [ESTRATEGIA D: DROP-SKIP] (Descomentar)
        # if self.training and torch.rand(1).item() < 0.5:
        #     skip2 = skip2 * 0.0

        x = torch.cat([x_up2, skip2], dim=1)
        x = self.conv2(x)

        # ---------------------------------------------------------
        # BLOQUE 1 (Superficial - DETALLES FINOS)
        # ---------------------------------------------------------
        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape: x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        skip1 = x1


        # [ESTRATEGIA A: CORTAR CABLE] 
        # Forzamos a que la skip connection sea negro absoluto (ceros). 
        # El modelo se ve obligado a inventar los detalles.
        # skip1 = skip1 * 0.0

        # [ESTRATEGIA B: SMART FILTER] (Descomentar)
        # skip1 = self.skip_conv1(skip1)

        # [ESTRATEGIA C: ATTENTION GATE] (Descomentar)
        # skip1 = self.att_gate1(g=x_up1, x=skip1)

        # [ESTRATEGIA D: DROP-SKIP] (Descomentar)
        # if self.training and torch.rand(1).item() < 0.5:
        #     skip1 = skip1 * 0.0

        x = torch.cat([x_up1, skip1], dim=1)
        x = self.conv1(x)

        return self.output_activation(self.outc(x))