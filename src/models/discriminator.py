"""
Discriminador PatchGAN para entrenamiento GAN
"""

import torch
import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        
        # Si usamos InstanceNorm, no necesitamos bias porque la normalización affine=False lo hace redundante/dañino a veces.
        # Pero PatchGAN suele usarse con BatchNorm o InstanceNorm.
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*sequence) # Wrapper inicial (no lo usamos directo para poder acceder a features)
        
        # Construimos las capas manualmente para tener acceso fácil
        self.layers = nn.ModuleList()
        self.layers.append(sequence[0]) # Conv
        self.layers.append(sequence[1]) # Leaky

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
            )
            self.layers.append(norm_layer(ndf * nf_mult))
            self.layers.append(nn.LeakyReLU(0.2, True))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        # Layer n_layers
        self.layers.append(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        )
        self.layers.append(norm_layer(ndf * nf_mult))
        self.layers.append(nn.LeakyReLU(0.2, True))

        # Output Layer
        self.layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        )
        
        # No Sigmoid! Usamos MSELoss (LSGAN) o BCEWithLogits, numéricamente más estable.

    def forward(self, input):
        """Standard forward."""
        # return self.model(input) # No usamos esto para poder soportar Feature Matching
        
        features = []
        x = input
        
        # Recorremos el ModuleList
        # Ojo: nuestra lógica de construcción fue lineal.
        # Conv -> Leaky -> (Conv -> Norm -> Leaky)*N -> Conv -> Norm -> Leaky -> Conv_final
        
        # Para Feature Matching, queremos las salidas de las capas intermedias ACTIVADAS.
        
        for layer in self.layers:
            x = layer(x)
            # Guardar features después de LeakyReLU o de la capa final
            if isinstance(layer, nn.LeakyReLU) or (isinstance(layer, nn.Conv2d) and layer == self.layers[-1]):
                features.append(x)
                
        return features[-1], features[:-1] 
        # Retorna: (predicción final, lista de features intermedios)
