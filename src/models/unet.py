import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # --- ENCODER ---
        self.inc = SimpleConvBlock(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels, base_channels*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*2, base_channels*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SimpleConvBlock(base_channels*4, base_channels*8))

        # --- DECODER ---
        # Up3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = SimpleConvBlock(base_channels*8 + base_channels*4, base_channels*4)
        
        # Up2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = SimpleConvBlock(base_channels*4 + base_channels*2, base_channels*2)
        
        # Up1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = SimpleConvBlock(base_channels*2 + base_channels, base_channels)

        # --- OUTPUT ---
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh() # Asumiendo que tus datos van de [-1, 1]

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # Bottleneck

        # Decoder
        
        # Bloque 3
        x_up3 = self.up3(x4)
        # Padding por si las dimensiones no cuadran (impar)
        if x_up3.shape != x3.shape:
            x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up3, x3], dim=1)
        x = self.conv3(x)

        # Bloque 2
        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape:
            x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up2, x2], dim=1)
        x = self.conv2(x)

        # Bloque 1
        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape:
            x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up1, x1], dim=1)
        x = self.conv1(x)

        # Salida
        return self.tanh(self.outc(x))


