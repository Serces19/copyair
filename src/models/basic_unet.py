import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock

class BasicUNet(nn.Module):
    """
    Basic U-Net implementation.
    Simple, reliable, and clean. No fancy gates or residuals.
    """
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=64, 
                 norm_type='batch', 
                 activation='relu', 
                 dropout=0.0):
        super().__init__()
        
        # --- ENCODER ---
        self.inc = ConvBlock(in_channels, base_channels, activation, norm_type, dropout=dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels, base_channels*2, activation, norm_type, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels*2, base_channels*4, activation, norm_type, dropout=dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels*4, base_channels*8, activation, norm_type, dropout=dropout))

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = ConvBlock(base_channels*8 + base_channels*4, base_channels*4, activation, norm_type, dropout=dropout)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = ConvBlock(base_channels*4 + base_channels*2, base_channels*2, activation, norm_type, dropout=dropout)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(base_channels*2 + base_channels, base_channels, activation, norm_type, dropout=dropout)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x_up3 = self.up3(x4)
        if x_up3.shape != x3.shape:
             x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up3, x3], dim=1)
        x = self.conv3(x)

        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape:
             x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up2, x2], dim=1)
        x = self.conv2(x)

        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape:
             x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up1, x1], dim=1)
        x = self.conv1(x)

        return self.tanh(self.outc(x))
