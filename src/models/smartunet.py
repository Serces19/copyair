import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, AttentionGate, SmartFilter

class SmartUNet(nn.Module):
    """
    Smart U-Net: U-Net configurable para VFX con skip connections 100% persistentes.
    - use_attention: Compuertas de atención (Attention Gates) en las conexiones skip.
    - use_smart_filter: Filtro convolucional 1x1 en las conexiones skip.
    """
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=64, 
                 norm_type='group', 
                 activation='silu', 
                 dropout=0.0,
                 use_attention=False,
                 use_smart_filter=False,
                 **kwargs):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_smart_filter = use_smart_filter
        
        # --- ENCODER ---
        self.inc = ConvBlock(in_channels, base_channels, activation, norm_type, dropout=dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels, base_channels*2, activation, norm_type, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels*2, base_channels*4, activation, norm_type, dropout=dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels*4, base_channels*8, activation, norm_type, dropout=dropout))

        # --- STRATEGIES ---
        if self.use_smart_filter:
            self.skip_conv3 = SmartFilter(base_channels*4, activation)
            self.skip_conv2 = SmartFilter(base_channels*2, activation)
            self.skip_conv1 = SmartFilter(base_channels, activation)
        
        if self.use_attention:
            self.att_gate3 = AttentionGate(F_g=base_channels*8, F_l=base_channels*4, F_int=base_channels*2)
            self.att_gate2 = AttentionGate(F_g=base_channels*4, F_l=base_channels*2, F_int=base_channels)
            self.att_gate1 = AttentionGate(F_g=base_channels*2, F_l=base_channels, F_int=base_channels//2)

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

        # Decoder - Block 3
        x_up3 = self.up3(x4)
        if x_up3.shape != x3.shape:
            x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        
        skip3 = x3
        if self.use_smart_filter:
            skip3 = self.skip_conv3(skip3)
        if self.use_attention:
            skip3 = self.att_gate3(x_up3, skip3)
        
        x = torch.cat([x_up3, skip3], dim=1)
        x = self.conv3(x)

        # Decoder - Block 2
        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape:
            x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        skip2 = x2
        if self.use_smart_filter:
            skip2 = self.skip_conv2(skip2)
        if self.use_attention:
            skip2 = self.att_gate2(x_up2, skip2)
        
        x = torch.cat([x_up2, skip2], dim=1)
        x = self.conv2(x)

        # Decoder - Block 1
        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape:
            x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        skip1 = x1
        if self.use_smart_filter:
            skip1 = self.skip_conv1(skip1)
        if self.use_attention:
            skip1 = self.att_gate1(x_up1, skip1)
        
        x = torch.cat([x_up1, skip1], dim=1)
        x = self.conv1(x)

        return self.tanh(self.outc(x))