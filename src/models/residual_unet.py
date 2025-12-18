import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock

class ResidualUNet(nn.Module):
    """
    Residual U-Net.
    Features:
    - Global Residual Learning (Output = Input + Network(Input))
    - ResBlock backbone (Pre-activation)
    - Dilated Bottleneck (Optional)
    """
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 base_channels=64, 
                 norm_type='group', 
                 activation='silu', 
                 dropout=0.0,
                 use_dilated_bottleneck=True):
        super().__init__()
        
        self.base_channels = base_channels
        self.use_dilated_bottleneck = use_dilated_bottleneck
        
        # --- ENCODER ---
        self.inc = ResBlock(in_channels, base_channels, activation, norm_type, dropout=dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels, base_channels*2, activation, norm_type, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels*2, base_channels*4, activation, norm_type, dropout=dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels*4, base_channels*8, activation, norm_type, dropout=dropout))

        # --- BOTTLENECK ---
        if use_dilated_bottleneck:
            # Dilated Conv Block for larger receptive field
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base_channels*8, base_channels*16, 3, padding=2, dilation=2),
                nn.GroupNorm(32, base_channels*16),
                nn.SiLU(inplace=True),
                nn.Conv2d(base_channels*16, base_channels*8, 3, padding=2, dilation=2),
                nn.GroupNorm(32, base_channels*8),
                nn.SiLU(inplace=True)
            )
        else:
            self.bottleneck = ResBlock(base_channels*8, base_channels*16, activation, norm_type, dropout=dropout)

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels to conv3: skip(8*base) + upsampled(8*base or 16*base depending on bottleneck)
        # If bottleneck is dilated and preserves channels, it's 8*base. If standard resblock, might change.
        # Let's align channels: Bottleneck output should be compatible.
        # If standard bottleneck (ResBlock in->out*2 ? No, ResBlock maintains or changes if spec.
        # Let's assume bottleneck preserves ch*8 for simplicity or expands. 
        # Actually my ResBlock logic: ResBlock(in, out).
        
        # Correction: If bottleneck is ResBlock(8, 16), output is 16. Up3 takes 16.
        bn_out = base_channels*8 if use_dilated_bottleneck else base_channels*16
        
        self.conv3 = ResBlock(bn_out + base_channels*4, base_channels*4, activation, norm_type, dropout=dropout)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = ResBlock(base_channels*4 + base_channels*2, base_channels*2, activation, norm_type, dropout=dropout)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ResBlock(base_channels*2 + base_channels, base_channels, activation, norm_type, dropout=dropout)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x_bn = self.bottleneck(x4)

        # Decoder
        x_up3 = self.up3(x_bn)
        if x_up3.shape != x3.shape: x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up3, x3], dim=1)
        x = self.conv3(x)

        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape: x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up2, x2], dim=1)
        x = self.conv2(x)

        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape: x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up1, x1], dim=1)
        x = self.conv1(x)
        
        residual = self.tanh(self.outc(x))
        
        # Global Residual Step
        return x + residual # Assuming x is input? No, x is local var.
                            # Use input x!
        
        # WARNING: x + residual requires dimensions match.
        # If input is 3ch and output is 3ch, this works.
        
    def forward(self, input_tensor):
        x1 = self.inc(input_tensor)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bn = self.bottleneck(x4)

        x_up3 = self.up3(x_bn)
        if x_up3.shape != x3.shape: x_up3 = F.interpolate(x_up3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up3, x3], dim=1)
        x = self.conv3(x)

        x_up2 = self.up2(x)
        if x_up2.shape != x2.shape: x_up2 = F.interpolate(x_up2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up2, x2], dim=1)
        x = self.conv2(x)

        x_up1 = self.up1(x)
        if x_up1.shape != x1.shape: x_up1 = F.interpolate(x_up1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up1, x1], dim=1)
        x = self.conv1(x)

        residual = self.tanh(self.outc(x))
        
        return torch.clamp(input_tensor + residual, -1.0, 1.0)
