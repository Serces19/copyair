import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model, Swinv2Config

class SwinDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Convolution to reduce channels after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None:
            # Handle potential size mismatch due to padding/odd dimensions
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)

class SwinV2UNet(nn.Module):
    def __init__(self, 
                 img_size=256, 
                 in_channels=3, 
                 out_channels=3, 
                 swin_type='tiny', 
                 pretrained=True,
                 use_global_residual=True):
        super().__init__()
        self.use_global_residual = use_global_residual
        
        # Map swin_type to Hugging Face model ID
        model_map = {
            'tiny': 'microsoft/swinv2-tiny-patch4-window8-256',
            'small': 'microsoft/swinv2-small-patch4-window8-256',
            'base': 'microsoft/swinv2-base-patch4-window8-256',
            'large': 'microsoft/swinv2-large-patch4-window12-192-22k' # Note: different window/size
        }
        
        model_id = model_map.get(swin_type, model_map['tiny'])
        
        print(f"Initializing SwinV2UNet with {model_id} (pretrained={pretrained})")
        
        if pretrained:
            self.encoder = Swinv2Model.from_pretrained(model_id)
        else:
            config = Swinv2Config.from_pretrained(model_id)
            self.encoder = Swinv2Model(config)
            
        # Ensure we output hidden states
        self.encoder.config.output_hidden_states = True
        
        # Feature channels for SwinV2 Tiny/Small/Base
        # Tiny: [96, 192, 384, 768]
        # Small: [96, 192, 384, 768]
        # Base: [128, 256, 512, 1024]
        
        embed_dim = self.encoder.config.embed_dim
        depths = self.encoder.config.depths
        num_heads = self.encoder.config.num_heads
        
        # Calculate channels for each stage (usually embed_dim * 2^i)
        self.enc_channels = [embed_dim * (2 ** i) for i in range(4)]
        
        # Decoder
        # Bottleneck is the last stage output (H/32, W/32)
        
        # Up 1: (H/32 -> H/16)
        self.up1 = SwinDecoderBlock(self.enc_channels[3], self.enc_channels[2], skip_channels=self.enc_channels[2])
        
        # Up 2: (H/16 -> H/8)
        self.up2 = SwinDecoderBlock(self.enc_channels[2], self.enc_channels[1], skip_channels=self.enc_channels[1])
        
        # Up 3: (H/8 -> H/4)
        self.up3 = SwinDecoderBlock(self.enc_channels[1], self.enc_channels[0], skip_channels=self.enc_channels[0])
        
        # Up 4: (H/4 -> H) - No skip from encoder here as Swin starts at H/4
        # We project to base channels
        self.up4 = SwinDecoderBlock(self.enc_channels[0], embed_dim // 2, skip_channels=0)
        
        # Final upsampling to match input resolution (H/4 -> H)
        # The Swin Transformer patch embedding reduces by 4x.
        # So up4 brings us to H, wait.
        # Swin stages:
        # Input: H, W
        # Stage 1: H/4, W/4
        # Stage 2: H/8, W/8
        # Stage 3: H/16, W/16
        # Stage 4: H/32, W/32
        
        # Decoder flow:
        # x4 (H/32) -> up1 -> (H/16) + skip3 (H/16) -> out (H/16)
        # out (H/16) -> up2 -> (H/8) + skip2 (H/8) -> out (H/8)
        # out (H/8) -> up3 -> (H/4) + skip1 (H/4) -> out (H/4)
        # out (H/4) -> up4 -> (H) (Upsample 4x)
        
        # Let's adjust up4 to be a 4x upsample or two 2x upsamples.
        # Standard U-Net usually does 2x at a time.
        # But Swin's first stage is already downsampled 4x.
        # So we need to go from H/4 to H.
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(self.enc_channels[0], embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        input_image = x
        
        # Encoder
        # Swin expects (B, C, H, W) but transformers might need normalization or specific handling?
        # The model handles standard normalization if configured, but usually we pass tensors.
        # SwinV2Model forward returns BaseModelOutputWithPoolingAndCrossAttentions
        
        outputs = self.encoder(x)
        
        # Hidden states: (embedding_output, stage1, stage2, stage3, stage4)
        # Note: hidden_states[0] is the output of the embedding layer (H/4, W/4)
        # hidden_states[1] is output of stage 1 (H/4, W/4)
        # hidden_states[2] is output of stage 2 (H/8, W/8)
        # hidden_states[3] is output of stage 3 (H/16, W/16)
        # hidden_states[4] is output of stage 4 (H/32, W/32)
        
        hidden_states = outputs.hidden_states
        
        # We use stages 1, 2, 3, 4 for skips and bottleneck
        # Skip 1: hidden_states[0] (H/4)
        # Skip 2: hidden_states[1] (H/8)
        # Skip 3: hidden_states[2] (H/16)
        # Bottleneck: hidden_states[3] (H/32)
        
        # Reshape features from (B, H*W, C) to (B, C, H, W)
        
        def reshape_feat(feat):
            B, L, C = feat.shape
            size = int(L**0.5)
            return feat.transpose(1, 2).view(B, C, size, size)

        # Extract and reshape
        # Indices based on debug output:
        # HS 0: H/4 (96)
        # HS 1: H/8 (192)
        # HS 2: H/16 (384)
        # HS 3: H/32 (768)
        
        s1 = reshape_feat(hidden_states[0]) # H/4
        s2 = reshape_feat(hidden_states[1]) # H/8
        s3 = reshape_feat(hidden_states[2]) # H/16
        s4 = reshape_feat(hidden_states[3]) # H/32
        
        # Decoder
        x = self.up1(s4, s3) # H/32 -> H/16
        x = self.up2(x, s2)  # H/16 -> H/8
        x = self.up3(x, s1)  # H/8 -> H/4
        
        # Final upsample
        x = self.final_up(x) # H/4 -> H
        
        out = self.tanh(x)
        
        if self.use_global_residual:
            return torch.clamp(input_image + out, -1, 1)
        else:
            return out
