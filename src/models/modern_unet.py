import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock, AttentionGate, FiLMLayer, AdaINBlock

class FiLMResBlock(nn.Module):
    """
    ResBlock with FiLM modulation.
    Structure: ... -> Conv -> Norm -> FiLM -> Act -> ...
    """
    def __init__(self, in_ch, out_ch, activation='silu', norm_type='group', groups=32, dropout=0.0, use_film=True):
        super().__init__()
        self.use_film = use_film
        
        # Pre-act structure
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        
        if use_film:
            self.film = FiLMLayer(out_ch)
            
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x, gamma=None, beta=None):
        res = x
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        if self.use_film and gamma is not None and beta is not None:
            x = self.film(x, gamma, beta)
            
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        
        return x + self.shortcut(res)

class ModernUNet(nn.Module):
    """
    U-Net Ultra-Completo para VFX (ModernUNet).
    Levels: 0 (Full), 1 (Half), 2 (Quarter), 3 (Eighth)
    Features: FiLM, AdaIN, Attention, Maps, Multi-Head.
    """
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3,
                 input_map_channels=0,
                 base_channels=64, 
                 norm_type='group', 
                 activation='silu', 
                 dropout=0.0,
                 use_film=True,
                 use_adain=False,
                 attention_type='self',
                 cond_dim=128):
        super().__init__()
        
        self.use_film = use_film
        self.use_adain = use_adain
        self.input_ch = in_channels + input_map_channels
        self.base_channels = base_channels
        
        # --- CONDITIONING (FiLM MLP) ---
        if use_film:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU()
            )
            self.film_heads = nn.ModuleList()
            # We have 4 Decoder Levels: 3, 2, 1, 0
            # Dec3 (Base*4), Dec2 (Base*2), Dec1 (Base), Dec0 (Base)
            self.film_head3 = nn.Linear(256, base_channels*4 * 2)
            self.film_head2 = nn.Linear(256, base_channels*2 * 2)
            self.film_head1 = nn.Linear(256, base_channels*2)
            self.film_head0 = nn.Linear(256, base_channels*2)

        # --- ENCODER ---
        # Stem
        self.stem = nn.Conv2d(self.input_ch, base_channels, 3, padding=1)
        
        # Level 0
        self.enc0 = ResBlock(base_channels, base_channels, activation, norm_type)
        self.down0 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1) # -> 64 ch, H/2
        
        # Level 1
        self.enc1 = ResBlock(base_channels, base_channels*2, activation, norm_type) # -> 128 ch
        self.down1 = nn.Conv2d(base_channels*2, base_channels*2, 3, stride=2, padding=1) # -> 128 ch, H/4
        
        # Level 2
        self.enc2 = ResBlock(base_channels*2, base_channels*4, activation, norm_type) # -> 256 ch
        self.down2 = nn.Conv2d(base_channels*4, base_channels*4, 3, stride=2, padding=1) # -> 256 ch, H/8
        
        # Level 3
        self.enc3 = ResBlock(base_channels*4, base_channels*8, activation, norm_type) # -> 512 ch
        self.down3 = nn.Conv2d(base_channels*8, base_channels*8, 3, stride=2, padding=1) # -> 512 ch, H/16

        # --- BOTTLENECK ---
        self.bottleneck_res = ResBlock(base_channels*8, base_channels*16, activation, norm_type) # -> 1024 ch
        
        if attention_type == 'self':
            self.attn = nn.MultiheadAttention(embed_dim=base_channels*16, num_heads=8, batch_first=True)
            self.att_norm = nn.GroupNorm(32, base_channels*16)
        
        if use_adain:
            self.adain = AdaINBlock()
            self.style_mlp = nn.Sequential(
                nn.Linear(cond_dim, 256),
                nn.SiLU(),
                nn.Linear(256, base_channels*16 * 2)
            )

        # --- DECODER ---
        
        # Up 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce3 = nn.Conv2d(base_channels*16 + base_channels*8, base_channels*8, 1) # Reduce before resblock
        self.att_gate3 = AttentionGate(base_channels*16, base_channels*8, base_channels*4)
        self.dec3 = FiLMResBlock(base_channels*8, base_channels*4, activation, norm_type, use_film=use_film) # -> 256 ch

        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce2 = nn.Conv2d(base_channels*4 + base_channels*4, base_channels*4, 1)
        self.att_gate2 = AttentionGate(base_channels*4, base_channels*4, base_channels*2)
        self.dec2 = FiLMResBlock(base_channels*4, base_channels*2, activation, norm_type, use_film=use_film) # -> 128 ch

        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce1 = nn.Conv2d(base_channels*2 + base_channels*2, base_channels*2, 1)
        self.att_gate1 = AttentionGate(base_channels*2, base_channels*2, base_channels)
        self.dec1 = FiLMResBlock(base_channels*2, base_channels, activation, norm_type, use_film=use_film) # -> 64 ch

        # Up 0 (To Full Res)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce0 = nn.Conv2d(base_channels + base_channels, base_channels, 1)
        self.att_gate0 = AttentionGate(base_channels, base_channels, base_channels//2)
        self.dec0 = FiLMResBlock(base_channels, base_channels, activation, norm_type, use_film=use_film) # -> 64 ch

        # --- HEADS ---
        self.out_rgb = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
        self.out_conf = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.out_mask = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond_vector=None, return_dict=False):
        # FiLM Preparation
        gammas, betas = {}, {}
        if self.use_film and cond_vector is not None:
            embed = self.cond_mlp(cond_vector)
            
            gb3 = self.film_head3(embed)
            gammas[3], betas[3] = gb3.chunk(2, dim=1)
            gb2 = self.film_head2(embed)
            gammas[2], betas[2] = gb2.chunk(2, dim=1)
            gb1 = self.film_head1(embed)
            gammas[1], betas[1] = gb1.chunk(2, dim=1)
            gb0 = self.film_head0(embed)
            gammas[0], betas[0] = gb0.chunk(2, dim=1)

        # Stem
        x = self.stem(x)
        
        # Encoder
        x0 = self.enc0(x)         # Full Res, 64ch
        
        x_d0 = self.down0(x0)
        x1 = self.enc1(x_d0)      # H/2, 128ch
        
        x_d1 = self.down1(x1)
        x2 = self.enc2(x_d1)      # H/4, 256ch
        
        x_d2 = self.down2(x2)
        x3 = self.enc3(x_d2)      # H/8, 512ch
        
        x_d3 = self.down3(x3)     # H/16
        
        # Bottleneck
        bn = self.bottleneck_res(x_d3) # H/16, 1024ch
        
        if hasattr(self, 'attn'):
            b, c, h, w = bn.shape
            bn_flat = bn.flatten(2).transpose(1, 2)
            attn_out, _ = self.attn(bn_flat, bn_flat, bn_flat)
            bn = bn + attn_out.transpose(1, 2).reshape(b, c, h, w)
            bn = self.att_norm(bn)

        if self.use_adain and cond_vector is not None:
            style_embed = self.style_mlp(cond_vector)
            s_mean, s_std = style_embed.chunk(2, dim=1)
            s_mean, s_std = s_mean.unsqueeze(-1).unsqueeze(-1), s_std.unsqueeze(-1).unsqueeze(-1)
            bn = self.adain(bn, s_mean, s_std)

        # Decoder 3
        u3 = self.up3(bn)
        if u3.shape[2:] != x3.shape[2:]: u3 = F.interpolate(u3, size=x3.shape[2:], mode='bilinear')
        
        s3 = self.att_gate3(u3, x3)
        c3 = torch.cat([u3, s3], dim=1) # 1024 + 512
        c3 = self.reduce3(c3)           # -> 512 or 256? Dec3 expects 8*B input to FilmBlock which outputs 4*B?
                                        # Reduce3 converts (16B+8B) -> 8B (512).
                                        # Dec3 inherits FiLMResBlock(8B -> 4B). Correct.
        d3 = self.dec3(c3, gammas.get(3), betas.get(3)) # -> 256

        # Decoder 2
        u2 = self.up2(d3)
        if u2.shape[2:] != x2.shape[2:]: u2 = F.interpolate(u2, size=x2.shape[2:], mode='bilinear')
        
        s2 = self.att_gate2(u2, x2)
        c2 = torch.cat([u2, s2], dim=1)
        c2 = self.reduce2(c2)
        d2 = self.dec2(c2, gammas.get(2), betas.get(2)) # -> 128

        # Decoder 1
        u1 = self.up1(d2)
        if u1.shape[2:] != x1.shape[2:]: u1 = F.interpolate(u1, size=x1.shape[2:], mode='bilinear')
        
        s1 = self.att_gate1(u1, x1)
        c1 = torch.cat([u1, s1], dim=1)
        c1 = self.reduce1(c1)
        d1 = self.dec1(c1, gammas.get(1), betas.get(1)) # -> 64

        # Decoder 0
        u0 = self.up0(d1)
        if u0.shape[2:] != x0.shape[2:]: u0 = F.interpolate(u0, size=x0.shape[2:], mode='bilinear')
        
        s0 = self.att_gate0(u0, x0)
        c0 = torch.cat([u0, s0], dim=1)
        c0 = self.reduce0(c0)
        d0 = self.dec0(c0, gammas.get(0), betas.get(0)) # -> 64

        # Output
        if not return_dict:
            return self.out_rgb(d0)
        else:
            return {
                'rgb': self.out_rgb(d0),
                'conf': self.out_conf(d0),
                'mask': self.out_mask(d0)
            }
