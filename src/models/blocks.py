import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ACTIVATION & NORM HELPERS ---

def get_activation(activation_name):
    name = activation_name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu': # Swish
        return nn.SiLU(inplace=True)
    elif name == 'mish':
        return nn.Mish(inplace=True)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.Identity()

def get_norm(norm_type, channels, groups=32):
    name = norm_type.lower()
    if name == 'batch':
        return nn.BatchNorm2d(channels)
    elif name == 'instance':
        return nn.InstanceNorm2d(channels)
    elif name == 'group':
        # Ensure groups <= channels and divisible
        actual_groups = min(groups, channels)
        if channels % actual_groups != 0:
            actual_groups = 1 # Fallback to LayerNorm behavior effectively (conceptually, though LN is different dim) or reduced groups
            # Try to find a divisor
            for k in range(actual_groups, 0, -1):
                if channels % k == 0:
                    actual_groups = k
                    break
        return nn.GroupNorm(actual_groups, channels)
    elif name == 'none':
        return nn.Identity()
    else:
        return nn.BatchNorm2d(channels)

# --- BASIC BLOCKS ---

class ConvBlock(nn.Module):
    """Standard Conv -> Norm -> Act -> Conv -> Norm -> Act"""
    def __init__(self, in_ch, out_ch, activation='relu', norm_type='batch', groups=32, dropout=0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = get_norm(norm_type, out_ch, groups)
        self.act1 = get_activation(activation)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(norm_type, out_ch, groups)
        self.act2 = get_activation(activation)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act2(self.norm2(self.conv2(x)))
        return x

class ResBlock(nn.Module):
    """Pre-activation ResBlock: Norm -> Act -> Conv -> Norm -> Act -> Conv"""
    def __init__(self, in_ch, out_ch, activation='silu', norm_type='group', groups=32, dropout=0.0, use_1x1=True):
        super().__init__()
        
        self.norm1 = get_norm(norm_type, in_ch, groups)
        self.act1 = get_activation(activation)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        
        self.norm2 = get_norm(norm_type, out_ch, groups)
        self.act2 = get_activation(activation)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Identity()
        if in_ch != out_ch or use_1x1:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        res = x
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        return x + self.shortcut(res)

# --- ADVANCED GATES & MODULES ---

class AttentionGate(nn.Module):
    """
    Attention Gate for Skip Connections.
    g: Gate signal (from decoder/upsample)
    x: Skip connection features (from encoder)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
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

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    y = gamma * x + beta
    """
    def __init__(self, in_channels, cond_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        # If cond_dim is provided, we assume an external MLP generates gamma/beta
        # Otherwise, this layer expects gamma/beta to be passed in forward
        
    def forward(self, x, gamma, beta):
        # x: (B, C, H, W)
        # gamma, beta: (B, C)
        
        # Ensure dimensions match for broadcasting
        gamma = gamma.view(x.size(0), self.in_channels, 1, 1)
        beta = beta.view(x.size(0), self.in_channels, 1, 1)
        
        return x * gamma + beta

class AdaINBlock(nn.Module):
    """
    Adaptive Instance Normalization.
    Normalized content x scaled and shifted by style y.
    """
    def __init__(self):
        super().__init__()
        
    def calc_mean_std(self, feat, eps=1e-5):
        # feat: (B, C, H, W)
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, style_mean, style_std):
        # x: Content features
        # style_mean/std: From style vector or reference image
        content_mean, content_std = self.calc_mean_std(x)
        normalized = (x - content_mean) / content_std
        return normalized * style_std + style_mean

class SmartFilter(nn.Module):
    """1x1 Conv + Activation to 'break' direct skip connection info"""
    def __init__(self, channels, activation='mish'):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            get_activation(activation)
        )
    def forward(self, x):
        return self.op(x)
