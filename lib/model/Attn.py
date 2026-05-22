import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock

class SwinConvAdapter(nn.Module):
    """
    将 Swin Transformer Block 封装成可以直接处理 (B, C, H, W) 数据的模块
    通常成对使用：一个 Window Attention + 一个 Shifted Window Attention
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        
        # 初始化 timm 的 SwinBlock
        # 注意：timm 的参数可能会随版本微调，但核心是这几个
        # 修复：timm 新版 SwinTransformerBlock 参数变化
        # 1. drop -> proj_drop
        # 2. 必须提供 input_resolution，这里使用 (window_size, window_size) 占位
        # 3. 启用 dynamic_mask=True 以支持动态分辨率
        self.block = SwinTransformerBlock(
            dim=dim,
            input_resolution=(window_size, window_size), 
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            proj_drop=drop,
            attn_drop=attn_drop,
            norm_layer=nn.LayerNorm, # Swin 标准是用 LayerNorm
            dynamic_mask=True
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. Padding: 确保 H 和 W 是 window_size 的整数倍，否则 Swin 会报错
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        
        # 2. 维度变换: (B, C, H', W') -> (B, H', W', C)
        # Swin 需要 Channel Last
        x = x.permute(0, 2, 3, 1)
        
        # 3. 进入 Swin Block
        x = self.block(x)
        
        # 4. 维度还原: (B, H', W', C) -> (B, C, H', W')
        x = x.permute(0, 3, 1, 2)
        
        # 5. Un-Padding: 切掉之前补的边
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W]
            
        return x

class SwinStage(nn.Module):
    """
    一个完整的 Swin Stage，包含两个 Block：
    1. Regular Window Attention (shift_size=0)
    2. Shifted Window Attention (shift_size > 0)
    这是 Swin 能够进行跨窗口信息交互的关键。
    """
    def __init__(self, dim, num_heads=8, window_size=8, pro=False):
        super().__init__()
        self.pro = pro
        # Block 1: 不移位
        self.block1 = SwinConvAdapter(dim, num_heads, window_size, shift_size=0)
        # Block 2: 移位 (通常是 window_size // 2)
        self.block2 = SwinConvAdapter(dim, num_heads, window_size, shift_size=window_size//2)
        
    def forward(self, x, *args):
        # 修改版本：添加残差连接
        if self.pro:
            shortcut = x
            return self.block2(self.block1(x)) + shortcut
        else:
            return self.block2(self.block1(x))
