import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT_IDWT_Layer(nn.Module):
    """
    Haar小波变换与逆变换的基础层
    无需引入外部库，基于卷积实现，完全可导
    """
    def __init__(self):
        super(DWT_IDWT_Layer, self).__init__()
        # 定义Haar小波核
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # 组合成卷积核 (4, 1, 2, 2)
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernels', kernels)

    def dwt(self, x):
        # x: [B, C, H, W] -> [B, C*4, H/2, W/2]
        B, C, H, W = x.shape
        # 对每个通道独立进行小波分解
        weight = self.kernels.repeat(C, 1, 1, 1)
        # stride=2 实现下采样
        y = F.conv2d(x, weight, padding=0, stride=2, groups=C)
        # 重排通道，使得输出格式方便后续处理
        # y shape: [B, 4*C, H/2, W/2] -> Reshape -> [B, C, 4, H/2, W/2]
        return y.view(B, C, 4, H // 2, W // 2)

    def idwt(self, x):
        # x: [B, C, 4, H, W] -> [B, C, 2H, 2W]
        B, C, _, H, W = x.shape
        # 将4个频带展平: [B, C*4, H, W]
        x = x.view(B, C * 4, H, W)
        
        # 逆变换的卷积核（转置卷积）
        weight = self.kernels.repeat(C, 1, 1, 1)
        
        # stride=2 实现上采样
        y = F.conv_transpose2d(x, weight, stride=2, groups=C)
        return y

class WaveletUpsample(nn.Module):
    """
    KDD 核心模块：替代原本的 Up + FNO
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt_layer = DWT_IDWT_Layer()
        
        # 这里的输入是低分变率特征图
        # 我们需要预测出 4 个子带 (LL, LH, HL, HH)
        # 所以输出通道是 out_channels * 4
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels * 2, out_channels * 4, 3, 1, 1) # 预测4个频带系数
        )
        
        # 可选：专门对高频子带进行额外增强的残差层
        self.high_freq_enhancer = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels * 3, 3, 1, 1, groups=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels * 3, out_channels * 3, 1) 
        )

    def forward(self, x, t_embed=None, extra_feats=None):
        # 1. 先进行一次最邻近上采样作为 Base (提供低频基础)
        # x_base = F.interpolate(x, scale_factor=2, mode='nearest') 
        
        # 2. 预测小波系数 [B, Out*4, H, W]
        coeffs = self.predictor(x)
        B, _, H, W = coeffs.shape
        C_out = coeffs.shape[1] // 4
        
        # Reshape 为 [B, Out, 4, H, W]
        # 0:LL, 1:LH, 2:HL, 3:HH
        coeffs = coeffs.view(B, C_out, 4, H, W)
        
        # 3. 对高频部分 (LH, HL, HH) 进行特殊的增强 (模拟降水的高频突变)
        # 这是一个创新点：Explicit High-Frequency Refinement
        ll = coeffs[:, :, 0:1, :, :]
        high = coeffs[:, :, 1:, :, :] # [B, Out, 3, H, W]
        
        high_flat = high.reshape(B, C_out * 3, H, W)
        high_refined = self.high_freq_enhancer(high_flat).reshape(B, C_out, 3, H, W)
        
        # 重新组合
        coeffs_refined = torch.cat([ll, high_refined], dim=2)
        
        # 4. 逆小波变换重构
        out = self.dwt_layer.idwt(coeffs_refined)
        
        return out