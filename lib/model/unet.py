import math
from sympy import E
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from torch import nn
from .wlum import WaveletUpsample
from .Attn import SwinStage

# ACT = functools.partial(F.leaky_relu, inplace=True)
ACT = F.silu


SWIN_ATTN = True
SWIN_ATTN_PRO = False


class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class MyConv2d(nn.Module):
    def __init__(self, *args, init_scale=1, **kwargs):
        super().__init__()
        self.network = nn.Conv2d(*args, **kwargs)

        with torch.no_grad():
            self.network.weight.mul_(init_scale)
            if self.network.bias is not None:
                self.network.bias.zero_()

    def forward(self, x):
        return self.network(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, extra_channels):
        super().__init__()

        assert in_channels % 2 == 0 and out_channels % 2 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_in = in_channels // 2
        self.half_out = out_channels // 2
        self.time_in = nn.Linear(time_channels, out_channels)
        self.extra_in = MyConv2d(extra_channels, out_channels, 1)
        g1 = min(in_channels // 4, 32)
        self.norm1 = nn.GroupNorm(max(g1 // 2, 1), self.half_in)
        self.conv1 = MyConv2d(in_channels, out_channels, 3, padding=1)
        g2 = min(out_channels // 4, 32)
        self.norm2 = nn.GroupNorm(max(g2 // 2, 1), self.half_out)
        self.conv2 = MyConv2d(out_channels, out_channels, 3, padding=1, init_scale=1e-3)
        self.dropout = nn.Identity()

        if in_channels != out_channels:
            self.skip = MyConv2d(in_channels, out_channels, 1, init_scale=0.1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_embed, extra_feats):
        t_embed = self.time_in(t_embed)
        if extra_feats is not None:
            extra_feats = FV.resize(
                extra_feats,
                (x.shape[2], x.shape[3]),
                interpolation=FV.InterpolationMode.BILINEAR,
            )
            extra_feats = self.extra_in(extra_feats)

        x_a, x_b = torch.chunk(x, 2, dim=1)

        y_a = self.norm1(x_a)
        y_b = self.norm1(x_b)
        y_a = ACT(y_a)
        y_b = ACT(y_b)
        y = torch.cat([y_a, y_b], dim=1)

        y = self.conv1(y)

        y_a, y_b = torch.chunk(y, 2, dim=1)
        t_a, t_b = torch.chunk(t_embed, 2, dim=1)
        y_a = y_a + t_a[:, :, None, None]
        y_b = y_b + t_b[:, :, None, None]

        if extra_feats is not None:
            e_a, e_b = torch.chunk(extra_feats, 2, dim=1)
            y_a = y_a + e_a
            y_b = y_b + e_b

        y_a = self.norm2(y_a)
        y_b = self.norm2(y_b)
        y_a = ACT(y_a)
        y_b = ACT(y_b)
        y_a = self.dropout(y_a)
        y_b = self.dropout(y_b)

        y = torch.cat([y_a, y_b], dim=1)
        y = self.conv2(y)

        x = self.skip(x)
        return x + y


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downconv = MyConv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, t_embed, extra_feats):
        x = self.downconv(x)
        return x


def concat(x, prev_x):
    assert 0 <= x.shape[2] - prev_x.shape[2] <= 1
    assert 0 <= x.shape[3] - prev_x.shape[3] <= 1
    x = x[:, :, : prev_x.shape[2], : prev_x.shape[3]]
    x = torch.cat([x, prev_x], dim=1)
    return x


def make_embed(t, channels):
    assert channels % 2 == 0
    half_dim = channels // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = (torch.arange(half_dim, device=t.device) * -emb).exp()
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    return emb


class UNet(nn.Module):
    def __init__(self, num_channels, time_channels, extra_channels, out_channels=None, restruct_channels=False):
        super().__init__()

        if out_channels is None:
            out_channels = num_channels
        self.time_channels = time_channels
        self.time_in = nn.Sequential(
            nn.Linear(time_channels, time_channels * 4),
            LambdaModule(ACT),
            nn.Linear(time_channels * 4, time_channels * 4),
            LambdaModule(ACT),
        )
        time_channels = time_channels * 4

        self.adaptor = nn.Conv2d(num_channels, 64, 3,1,1)
        self.in_conv = MyConv2d(64, 64, 3, padding=1)
        prev_channel = 64

        channels = [64, 128, 256, 256]
        num_blocks = [2, 2, 2, 2]

        self.downs = nn.ModuleList()
        self.downs_phi = nn.ModuleList()
        feat_channels = [prev_channel]
        for i in range(len(channels)):
            for j in range(num_blocks[i]):
                self.downs.append(
                    ResnetBlock(
                        prev_channel, channels[i], time_channels, extra_channels
                    )
                )
                feat_channels.append(channels[i])
                prev_channel = channels[i]
            if i < len(channels) - 1:
                self.downs.append(Down(prev_channel, prev_channel))
                feat_channels.append(prev_channel)

        global SWIN_ATTN
        global SWIN_ATTN_PRO
        if SWIN_ATTN:
            if SWIN_ATTN_PRO:
                self.mid = nn.ModuleList(
                    [
                        SwinStage(prev_channel, 4, 8, pro=True),
                        SwinStage(prev_channel, 4, 8, pro=True),
                        SwinStage(prev_channel, 4, 8, pro=True),
                    ]
                )
            else:
                self.mid = nn.ModuleList(
                    [
                        SwinStage(prev_channel, 4, 8),
                        SwinStage(prev_channel, 4, 8),
                    ]
                )
        else:
            self.mid = nn.ModuleList(
                [
                    ResnetBlock(prev_channel, prev_channel, time_channels, extra_channels),
                    ResnetBlock(prev_channel, prev_channel, time_channels, extra_channels),
                ]
            )

        self.ups = nn.ModuleList()
        self.ups_phi = nn.ModuleList()
        self.up_need_res = []
        for i in range(len(channels) - 1, -1, -1):
            for j in range(num_blocks[i] + 1):
                res_channel = feat_channels.pop()
                self.ups.append(
                    ResnetBlock(
                        prev_channel + res_channel,
                        channels[i],
                        time_channels,
                        extra_channels,
                    )
                )
                self.up_need_res.append(True)
                prev_channel = channels[i]
            if i > 0:
                self.ups.append(WaveletUpsample(prev_channel, prev_channel))
                self.up_need_res.append(False)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(16, prev_channel),
            LambdaModule(ACT),
            MyConv2d(prev_channel, out_channels, 3, padding=1),
        )

        self.restruct_channels = restruct_channels

    def forward(self, x, t, extra_feats):
        t_embed = make_embed(t, self.time_channels)
        t_embed = self.time_in(t_embed)


        x = self.adaptor(x)
        x = self.in_conv(x)
        feats = [x]

        for layer in self.downs:
            x = layer(x, t_embed, extra_feats)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t_embed, extra_feats)

        for layer, need_res in zip(self.ups, self.up_need_res):
            if need_res:
                x = concat(x, feats.pop())
            x = layer(x, t_embed, extra_feats)

        assert not feats

        x = self.out_conv(x)
        if self.restruct_channels:
            C = x.shape[1]
            eps_a = x[:,:C//4,:,:]
            x0_a = x[:,C//4:C//4*2,:,:]
            eps_b = x[:,C//4*2:C//4*3,:,:]
            x0_b = x[:,C//4*3:,:,:]
            x = torch.cat([eps_a, eps_b, x0_a, x0_b], dim=1)
        return x
