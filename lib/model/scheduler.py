import math
import torch
from torch import nn

class NoiseScheduler(nn.Module):
    def __init__(self, t_max=300, b_min=0.1, b_max=20):
        super().__init__()
        self.t_max = t_max
        self.b_min = b_min
        self.b_max = b_max

        ts = torch.arange(0, t_max + 1) / t_max
        alpha_bars_ext = self.t_to_aa(ts)
        alphas = alpha_bars_ext[1:] / alpha_bars_ext[:-1]
        betas = 1 - alphas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars_ext[1:])

    def t_to_aa(self, ts):
        return torch.exp(-ts * (ts * (self.b_max - self.b_min) / 2 + self.b_min))

    def alpha_bar_at_index(self, idx):
        return self.alpha_bars[idx]

    def alpha_bar_cont(self, t_cont):
        # t_cont: can be tensor/float in range [0, len(alpha_bars)]
        return self.t_to_aa(t_cont / len(self.alpha_bars))

    def beta_cont(self, t_cont):
        return self.b_min + (self.b_max - self.b_min) * (t_cont / len(self.alpha_bars))

    def get_inject_noise_var(self, t):
        if t == 0:
            return 0.0
        beta_tilde = (
            (1 - self.alpha_bars[t] / self.alphas[t])
            / (1 - self.alpha_bars[t])
            * self.betas[t]
        )
        return math.exp(math.log(beta_tilde) * 0.0 + math.log(self.betas[t]) * 1.0)