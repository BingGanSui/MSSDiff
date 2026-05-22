import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def predict_eps_x0(self, x, t, cond):
        # t : integer index tensor or shape [B] or scalar expanded as needed
        out = self.network(x, t, cond)
        c = out.shape[1] // 2
        eps_pred, x0_pred = out.split(c, dim=1)
        return eps_pred, x0_pred

    def combine_to_v(self, eps_pred, x0_pred, alpha_bar):
        # alpha_bar: scalar or tensor with batch dimension
        a_sqrt = alpha_bar.sqrt()
        one_minus_sqrt = (1 - alpha_bar).sqrt()
        # ensure broadcasting to [B,C,H,W]
        a_sqrt = a_sqrt[:, None, None, None] if a_sqrt.dim() == 1 else a_sqrt
        one_minus_sqrt = one_minus_sqrt[:, None, None, None] if one_minus_sqrt.dim() == 1 else one_minus_sqrt
        return eps_pred * a_sqrt - x0_pred * one_minus_sqrt

    def reconstruct_eps_from_v(self, x_t, v, alpha_bar):
        # eps_pred = sqrt(1-alpha_bar) * x_t + sqrt(alpha_bar) * v
        a_sqrt = alpha_bar.sqrt()
        one_minus_sqrt = (1 - alpha_bar).sqrt()
        a_sqrt = a_sqrt[:, None, None, None] if a_sqrt.dim() == 1 else a_sqrt
        one_minus_sqrt = one_minus_sqrt[:, None, None, None] if one_minus_sqrt.dim() == 1 else one_minus_sqrt
        return x_t * one_minus_sqrt + v * a_sqrt