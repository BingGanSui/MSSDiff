import piq
import torch


def ssim(x, y):
    # piq.ssim() with downsample=False is equivalent to skimage's ssim
    # with gaussian_weights=True and use_sample_covariance=False, i.e.
    # the same calculation as the original paper.
    if x.isnan().any() or y.isnan().any():
        return torch.tensor(float("nan"), device=x.device)
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    return piq.ssim(x, y, data_range=1.0, downsample=False, reduction="none")
