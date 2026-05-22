import numpy as np
import torch.nn.functional as F
from PIL import Image
import torch


def quantize(x):
    return (x.clamp(0, 1) * 255).round() / 255


def pad_to_multiple(tensor, k):
    size_x = tensor.shape[2]
    size_y = tensor.shape[3]
    new_x = (size_x + k - 1) // k * k
    new_y = (size_y + k - 1) // k * k
    tensor = F.pad(tensor, (0, new_y - size_y, 0, new_x - size_x), mode="replicate")
    return tensor


def save_result(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.detach().clone().cpu(), path)
