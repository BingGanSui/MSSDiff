import torch
import torchvision.transforms.functional as FV
from torch import nn

from .diffusion import Diffusion
from .utils import ImageSizeMixin
from .rrdbnet import RRDBNet
from .unet import UNet


class MSSDiff(ImageSizeMixin, nn.Module):
    def __init__(self, options):
        super().__init__()

        self.in_channels = options.model.input_channels
        self.diffusion = Diffusion(
            UNet(
                self.in_channels,
                128,
                options.model.rrdb_channels,
                out_channels=2 * self.in_channels,
                restruct_channels=True
            )
        )

        rrdbnet = RRDBNet(options, rrdb_channels=options.model.rrdb_network_features)
        
        if hasattr(options.model, 'rrdb_pretrained_path') and options.model.rrdb_pretrained_path:
            t = torch.load(options.model.rrdb_pretrained_path, map_location=torch.device("cpu"))
        elif options.train.dataset in ['Asia']:
            t = torch.load(
                f"pretrained-rrdbnet-{options.train.dataset}.pt", map_location=torch.device("cpu")
            )
        else:
            raise ValueError("unknown dataset")
        rrdbnet.load_state_dict(t)
        self.lr_feats = rrdbnet.rrdb

        self.residual_scale = getattr(options.model, 'residual_scale', 5)

        self.phy_constraints = getattr(options.train, 'phy_constraints', None)

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def _calculate_loss(self, x, *, cond):
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = self.residual_scale

        x_target = (x - cond_scaled) * scale

        diff_loss, pred_residual = self.diffusion.normalize(x_target, cond=(lr_feats, cond_scaled, scale))
        
        phy_loss = 0.0
        if self.phy_constraints and getattr(self.phy_constraints, 'enable', False):
            pred_image = (pred_residual / scale) + cond_scaled
            w_nonneg = getattr(self.phy_constraints, 'weight_precip_nonneg', 0.0)
            if w_nonneg > 0:
                precip_pred = pred_image[:, 5:, :, :]
                loss_nonneg = torch.nn.functional.relu(-precip_pred).mean()
                phy_loss += w_nonneg * loss_nonneg

            w_smooth = getattr(self.phy_constraints, 'weight_temp_smooth', 0.0)
            if w_smooth > 0:
                temp_pred = pred_image[:, :5, :, :]
                h_diff = (temp_pred[:, :, 1:, :] - temp_pred[:, :, :-1, :]).abs().mean()
                w_diff = (temp_pred[:, :, :, 1:] - temp_pred[:, :, :, :-1]).abs().mean()
                loss_smooth = h_diff + w_diff
                phy_loss += w_smooth * loss_smooth
                
        return diff_loss + phy_loss

    def _generate_sample(self, x, *, cond):
        x = x.view(x.shape[0], self.in_channels, self.image_size_x, self.image_size_y)
        if self.diffusion.ddim:
            import torch.utils.checkpoint

            lr_feats = torch.utils.checkpoint.checkpoint(self.lr_feats, cond)
        else:
            lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(
            cond, (x.shape[2], x.shape[3]), interpolation=FV.InterpolationMode.BICUBIC
        )
        scale = self.residual_scale
        x = self.diffusion.generate(x, cond=(lr_feats, cond_scaled, scale))
        x = x / scale
        x = x + cond_scaled
        return x

    def set_generate_steps(self, steps):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_steps(steps)

        self.apply(visit)

    def set_generate_verbose(self, verbose):
        def visit(module):
            if isinstance(module, Diffusion):
                module.set_generate_verbose(verbose)

        self.apply(visit)
