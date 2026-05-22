import torch
import torch.utils.checkpoint
from torch import nn
from torchdiffeq import odeint

class Sampler(nn.Module):
    def __init__(self, predictor: nn.Module, scheduler: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.scheduler = scheduler
        self.gen_steps = 0
        self.gen_verbose = False
        self.ddim = False
        self.use_ode = False

    def set_generate_steps(self, steps):
        self.gen_steps = steps

    def set_generate_verbose(self, verbose):
        self.gen_verbose = verbose

    def ode_func(self, net_in, t, cond):
        if self.gen_verbose:
            print("eval at", t)
        sampled_aa = self.scheduler.alpha_bar_cont(t)
        sampled_beta = self.scheduler.beta_cont(t)
        # network expects time as 1-dim tensor
        network_eps, network_x0 = self.predictor.predict_eps_x0(net_in, t[None], cond)
        network_v = self.predictor.combine_to_v(network_eps, network_x0, sampled_aa)
        eps_pred = self.predictor.reconstruct_eps_from_v(net_in, network_v, sampled_aa)
        value = -0.5 * sampled_beta * net_in + 0.5 * sampled_beta * eps_pred / (1 - sampled_aa).sqrt()
        return value / len(self.scheduler.alpha_bars)

    def generate(self, x, *, cond):
        cond, diff_mean, diff_scale = cond

        if self.use_ode:
            x = odeint(
                lambda t, net_in: self.ode_func(net_in, t, cond),
                x,
                torch.tensor([len(self.scheduler.alpha_bars), 0.1], device=x.device),
                atol=1e-5,
                rtol=1e-5,
                options={"jump_t": torch.tensor([0.1], device=x.device)},
            )[1]
            return x

        end_step = 0 if self.ddim else 5
        for t in range(self.gen_steps - 1, end_step - 1, -1):
            def next_step(x_local, cond_local, tt, the_eps):
                idx = tt
                alpha_bar = self.scheduler.alpha_bar_at_index(idx)
                network_eps, network_x0 = self.predictor.predict_eps_x0(
                    x_local, torch.tensor([idx + 1], device=x.device).expand(x_local.shape[0]), cond_local
                )
                v = self.predictor.combine_to_v(network_eps, network_x0, alpha_bar)
                eps_pred = self.predictor.reconstruct_eps_from_v(x_local, v, alpha_bar)

                if self.ddim:
                    eps_coeff = (1 - self.scheduler.alphas[idx]) / (
                        (1 - alpha_bar).sqrt() + (self.scheduler.alphas[idx] - alpha_bar).sqrt()
                    )
                    x_next = (x_local - eps_coeff * eps_pred) / self.scheduler.alphas[idx].sqrt()
                else:
                    if tt == end_step:
                        x_next = (x_local - (1 - alpha_bar).sqrt() * eps_pred) / self.scheduler.alphas[idx].sqrt()
                    else:
                        eps_coeff = (1 - self.scheduler.alphas[idx]) / (1 - alpha_bar).sqrt()
                        x_next = (x_local - eps_coeff * eps_pred) / self.scheduler.alphas[idx].sqrt()
                        x_next = x_next + the_eps * (self.scheduler.get_inject_noise_var(idx) ** 0.5)
                return x_next

            if t == self.gen_steps - 1:
                x = next_step(x, cond, t, torch.randn_like(x))
            else:
                x = torch.utils.checkpoint.checkpoint(next_step, x, cond, t, torch.randn_like(x))
            if self.gen_verbose:
                print(t, x.mean().item(), x.std().item())
        return x
