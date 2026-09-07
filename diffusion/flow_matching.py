"""Quality-Adaptive Flow Matching scheduler for hand motion generation."""

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingScheduler:
    """Quality-Adaptive Flow Matching with per-frame noise control."""

    def __init__(self, sigma_min: float = 1e-4, pred_type: str = "v"):
        """Create the velocity-prediction scheduler used by the released models."""
        if pred_type != "v":
            raise ValueError("Released motion priors require pred_type='v'")
        self.sigma_min = sigma_min
        self.pred_type = pred_type

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        model_kwargs: dict = None,
        num_steps: int = 50,
        device: torch.device = None,
        x_init: Tensor = None,
        t_start: float = 0.0,
        quality_per_hand: Tensor = None,
    ) -> Tensor:
        """Generate samples by solving the ODE from t=t_start to t=1."""
        if model_kwargs is None:
            model_kwargs = {}

        if x_init is None:
            x = torch.randn(shape, device=device)
        else:
            x = x_init.to(device)

        dt = (1.0 - t_start) / num_steps
        timesteps = torch.linspace(t_start, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i]
            t_batch = t.expand(shape[0])

            eff_t = None
            if quality_per_hand is not None:
                eff_t_left = 1.0 - (1.0 - t) * (1.0 - quality_per_hand[:, 0, :])
                eff_t_right = 1.0 - (1.0 - t) * (1.0 - quality_per_hand[:, 1, :])
                eff_t = torch.stack([eff_t_left, eff_t_right], dim=1)

            model_out = model(
                hidden_states=x,
                timestep=t_batch,
                effective_t=eff_t,
                **model_kwargs,
            )

            x = x + dt * model_out

        return x

    @torch.no_grad()
    def sample_with_quality_schedule(
        self,
        model: nn.Module,
        shape: tuple,
        quality_left: Tensor,
        quality_right: Tensor,
        x_proposal: Tensor,
        model_kwargs: dict = None,
        num_steps: int = 50,
        device: torch.device = None,
        q_per_dim: Tensor = None,
    ) -> Tensor:
        """Quality-adaptive sampling with per-hand (default) or per-dimension quality."""
        if model_kwargs is None:
            model_kwargs = {}

        B, C, T = shape
        hand_ch = C // 2
        noise = torch.randn(shape, device=device)

        if q_per_dim is not None:
            q_init = q_per_dim
            t_start = float(q_per_dim.min().item())
        else:
            q_l = quality_left.unsqueeze(1).expand(-1, hand_ch, -1)
            q_r = quality_right.unsqueeze(1).expand(-1, hand_ch, -1)
            q_init = torch.cat([q_l, q_r], dim=1)
            t_start = min(quality_left.min().item(), quality_right.min().item())

        x = q_init * x_proposal + (1.0 - q_init) * noise

        t_start = max(t_start, 0.0)
        t_start = min(t_start, 0.95)

        q_per_hand = torch.stack([quality_left, quality_right], dim=1)

        return self.sample(
            model=model,
            shape=shape,
            model_kwargs=model_kwargs,
            num_steps=num_steps,
            device=device,
            x_init=x,
            t_start=t_start,
            quality_per_hand=q_per_hand,
        )
