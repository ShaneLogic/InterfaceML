"""
Conditional Flow Matching (CFM) scheduler for fullerene diffusion.

Replaces DDPM epsilon prediction with optimal-transport conditional flow
matching (Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023).

Key advantages over DDPM:
  1. Simpler objective: MSE(v_pred, x_0 - noise) — no beta schedules
  2. Straighter trajectories: fewer sampling steps needed (50 vs 200+)
  3. Better mode coverage: uniform t sampling by design
  4. No variance explosion: linear interpolation path is well-conditioned

Training:
  t ~ U(0, 1)
  x_t = (1 - t) * x_0 + t * noise      (linear interpolation)
  target_v = noise - x_0                  (velocity field: x_0 -> noise)
  loss = MSE(model(x_t, t), target_v)

Sampling (ODE integration from t=1 -> t=0):
  x_{t-dt} = x_t - dt * v_pred(x_t, t)  (Euler step)

Note on conventions:
  t=0 corresponds to clean data x_0
  t=1 corresponds to pure noise
  We integrate backwards from t=1 to t=0 during generation.

Author: InterfaceML Project
Date: 2026-02
"""

import logging
from typing import Optional, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FlowMatchingScheduler:
    """Conditional Flow Matching scheduler.

    Implements the linear interpolation path:
        x_t = (1 - t) * x_0 + t * noise
    with velocity field target:
        v = noise - x_0
    """

    def __init__(self, sigma_min: float = 1e-4):
        """
        Args:
            sigma_min: Small constant for numerical stability at t=0.
        """
        self.sigma_min = sigma_min

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample training times uniformly from U(0, 1).

        Args:
            batch_size: Number of samples.
            device: Target device.

        Returns:
            t: [B] times in (0, 1).
        """
        # Avoid exact 0 and 1 for numerical stability
        return torch.rand(batch_size, device=device).clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward process: interpolate between data and noise.

        Args:
            x_0: [N, 3] clean coordinates.
            t: [N] or [N, 1] per-node time values.
            noise: [N, 3] noise sample (generated if None).

        Returns:
            x_t: [N, 3] noisy coordinates at time t.
            target_v: [N, 3] target velocity field (noise - x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]

        x_t = (1.0 - t) * x_0 + t * noise
        target_v = noise - x_0

        return x_t, target_v

    def sample_step(
        self,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Single Euler ODE step: x_{t-dt} = x_t - dt * v_pred.

        We integrate backwards (t=1 -> t=0), so we subtract.

        Args:
            x_t: [N, 3] current positions at time t.
            v_pred: [N, 3] predicted velocity at time t.
            dt: Step size (positive).

        Returns:
            x_next: [N, 3] positions at time t - dt.
        """
        return x_t - dt * v_pred

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        edge_index: torch.Tensor,
        C_cond: torch.Tensor,
        batch: torch.Tensor,
        num_steps: int = 50,
        cfg_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Full ODE integration from t=1 (noise) to t=0 (data).

        Uses Euler integration with uniform step sizes.

        Args:
            model: Diffusion model with forward(pos, edge_index, t, C, batch).
                   Must also have forward_cfg() if cfg_weight > 1.
            shape: (N, 3) shape of output.
            edge_index: [2, E] graph connectivity.
            C_cond: [B] carbon count conditioning.
            batch: [N] batch assignment.
            num_steps: Number of Euler steps.
            cfg_weight: CFG guidance weight (0 or 1 = no guidance, >1 = guided).
            device: Target device.

        Returns:
            x_0: [N, 3] generated clean coordinates.
        """
        if device is None:
            device = edge_index.device

        # Start from pure noise at t=1
        x_t = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        use_cfg = cfg_weight > 1.0 and hasattr(model, 'forward_cfg')

        for step in range(num_steps):
            t_val = 1.0 - step * dt
            # Create time tensor for model (convert to integer scale for compatibility)
            t_batch = torch.full(
                (C_cond.size(0),), t_val * 999, device=device, dtype=torch.long
            )

            if use_cfg:
                v_pred = model.forward_cfg(x_t, edge_index, t_batch, C_cond, batch, cfg_weight=cfg_weight)
            else:
                v_pred = model(x_t, edge_index, t_batch, C_cond, batch)

            # Euler step
            x_t = self.sample_step(x_t, v_pred, dt)

            # CoM projection at each step for translation invariance
            batch_size = batch.max().item() + 1
            # scatter mean via index_add + counts (no torch_scatter dependency)
            com = torch.zeros(batch_size, x_t.size(1), device=device, dtype=x_t.dtype)
            counts = torch.zeros(batch_size, 1, device=device, dtype=x_t.dtype)
            com.index_add_(0, batch, x_t)
            counts.index_add_(0, batch, torch.ones(batch.size(0), 1, device=device, dtype=x_t.dtype))
            com = com / counts.clamp(min=1)
            x_t = x_t - com[batch]

            # NaN guard
            if torch.isnan(x_t).any():
                logger.warning("NaN at step %d/%d — aborting", step, num_steps)
                break

        return x_t


if __name__ == '__main__':
    """Quick test of flow matching scheduler."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    scheduler = FlowMatchingScheduler()

    # Test add_noise
    x_0 = torch.randn(60, 3)
    t = torch.full((60,), 0.5)
    noise = torch.randn_like(x_0)
    x_t, target_v = scheduler.add_noise(x_0, t, noise)

    logger.info("x_0 mean: %.4f", x_0.mean())
    logger.info("x_t mean: %.4f (should be avg of x_0 and noise)", x_t.mean())
    logger.info("target_v = noise - x_0: error = %.6f",
                (target_v - (noise - x_0)).norm().item())

    # Test sample_step
    v_pred = target_v  # perfect prediction
    x_next = scheduler.sample_step(x_t, v_pred, dt=0.1)
    logger.info("Euler step: moved %.4f units", (x_next - x_t).norm().item())

    # Test t sampling
    t_samples = scheduler.sample_t(10000, device=torch.device('cpu'))
    logger.info("t distribution: mean=%.3f, std=%.3f (should be ~0.5, ~0.29)",
                t_samples.mean(), t_samples.std())

    logger.info("All flow matching tests passed!")
