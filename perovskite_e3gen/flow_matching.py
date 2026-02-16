"""Flow Matching scheduler for perovskite crystal generation.

Implements flow matching on three coupled spaces:
  1. Fractional coordinates: flow on the 3D torus T^3 = [0, 1)^3
  2. Lattice parameters: standard Euclidean flow (after z-score normalization)
  3. Atom types: continuous relaxation with cross-entropy loss

Coordinates use geodesic interpolation on the torus to respect periodicity.
Lattice parameters use standard linear interpolation.
Atom types use one-hot → uniform noise continuous relaxation.

Training:
  t ~ U(0, 1)
  F_t = (1 - t) * F_0 + t * noise    (with torus wrapping)
  L_t = (1 - t) * L_0 + t * L_noise
  A_t = (1 - t) * one_hot(A_0) + t * uniform
  target_v_F = noise - F_0            (wrapped to [-0.5, 0.5))
  target_v_L = L_noise - L_0
  loss = MSE(v_F_pred, target_v_F) + MSE(v_L_pred, target_v_L) + CE(type_logits, A_0)

Sampling (ODE integration from t=1 -> t=0):
  F_{t-dt} = F_t - dt * v_F_pred      (wrapped to [0, 1))
  L_{t-dt} = L_t - dt * v_L_pred

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from units import lattice_params_to_matrix, wrap_frac_coords

logger = logging.getLogger(__name__)


def wrap_diff(diff: torch.Tensor) -> torch.Tensor:
    """Wrap differences to [-0.5, 0.5) for torus geodesic."""
    return diff - torch.round(diff)


class CrystalFlowMatcher:
    """Flow matching scheduler for crystal structures.

    Handles the three coupled denoising processes:
    - Fractional coordinates (torus)
    - Lattice parameters (Euclidean)
    - Atom types (categorical via continuous relaxation)
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        coord_noise_type: str = "wrapped_normal",
        lattice_noise_scale: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.coord_noise_type = coord_noise_type
        self.lattice_noise_scale = lattice_noise_scale

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample training times uniformly from U(0, 1)."""
        return torch.rand(batch_size, device=device).clamp(
            min=self.sigma_min, max=1.0 - self.sigma_min
        )

    # ---- Coordinate flow (torus) ----

    def add_coord_noise(
        self,
        frac_coords: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process for fractional coordinates on torus.

        Uses geodesic interpolation: shortest path on torus.

        Args:
            frac_coords: [N, 3] clean fractional coordinates in [0, 1).
            t: [N] or [N, 1] per-node time values.
            noise: [N, 3] noise sample (generated if None).

        Returns:
            frac_t: [N, 3] noisy fractional coordinates (wrapped to [0, 1)).
            target_v: [N, 3] target velocity field (wrapped to [-0.5, 0.5)).
        """
        if noise is None:
            noise = torch.randn_like(frac_coords)
            if self.coord_noise_type == "wrapped_normal":
                noise = noise % 1.0  # Wrap to [0, 1) for torus

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Geodesic interpolation on torus
        # Target velocity: shortest path from data to noise
        target_v = wrap_diff(noise - frac_coords)  # [-0.5, 0.5)

        # Interpolate along geodesic
        frac_t = frac_coords + t * target_v
        frac_t = wrap_frac_coords(frac_t)  # Wrap to [0, 1)

        return frac_t, target_v

    # ---- Lattice flow (Euclidean) ----

    def add_lattice_noise(
        self,
        lattice_params: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process for lattice parameters (standard Euclidean).

        Args:
            lattice_params: [B, 6] normalized lattice parameters.
            t: [B] or [B, 1] per-batch time values.
            noise: [B, 6] noise sample.

        Returns:
            lattice_t: [B, 6] noisy lattice parameters.
            target_v: [B, 6] target velocity field.
        """
        if noise is None:
            noise = torch.randn_like(lattice_params) * self.lattice_noise_scale

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        lattice_t = (1.0 - t) * lattice_params + t * noise
        target_v = noise - lattice_params

        return lattice_t, target_v

    # ---- Atom type flow (continuous relaxation) ----

    def add_type_noise(
        self,
        atom_types: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Forward process for atom types via continuous relaxation.

        Interpolates between one-hot encoding and uniform distribution.

        Args:
            atom_types: [N] integer atom type indices.
            t: [N] or [N, 1] per-node time values.
            num_classes: Number of element classes.

        Returns:
            type_t: [N, num_classes] noisy continuous type distribution.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        one_hot = F.one_hot(atom_types, num_classes=num_classes).float()  # [N, C]
        uniform = torch.ones_like(one_hot) / num_classes

        type_t = (1.0 - t) * one_hot + t * uniform
        return type_t

    # ---- Sampling steps ----

    def coord_step(self, frac_t: torch.Tensor, v_pred: torch.Tensor, dt: float) -> torch.Tensor:
        """Euler step for fractional coordinates on torus."""
        frac_next = frac_t - dt * v_pred
        return wrap_frac_coords(frac_next)

    def lattice_step(self, lattice_t: torch.Tensor, v_pred: torch.Tensor, dt: float) -> torch.Tensor:
        """Euler step for lattice parameters."""
        return lattice_t - dt * v_pred

    # ---- Full sampling loop ----

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        num_atoms: int,
        atom_types: torch.Tensor,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full ODE integration from t=1 (noise) to t=0 (data).

        Args:
            model: PerovskitePaiNNModel.
            num_atoms: Number of atoms in the structure.
            atom_types: [N] atom type indices for the target composition.
            num_steps: Number of Euler steps.
            device: Target device.

        Returns:
            frac_coords: [N, 3] generated fractional coordinates.
            lattice_params: [6] generated lattice parameters (normalized).
            type_probs: [N, num_elements] atom type probabilities.
        """
        if device is None:
            device = next(model.parameters()).device

        batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
        atom_types = atom_types.to(device)

        # Start from noise
        frac_t = torch.rand(num_atoms, 3, device=device)  # Uniform on torus
        lattice_t = torch.randn(1, 6, device=device) * self.lattice_noise_scale
        type_probs = torch.ones(num_atoms, model.num_elements, device=device) / model.num_elements

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = 1.0 - step * dt
            t_batch = torch.full((1,), t_val, device=device)

            # Build lattice matrix from current lattice params for graph construction
            # Use placeholder lattice for edge computation
            from dataset import build_radius_graph_pbc
            lengths_angles = lattice_t.squeeze(0)  # [6]

            # For graph building, we need a reasonable lattice
            # During sampling we use the denormalized lattice
            lattice_matrix = lattice_params_to_matrix(
                lengths_angles[:3].unsqueeze(0),
                lengths_angles[3:].unsqueeze(0),
            )  # [1, 3, 3]

            # Build radius graph
            edge_index, edge_shift = build_radius_graph_pbc(
                frac_t, lattice_matrix.squeeze(0), cutoff=model.cutoff
            )

            # Forward pass
            coord_vel, lattice_vel, type_logits = model(
                frac_t, atom_types, lattice_matrix, t_batch,
                edge_index, edge_shift, batch,
            )

            # Euler steps
            frac_t = self.coord_step(frac_t, coord_vel, dt)
            lattice_t = self.lattice_step(lattice_t, lattice_vel, dt)

            # Update type probabilities
            type_probs = F.softmax(type_logits, dim=-1)

            if torch.isnan(frac_t).any() or torch.isnan(lattice_t).any():
                logger.warning("NaN at step %d/%d — aborting", step, num_steps)
                break

        return frac_t, lattice_t.squeeze(0), type_probs


if __name__ == '__main__':
    """Quick test of crystal flow matching."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fm = CrystalFlowMatcher()

    # Test coordinate flow on torus
    frac = torch.rand(10, 3)
    t = torch.full((10,), 0.5)
    frac_t, target_v = fm.add_coord_noise(frac, t)
    logger.info("Coord noise: frac_t in [%.3f, %.3f], target_v in [%.3f, %.3f]",
                frac_t.min(), frac_t.max(), target_v.min(), target_v.max())
    assert (frac_t >= 0).all() and (frac_t < 1).all(), "Coords not in [0, 1)"
    assert (target_v >= -0.5).all() and (target_v <= 0.5).all(), "Velocity not in [-0.5, 0.5)"

    # Test lattice flow
    lattice = torch.randn(2, 6)
    t_batch = torch.tensor([0.3, 0.7])
    lattice_t, target_v_l = fm.add_lattice_noise(lattice, t_batch)
    logger.info("Lattice noise: shape %s", lattice_t.shape)

    # Test type noise
    types = torch.tensor([1, 2, 3, 1, 2])
    type_t = fm.add_type_noise(types, torch.full((5,), 0.5), num_classes=10)
    logger.info("Type noise: shape %s, sum per row: %s", type_t.shape, type_t.sum(dim=-1))

    logger.info("All flow matching tests passed!")
