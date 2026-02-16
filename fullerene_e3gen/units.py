"""Utilities for consistent coordinate units/normalization.

This project trains in *normalized* coordinate space.
Current dataset normalization (see dataset.py):
- center coordinates
- scale each molecule so mean atomic radius == 1.0

Therefore:
- Typical fullerene radius in training space is ~1.0
- Typical C–C bond length in training space is ~0.35–0.45 depending on C

These helpers provide a simple, consistent mapping between
physical Å space and the normalized training space.
"""

from __future__ import annotations

import torch


def estimate_fullerene_radius_angstrom(C: torch.Tensor | int, *, radius_coeff: float = 0.45) -> torch.Tensor:
    """Estimate fullerene radius in Å from carbon count.

    A simple empirical scaling: R(C) ≈ a * sqrt(C).
    Calibrated so C60 gives ~3.5 Å for a≈0.45.

    This is used only for *approximate* denormalization of generated samples
    and for converting physical bond targets to normalized-space targets.
    """

    if not torch.is_tensor(C):
        C = torch.tensor(C, dtype=torch.float32)
    else:
        C = C.to(dtype=torch.float32)

    return radius_coeff * torch.sqrt(C)


def bond_target_normalized(
    C: torch.Tensor | int,
    *,
    bond_length_angstrom: float = 1.42,
    radius_coeff: float = 0.45,
) -> torch.Tensor:
    """Convert a physical bond length (Å) to normalized-space target."""

    R = estimate_fullerene_radius_angstrom(C, radius_coeff=radius_coeff)
    return torch.as_tensor(bond_length_angstrom, dtype=torch.float32, device=R.device) / R


def bond_tolerance_normalized(
    C: torch.Tensor | int,
    *,
    tolerance_angstrom: float,
    radius_coeff: float = 0.45,
) -> torch.Tensor:
    """Convert a physical tolerance (Å) to normalized-space tolerance."""

    R = estimate_fullerene_radius_angstrom(C, radius_coeff=radius_coeff)
    return torch.as_tensor(tolerance_angstrom, dtype=torch.float32, device=R.device) / R


def denormalize_positions(
    pos_normalized: torch.Tensor,
    C: int,
    *,
    radius_coeff: float = 0.45,
) -> torch.Tensor:
    """Map normalized positions back to approximate Å space."""

    scale = estimate_fullerene_radius_angstrom(C, radius_coeff=radius_coeff).to(pos_normalized.device)
    return pos_normalized * scale
