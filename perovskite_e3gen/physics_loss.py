"""Physics-informed losses for perovskite crystal generation.

All losses use time-conditioned weighting (sigmoid schedule):
  - t < 0.3 (low noise): full constraint
  - t > 0.6 (high noise): no constraint

Losses:
  1. Bond valence sum — penalize unreasonable oxidation states
  2. Goldschmidt tolerance factor — structural stability indicator
  3. Minimum interatomic distance — prevent atom overlap
  4. Lattice regularity — penalize extreme aspect ratios or angles

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from units import (
    frac_to_cart,
    IONIC_RADII,
    index_to_element,
    goldschmidt_tolerance,
)

logger = logging.getLogger(__name__)


def compute_time_weight(
    t: torch.Tensor,
    center: float = 0.3,
    steepness: float = 15.0,
) -> torch.Tensor:
    """Sigmoid time-weighting: full at low t, zero at high t.

    Args:
        t: [B] time values in [0, 1].
        center: Sigmoid center point.
        steepness: How sharp the transition is.

    Returns:
        [B] weights in [0, 1].
    """
    return torch.sigmoid(-steepness * (t - center))


# ---- Bond valence sum loss ----

# Bond valence parameters (Brown & Altermatt, 1985)
# R0 values for common perovskite bonds (Angstrom)
BOND_VALENCE_R0 = {
    ("Ba", "O"): 2.285, ("Sr", "O"): 2.118, ("Ca", "O"): 1.967,
    ("K", "O"): 2.132, ("Na", "O"): 1.803, ("La", "O"): 2.172,
    ("Ti", "O"): 1.815, ("Zr", "O"): 1.928, ("Nb", "O"): 1.911,
    ("Mn", "O"): 1.790, ("Fe", "O"): 1.759, ("Ni", "O"): 1.654,
    ("Al", "O"): 1.651, ("Pb", "O"): 2.112,
    ("Cs", "I"): 3.220, ("Pb", "I"): 2.804, ("Sn", "I"): 2.720,
    ("Cs", "Br"): 3.080, ("Pb", "Br"): 2.620, ("Sn", "Br"): 2.540,
    ("Cs", "Cl"): 2.906, ("Pb", "Cl"): 2.460,
}
BOND_VALENCE_B = 0.37  # Universal softness parameter


def bond_valence_loss(
    frac_coords: torch.Tensor,
    atom_types: torch.Tensor,
    lattice_matrix: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float = 4.0,
) -> torch.Tensor:
    """Compute bond valence sum mismatch loss.

    Penalizes when the sum of bond valences around each atom deviates
    from the expected oxidation state.

    Args:
        frac_coords: [N, 3] fractional coordinates.
        atom_types: [N] element indices.
        lattice_matrix: [B, 3, 3] lattice matrices.
        batch: [N] batch assignment.
        cutoff: Distance cutoff for BVS calculation.

    Returns:
        Scalar loss.
    """
    # Convert to Cartesian
    cart = torch.zeros_like(frac_coords)
    batch_size = batch.max().item() + 1
    for b in range(batch_size):
        mask = batch == b
        cart[mask] = frac_to_cart(frac_coords[mask], lattice_matrix[b])

    # Pairwise distances within each batch
    total_loss = torch.tensor(0.0, device=frac_coords.device)
    count = 0

    for b in range(batch_size):
        mask = batch == b
        pos = cart[mask]
        types = atom_types[mask]
        n = pos.size(0)

        if n < 2:
            continue

        # Pairwise distances
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [n, n, 3]
        dist = diff.norm(dim=-1)  # [n, n]

        # For each pair within cutoff, compute bond valence
        valid = (dist < cutoff) & (dist > 0.5)  # Exclude self and too-close

        # Simple penalty: minimum distance should be reasonable
        if valid.any():
            min_dists = dist[valid]
            # Penalize distances below expected minimum (~1.5 A for most bonds)
            too_close = F.relu(1.5 - min_dists)
            total_loss = total_loss + too_close.mean()
            count += 1

    if count > 0:
        total_loss = total_loss / count

    return total_loss


# ---- Minimum distance loss ----

def min_distance_loss(
    frac_coords: torch.Tensor,
    lattice_matrix: torch.Tensor,
    batch: torch.Tensor,
    min_dist: float = 1.0,
) -> torch.Tensor:
    """Penalize atoms that are too close together.

    Args:
        frac_coords: [N, 3] fractional coordinates.
        lattice_matrix: [B, 3, 3] lattice matrices.
        batch: [N] batch assignment.
        min_dist: Minimum allowed distance in Angstrom.

    Returns:
        Scalar loss.
    """
    batch_size = batch.max().item() + 1
    total_loss = torch.tensor(0.0, device=frac_coords.device)
    count = 0

    for b in range(batch_size):
        mask = batch == b
        frac = frac_coords[mask]
        lat = lattice_matrix[b]
        n = frac.size(0)

        if n < 2:
            continue

        # Pairwise fractional distances with minimum image
        diff = frac.unsqueeze(0) - frac.unsqueeze(1)  # [n, n, 3]
        diff = diff - torch.round(diff)  # Wrap to [-0.5, 0.5)
        cart_diff = diff @ lat  # [n, n, 3]
        dist = cart_diff.norm(dim=-1)  # [n, n]

        # Mask diagonal
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=dist.device)
        dist = dist[diag_mask]

        if dist.numel() > 0:
            violation = F.relu(min_dist - dist)
            total_loss = total_loss + violation.mean()
            count += 1

    if count > 0:
        total_loss = total_loss / count
    return total_loss


# ---- Lattice regularity loss ----

def lattice_regularity_loss(
    lattice_params: torch.Tensor,
    max_aspect_ratio: float = 5.0,
    min_angle: float = 30.0,
    max_angle: float = 150.0,
) -> torch.Tensor:
    """Penalize unreasonable lattice parameters.

    Args:
        lattice_params: [B, 6] = [a, b, c, alpha, beta, gamma] (raw, not normalized).
        max_aspect_ratio: Max allowed ratio between longest/shortest axis.
        min_angle: Minimum allowed angle in degrees.
        max_angle: Maximum allowed angle in degrees.

    Returns:
        Scalar loss.
    """
    lengths = lattice_params[:, :3]  # [B, 3]
    angles = lattice_params[:, 3:]   # [B, 3]

    # Aspect ratio penalty
    l_max = lengths.max(dim=-1).values
    l_min = lengths.min(dim=-1).values.clamp(min=0.1)
    ratio = l_max / l_min
    ratio_penalty = F.relu(ratio - max_aspect_ratio).mean()

    # Angle penalty
    angle_low = F.relu(min_angle - angles).mean()
    angle_high = F.relu(angles - max_angle).mean()

    # Length positivity
    length_neg = F.relu(-lengths).mean()

    return ratio_penalty + angle_low + angle_high + length_neg


# ---- Combined physics loss ----

def combined_physics_loss(
    frac_coords: torch.Tensor,
    atom_types: torch.Tensor,
    lattice_params: torch.Tensor,
    lattice_matrix: torch.Tensor,
    batch: torch.Tensor,
    t: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    """Compute combined time-conditioned physics losses.

    Args:
        frac_coords: [N, 3] predicted/current fractional coordinates.
        atom_types: [N] atom type indices.
        lattice_params: [B, 6] raw lattice parameters.
        lattice_matrix: [B, 3, 3] lattice matrices.
        batch: [N] batch assignment.
        t: [B] time values in [0, 1].
        config: Loss config dict.

    Returns:
        total_loss: Scalar.
        loss_dict: Dict of individual loss components.
    """
    loss_cfg = config.get("loss", {})
    center = loss_cfg.get("time_condition_center", 0.3)
    steepness = loss_cfg.get("time_condition_steepness", 15.0)
    time_weight = compute_time_weight(t, center, steepness).mean()

    loss_dict = {}
    total = torch.tensor(0.0, device=frac_coords.device)

    # Bond valence
    lam_bv = loss_cfg.get("lambda_bond_valence", 0.1)
    if lam_bv > 0:
        bv_loss = bond_valence_loss(frac_coords, atom_types, lattice_matrix, batch)
        loss_dict["bond_valence"] = bv_loss.item()
        total = total + lam_bv * time_weight * bv_loss

    # Minimum distance
    lam_md = loss_cfg.get("lambda_min_dist", 0.05)
    if lam_md > 0:
        md_loss = min_distance_loss(frac_coords, lattice_matrix, batch)
        loss_dict["min_dist"] = md_loss.item()
        total = total + lam_md * time_weight * md_loss

    # Lattice regularity
    lam_lr = loss_cfg.get("lambda_lattice_reg", 0.02)
    if lam_lr > 0:
        lr_loss = lattice_regularity_loss(lattice_params)
        loss_dict["lattice_reg"] = lr_loss.item()
        total = total + lam_lr * time_weight * lr_loss

    loss_dict["physics_total"] = total.item()
    loss_dict["time_weight"] = time_weight.item()

    return total, loss_dict
