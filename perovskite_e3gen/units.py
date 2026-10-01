"""Lattice and coordinate conversion utilities for perovskite generation.

Handles conversions between:
- Lattice parameters [a, b, c, alpha, beta, gamma] <-> lattice matrix [3, 3]
- Fractional coordinates <-> Cartesian coordinates
- Lattice parameter normalization (z-score) for training

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import math

import torch
import numpy as np


def lattice_params_to_matrix(
    lengths: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Convert lattice parameters to 3x3 lattice matrix.

    Uses the convention where:
      a along x-axis
      b in xy-plane
      c determined by angles

    Args:
        lengths: [..., 3] tensor of (a, b, c) in Angstrom.
        angles: [..., 3] tensor of (alpha, beta, gamma) in degrees.

    Returns:
        [..., 3, 3] lattice matrix where rows are lattice vectors.
    """
    angles_rad = angles * (math.pi / 180.0)
    alpha, beta, gamma = angles_rad[..., 0], angles_rad[..., 1], angles_rad[..., 2]
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # a along x
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)

    # b in xy-plane
    bx = b * cos_gamma
    by = b * sin_gamma
    bz = torch.zeros_like(b)

    # c from angles
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma.clamp(min=1e-8)
    cz = torch.sqrt(
        (c ** 2 - cx ** 2 - cy ** 2).clamp(min=1e-8)
    )

    # Stack into matrix: rows = lattice vectors
    # Shape: [..., 3, 3]
    row_a = torch.stack([ax, ay, az], dim=-1)
    row_b = torch.stack([bx, by, bz], dim=-1)
    row_c = torch.stack([cx, cy, cz], dim=-1)

    return torch.stack([row_a, row_b, row_c], dim=-2)


def lattice_matrix_to_params(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 3x3 lattice matrix to lattice parameters.

    Args:
        matrix: [..., 3, 3] lattice matrix (rows = lattice vectors).

    Returns:
        lengths: [..., 3] tensor of (a, b, c) in Angstrom.
        angles: [..., 3] tensor of (alpha, beta, gamma) in degrees.
    """
    va = matrix[..., 0, :]  # [..., 3]
    vb = matrix[..., 1, :]
    vc = matrix[..., 2, :]

    a = va.norm(dim=-1)
    b = vb.norm(dim=-1)
    c = vc.norm(dim=-1)

    cos_alpha = (vb * vc).sum(dim=-1) / (b * c).clamp(min=1e-8)
    cos_beta = (va * vc).sum(dim=-1) / (a * c).clamp(min=1e-8)
    cos_gamma = (va * vb).sum(dim=-1) / (a * b).clamp(min=1e-8)

    # Clamp for numerical safety before acos
    cos_alpha = cos_alpha.clamp(-1 + 1e-7, 1 - 1e-7)
    cos_beta = cos_beta.clamp(-1 + 1e-7, 1 - 1e-7)
    cos_gamma = cos_gamma.clamp(-1 + 1e-7, 1 - 1e-7)

    alpha = torch.acos(cos_alpha) * (180.0 / math.pi)
    beta = torch.acos(cos_beta) * (180.0 / math.pi)
    gamma = torch.acos(cos_gamma) * (180.0 / math.pi)

    lengths = torch.stack([a, b, c], dim=-1)
    angles = torch.stack([alpha, beta, gamma], dim=-1)
    return lengths, angles


def frac_to_cart(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Convert fractional coordinates to Cartesian.

    Args:
        frac_coords: [N, 3] fractional coordinates.
        lattice: [3, 3] lattice matrix (rows = lattice vectors).

    Returns:
        [N, 3] Cartesian coordinates in Angstrom.
    """
    return frac_coords @ lattice


def cart_to_frac(
    cart_coords: torch.Tensor,
    lattice: torch.Tensor,
) -> torch.Tensor:
    """Convert Cartesian coordinates to fractional.

    Args:
        cart_coords: [N, 3] Cartesian coordinates in Angstrom.
        lattice: [3, 3] lattice matrix (rows = lattice vectors).

    Returns:
        [N, 3] fractional coordinates.
    """
    return cart_coords @ torch.linalg.inv(lattice)


def wrap_frac_coords(frac_coords: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coordinates to [0, 1)."""
    return frac_coords % 1.0


def lattice_params_to_vector(
    lengths: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Pack lattice parameters into a single [6] vector.

    Args:
        lengths: [..., 3] (a, b, c).
        angles: [..., 3] (alpha, beta, gamma) in degrees.

    Returns:
        [..., 6] vector: [a, b, c, alpha, beta, gamma].
    """
    return torch.cat([lengths, angles], dim=-1)


def vector_to_lattice_params(
    vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack [6] vector into lengths and angles.

    Args:
        vec: [..., 6] vector: [a, b, c, alpha, beta, gamma].

    Returns:
        lengths: [..., 3] (a, b, c).
        angles: [..., 3] (alpha, beta, gamma) in degrees.
    """
    return vec[..., :3], vec[..., 3:]


class LatticeScaler:
    """Z-score normalization for lattice parameters.

    Computes mean/std from dataset and provides normalize/denormalize methods.
    Lattice parameters: [a, b, c, alpha, beta, gamma]
    """

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean = mean  # [6]
        self.std = std    # [6]

    @classmethod
    def from_dataset(cls, lattice_params: torch.Tensor) -> "LatticeScaler":
        """Compute scaler statistics from a dataset of lattice parameters.

        Args:
            lattice_params: [N, 6] tensor of [a, b, c, alpha, beta, gamma].

        Returns:
            LatticeScaler with mean/std computed.
        """
        mean = lattice_params.mean(dim=0)
        std = lattice_params.std(dim=0).clamp(min=1e-4)
        return cls(mean=mean, std=std)

    def normalize(self, params: torch.Tensor) -> torch.Tensor:
        """Normalize lattice params to zero mean, unit variance."""
        return (params - self.mean.to(params.device)) / self.std.to(params.device)

    def denormalize(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize lattice params back to physical units."""
        return params_norm * self.std.to(params_norm.device) + self.mean.to(params_norm.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "LatticeScaler":
        return cls(mean=state["mean"], std=state["std"])


# ---- Element utilities ----

# Common elements in perovskites (subset of periodic table)
PEROVSKITE_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

# Map element symbol -> index (0-based, 0 reserved for padding)
ELEMENT_TO_IDX = {elem: i + 1 for i, elem in enumerate(PEROVSKITE_ELEMENTS)}
IDX_TO_ELEMENT = {i + 1: elem for i, elem in enumerate(PEROVSKITE_ELEMENTS)}


def element_to_index(symbol: str) -> int:
    """Convert element symbol to integer index."""
    return ELEMENT_TO_IDX.get(symbol, 0)


def index_to_element(idx: int) -> str:
    """Convert integer index to element symbol."""
    return IDX_TO_ELEMENT.get(idx, "X")


# ---- Shannon ionic radii for tolerance factor (common ions) ----
# Source: Shannon, Acta Cryst. A32, 751 (1976)
IONIC_RADII = {
    # A-site cations (12-coordinate or 6-coordinate)
    "Cs": 1.88, "Rb": 1.72, "K": 1.64, "Na": 1.39, "Li": 0.92,
    "Ba": 1.61, "Sr": 1.44, "Ca": 1.34, "La": 1.36, "Bi": 1.17,
    # B-site cations (6-coordinate)
    "Ti": 0.605, "Zr": 0.72, "Nb": 0.64, "Ta": 0.64,
    "Mn": 0.645, "Fe": 0.645, "Ni": 0.69, "Co": 0.545,
    "Al": 0.535, "Ga": 0.62, "In": 0.80, "Sc": 0.745,
    "Sn": 0.69, "Pb": 1.19, "Ge": 0.53, "Ru": 0.62,
    # Anions
    "O": 1.40, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "S": 1.84, "Se": 1.98,
}


def goldschmidt_tolerance(r_A: float, r_B: float, r_X: float) -> float:
    """Compute Goldschmidt tolerance factor t = (r_A + r_X) / (sqrt(2) * (r_B + r_X)).

    For ideal perovskite: t ~ 0.8 - 1.0
    """
    return (r_A + r_X) / (math.sqrt(2) * (r_B + r_X))
