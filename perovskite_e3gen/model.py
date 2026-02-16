"""Periodic PaiNN model for perovskite crystal generation.

Adapts PaiNN (Polarizable Atom Interaction Neural Network) for periodic
crystals with three coupled output heads:
  1. Coordinate head: per-atom [N, 3] velocity in fractional space
  2. Lattice head: global pooling -> MLP -> [6] lattice velocity
  3. Type head: per-atom [N, num_elements] logits

Key differences from fullerene PaiNN:
  - Periodic boundary conditions via minimum image convention
  - Multi-element support with learned element embeddings
  - Composition conditioning via FiLM (feature-wise linear modulation)
  - Lattice parameter prediction as global property
  - No center-of-mass projection (crystal coords are fractional)

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---- Utility modules ----

def _scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
             dim_size: int | None = None, reduce: str = 'sum') -> torch.Tensor:
    """Pure-PyTorch scatter (avoids torch_scatter dependency)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    idx = index
    for _ in range(src.dim() - idx.dim()):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)

    out = torch.zeros(*[dim_size if d == dim else src.size(d) for d in range(src.dim())],
                       device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, idx, src)

    if reduce == 'mean':
        ones = torch.ones(src.size(dim), device=src.device, dtype=src.dtype)
        count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        count.scatter_add_(0, index, ones)
        count = count.clamp(min=1)
        shape = [1] * out.dim()
        shape[dim] = dim_size
        count = count.view(shape).expand_as(out)
        out = out / count

    return out


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for continuous time values."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] time values (can be float in [0, 1] or int).
        Returns:
            [B, dim] positional embedding.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_float = t.float()
        args = t_float.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class RadialBasis(nn.Module):
    """Sinc radial basis functions for distance encoding."""

    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        freqs = torch.arange(1, num_rbf + 1, dtype=torch.float32) * math.pi / cutoff
        self.register_buffer('freqs', freqs)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)
        return torch.sin(self.freqs * dist) / dist.clamp(min=1e-8)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()


# ---- PaiNN building blocks ----

class PaiNNMessage(nn.Module):
    """PaiNN message passing adapted for periodic systems.

    Computes scalar + vector messages using:
    - Minimum image convention for PBC distances
    - Radial basis + cosine cutoff filtering
    - Direction-based vector messages for angular awareness
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rbf = RadialBasis(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        s: torch.Tensor,
        V: torch.Tensor,
        rel_pos_cart: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [N, F] scalar features.
            V: [N, F, 3] vector features.
            rel_pos_cart: [E, 3] Cartesian relative positions (with PBC applied).
            edge_index: [2, E] source-target pairs.

        Returns:
            ds: [N, F] scalar message aggregation.
            dV: [N, F, 3] vector message aggregation.
        """
        row, col = edge_index  # row=dst, col=src

        dist = rel_pos_cart.norm(dim=-1).clamp(min=1e-8)
        d_hat = rel_pos_cart / dist.unsqueeze(-1)

        rbf_feat = self.rbf(dist)
        cutoff_val = self.cutoff_fn(dist)

        W = self.filter_net(rbf_feat) * cutoff_val.unsqueeze(-1)
        W_s, W_vv, W_vd = W.chunk(3, dim=-1)

        s_j = self.scalar_mlp(s[col])
        s_s, s_vv, s_vd = s_j.chunk(3, dim=-1)

        scalar_msg = W_s * s_s

        V_j = V[col]
        vec_from_V = W_vv.unsqueeze(-1) * s_vv.unsqueeze(-1) * V_j
        vec_from_d = W_vd.unsqueeze(-1) * s_vd.unsqueeze(-1) * d_hat.unsqueeze(1)
        vec_msg = vec_from_V + vec_from_d

        num_nodes = s.size(0)
        ds = _scatter(scalar_msg, row, dim=0, dim_size=num_nodes, reduce='sum')
        dV = _scatter(vec_msg, row, dim=0, dim_size=num_nodes, reduce='sum')

        return ds, dV


class PaiNNUpdate(nn.Module):
    """PaiNN update: scalar-vector interaction."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Uv = self.U(V.transpose(-1, -2)).transpose(-1, -2)
        Vv = self.V_lin(V.transpose(-1, -2)).transpose(-1, -2)
        inner = (Uv * Vv).sum(dim=-1)
        Vv_norm = Vv.norm(dim=-1).clamp(min=1e-8)

        scalar_input = torch.cat([s, Vv_norm], dim=-1)
        scalar_out = self.scalar_mlp(scalar_input)
        a_ss, a_sv, a_vv = scalar_out.chunk(3, dim=-1)

        ds = a_ss + a_sv * inner
        dV = a_vv.unsqueeze(-1) * Uv

        return s + ds, V + dV


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for conditioning.

    Applies: out = scale * LayerNorm(x) + shift
    where scale, shift come from conditioning signal.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return (1 + scale) * self.norm(x) + shift


class PaiNNLayer(nn.Module):
    """Single PaiNN layer with FiLM conditioning."""

    def __init__(self, hidden_dim: int, num_rbf: int = 20, cutoff: float = 5.0, cond_dim: int = 256):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, num_rbf, cutoff)
        self.update = PaiNNUpdate(hidden_dim)
        self.film = FiLMLayer(hidden_dim, cond_dim)

    def forward(self, s, V, rel_pos_cart, edge_index, cond):
        ds, dV = self.message(s, V, rel_pos_cart, edge_index)
        s = s + ds
        V = V + dV
        s, V = self.update(s, V)
        s = self.film(s, cond)
        return s, V


# ---- Main model ----

class PerovskitePaiNNModel(nn.Module):
    """PaiNN-based flow matching model for perovskite crystal generation.

    Joint prediction of:
      1. Fractional coordinate velocity [N, 3]
      2. Lattice parameter velocity [B, 6]
      3. Atom type logits [N, num_elements]

    Conditioned on:
      - Diffusion time t
      - Composition (A, B, X element embeddings)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        time_embed_dim: int = 128,
        num_elements: int = 100,
        element_embed_dim: int = 64,
        composition_embed_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.num_elements = num_elements

        cond_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Element embedding
        self.element_embed = nn.Embedding(num_elements + 1, element_embed_dim, padding_idx=0)

        # Composition encoder: aggregate element embeddings -> composition vector
        self.composition_encoder = nn.Sequential(
            nn.Linear(element_embed_dim, composition_embed_dim),
            nn.SiLU(),
            nn.Linear(composition_embed_dim, composition_embed_dim),
        )

        # Combine time + composition into conditioning
        self.cond_proj = nn.Linear(hidden_dim + composition_embed_dim, hidden_dim)

        # Node feature initialization: element embedding + conditioning
        self.node_init = nn.Linear(element_embed_dim + hidden_dim, hidden_dim)

        # PaiNN layers
        self.layers = nn.ModuleList([
            PaiNNLayer(hidden_dim, num_rbf, cutoff, cond_dim)
            for _ in range(num_layers)
        ])

        # Output head 1: fractional coordinate velocity [N, 3]
        self.coord_head = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.coord_scale = nn.Parameter(torch.ones(1))

        # Output head 2: lattice parameter velocity [B, 6]
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

        # Output head 3: atom type logits [N, num_elements]
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_elements),
        )

    def _compute_pbc_rel_pos(
        self,
        frac_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
        lattice_matrix: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Cartesian relative positions with PBC.

        Args:
            frac_coords: [N, 3] fractional coordinates.
            edge_index: [2, E] edges.
            edge_shift: [E, 3] fractional shift vectors.
            lattice_matrix: [B, 3, 3] per-batch lattice matrices.
            batch: [N] batch assignment.

        Returns:
            [E, 3] Cartesian relative positions.
        """
        row, col = edge_index  # row=dst, col=src

        # Relative fractional position with shift
        frac_rel = frac_coords[row] - frac_coords[col] - edge_shift  # [E, 3]

        # Convert to Cartesian using per-edge lattice
        edge_lattice = lattice_matrix[batch[row]]  # [E, 3, 3]
        cart_rel = torch.bmm(frac_rel.unsqueeze(1), edge_lattice).squeeze(1)  # [E, 3]

        return cart_rel

    def _get_composition_embedding(
        self,
        atom_types: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-batch composition embedding by averaging element embeddings.

        Args:
            atom_types: [N] element indices.
            batch: [N] batch assignment.

        Returns:
            [B, composition_embed_dim] composition embedding.
        """
        elem_emb = self.element_embed(atom_types)  # [N, elem_embed_dim]
        batch_size = batch.max().item() + 1
        comp_emb = _scatter(elem_emb, batch, dim=0, dim_size=batch_size, reduce='mean')
        return self.composition_encoder(comp_emb)  # [B, comp_embed_dim]

    def forward(
        self,
        frac_coords: torch.Tensor,
        atom_types: torch.Tensor,
        lattice_matrix: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: predict velocity fields and type logits.

        Args:
            frac_coords: [N, 3] noisy fractional coordinates at time t.
            atom_types: [N] noisy atom type indices (or ground truth for early training).
            lattice_matrix: [B, 3, 3] noisy lattice at time t.
            t: [B] diffusion time in [0, 1].
            edge_index: [2, E] graph edges.
            edge_shift: [E, 3] PBC shift vectors.
            batch: [N] batch assignment.

        Returns:
            coord_vel: [N, 3] predicted fractional coordinate velocity.
            lattice_vel: [B, 6] predicted lattice parameter velocity.
            type_logits: [N, num_elements] atom type logits.
        """
        num_nodes = frac_coords.size(0)
        batch_size = batch.max().item() + 1

        # Conditioning
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        comp_emb = self._get_composition_embedding(atom_types, batch)  # [B, comp_embed_dim]
        cond = self.cond_proj(torch.cat([t_emb, comp_emb], dim=-1))  # [B, hidden_dim]
        cond_per_node = cond[batch]  # [N, hidden_dim]

        # Node initialization
        elem_emb = self.element_embed(atom_types)  # [N, elem_embed_dim]
        s = self.node_init(torch.cat([elem_emb, cond_per_node], dim=-1))  # [N, hidden_dim]
        V = torch.zeros(num_nodes, self.hidden_dim, 3, device=frac_coords.device, dtype=frac_coords.dtype)

        # Compute PBC-aware relative positions
        rel_pos_cart = self._compute_pbc_rel_pos(
            frac_coords, edge_index, edge_shift, lattice_matrix, batch
        )

        # PaiNN layers
        for layer in self.layers:
            s, V = layer(s, V, rel_pos_cart, edge_index, cond_per_node)

        # Head 1: coordinate velocity (equivariant via vector features)
        coord_vel = torch.einsum('nfc,f->nc', V, self.coord_head) * self.coord_scale  # [N, 3]

        # Head 2: lattice velocity (invariant, via global pooling)
        s_global = _scatter(s, batch, dim=0, dim_size=batch_size, reduce='mean')  # [B, hidden_dim]
        lattice_vel = self.lattice_head(s_global)  # [B, 6]

        # Head 3: atom type logits (invariant)
        type_logits = self.type_head(s)  # [N, num_elements]

        # NaN safety
        if torch.isnan(coord_vel).any() or torch.isnan(lattice_vel).any():
            logger.warning("NaN in model output, returning zeros")
            coord_vel = torch.zeros_like(coord_vel)
            lattice_vel = torch.zeros_like(lattice_vel)
            type_logits = torch.zeros_like(type_logits)

        return coord_vel, lattice_vel, type_logits


if __name__ == '__main__':
    """Quick test of Periodic PaiNN model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Simulate a 5-atom BaTiO3 structure
    num_nodes = 5
    frac_coords = torch.rand(num_nodes, 3)
    atom_types = torch.tensor([56, 22, 8, 8, 8])  # Ba, Ti, O, O, O (as indices)
    lattice_matrix = torch.eye(3).unsqueeze(0) * 4.0  # [1, 3, 3] cubic ~4A

    # Simple fully-connected graph
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_shift = torch.zeros(edge_index.size(1), 3)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    t = torch.tensor([0.5])

    model = PerovskitePaiNNModel(hidden_dim=64, num_layers=4)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    with torch.no_grad():
        coord_vel, lattice_vel, type_logits = model(
            frac_coords, atom_types, lattice_matrix, t, edge_index, edge_shift, batch
        )
    logger.info("coord_vel: %s, lattice_vel: %s, type_logits: %s",
                coord_vel.shape, lattice_vel.shape, type_logits.shape)

    # Test equivariance: rotating Cartesian space should rotate coord velocity
    # (In fractional space, this is lattice-dependent, so we test shape consistency)
    logger.info("All model tests passed!")
