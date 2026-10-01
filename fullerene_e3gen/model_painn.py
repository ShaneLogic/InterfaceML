"""
PaiNN (Polarizable Atom Interaction Neural Network) for fullerene diffusion.

PaiNN maintains scalar s [N, F] + vector V [N, F, 3] features, giving it
angular awareness that EGNN lacks. This is critical for sp2 carbon where
120-degree bond angles define the structure.

Key advantages over EGNN:
  1. Angular information via vector-scalar interactions (inner products)
  2. No distance suppression — uses smooth cosine cutoff instead
  3. AdaLayerNorm for time conditioning at every layer
  4. CFG (Classifier-Free Guidance) via conditional C dropout

Reference:
  Schütt et al., "Equivariant message passing for the prediction of
  tensorial properties and molecular spectra", ICML 2021

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

from model import SinusoidalPositionEmbedding, ContinuousCEmbedding


def _scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
             dim_size: int | None = None, reduce: str = 'sum') -> torch.Tensor:
    """Pure-PyTorch scatter replacement (avoids torch_scatter dependency).

    Supports reduce='sum' and reduce='mean'.
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Expand index to match src shape
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
        # Expand count to match out shape
        shape = [1] * out.dim()
        shape[dim] = dim_size
        count = count.view(shape).expand_as(out)
        out = out / count

    return out

logger = logging.getLogger(__name__)


class RadialBasis(nn.Module):
    """Sinc radial basis functions for distance encoding.

    Maps scalar distances to a feature vector using sinc functions:
        RBF_n(d) = sin(n * pi * d / cutoff) / d

    This provides a smooth, orthogonal basis that resolves distance
    differences within the cutoff.
    """

    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        # Frequencies: n * pi / cutoff for n = 1..num_rbf
        freqs = torch.arange(1, num_rbf + 1, dtype=torch.float32) * math.pi / cutoff
        self.register_buffer('freqs', freqs)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: [E] pairwise distances.
        Returns:
            [E, num_rbf] radial basis features.
        """
        dist = dist.unsqueeze(-1)  # [E, 1]
        # sinc basis: sin(n*pi*d/cutoff) / d
        return torch.sin(self.freqs * dist) / dist.clamp(min=1e-8)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope.

    Smoothly goes from 1 at d=0 to 0 at d=cutoff:
        f(d) = 0.5 * (cos(pi * d / cutoff) + 1)  for d < cutoff
        f(d) = 0                                    for d >= cutoff
    """

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: [E] pairwise distances.
        Returns:
            [E] cutoff envelope values in [0, 1].
        """
        return 0.5 * (torch.cos(dist * math.pi / self.cutoff) + 1.0) * (dist < self.cutoff).float()


class PaiNNMessage(nn.Module):
    """PaiNN message passing: compute scalar + vector messages.

    Scalar messages are filtered by radial basis and cutoff.
    Vector messages mix neighbor vector features V_j with unit
    direction vectors d_hat_ij, enabling angular awareness.
    """

    def __init__(self, hidden_dim: int, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Radial basis and cutoff
        self.rbf = RadialBasis(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Filter network: RBF features -> 3*hidden_dim (for scalar, vec_scale, vec_dir)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Scalar pre-processing
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        s: torch.Tensor,       # [N, F]
        V: torch.Tensor,       # [N, F, 3]
        pos: torch.Tensor,     # [N, 3]
        edge_index: torch.Tensor,  # [2, E]
    ) -> tuple:
        """
        Returns:
            ds: [N, F] scalar message aggregation
            dV: [N, F, 3] vector message aggregation
        """
        row, col = edge_index  # row = destination, col = source

        # Pairwise geometry
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist = rel_pos.norm(dim=-1).clamp(min=1e-8)  # [E]
        d_hat = rel_pos / dist.unsqueeze(-1)  # [E, 3] unit direction

        # Radial features and cutoff
        rbf_feat = self.rbf(dist)  # [E, num_rbf]
        cutoff_val = self.cutoff_fn(dist)  # [E]

        # Filter: [E, 3*F]
        W = self.filter_net(rbf_feat) * cutoff_val.unsqueeze(-1)

        # Split into three channels
        W_s, W_vv, W_vd = W.chunk(3, dim=-1)  # each [E, F]

        # Scalar pre-processing of source nodes
        s_j = self.scalar_mlp(s[col])  # [E, 3*F]
        s_s, s_vv, s_vd = s_j.chunk(3, dim=-1)  # each [E, F]

        # Scalar messages: element-wise product of filter and source scalar
        scalar_msg = W_s * s_s  # [E, F]

        # Vector messages from neighbor vectors V_j, scaled by filter
        V_j = V[col]  # [E, F, 3]
        vec_from_V = W_vv.unsqueeze(-1) * s_vv.unsqueeze(-1) * V_j  # [E, F, 3]

        # Vector messages from direction d_hat, scaled by filter
        vec_from_d = W_vd.unsqueeze(-1) * s_vd.unsqueeze(-1) * d_hat.unsqueeze(1)  # [E, F, 3] via broadcast [E,1,3]

        vec_msg = vec_from_V + vec_from_d  # [E, F, 3]

        # Aggregate to destination nodes
        num_nodes = s.size(0)
        ds = _scatter(scalar_msg, row, dim=0, dim_size=num_nodes, reduce='sum')  # [N, F]
        dV = _scatter(vec_msg, row, dim=0, dim_size=num_nodes, reduce='sum')  # [N, F, 3]

        return ds, dV


class PaiNNUpdate(nn.Module):
    """PaiNN update: scalar-vector interaction.

    The inner product <U*V_i, W*V_i> gives angular information to the
    scalar channel. The scalar channel then gates vector updates.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear transforms for vector channels (no bias for equivariance)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_lin = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Scalar update MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, s: torch.Tensor, V: torch.Tensor) -> tuple:
        """
        Args:
            s: [N, F] scalar features
            V: [N, F, 3] vector features
        Returns:
            s_out: [N, F] updated scalar features
            V_out: [N, F, 3] updated vector features
        """
        # Apply linear transforms to vector features
        # V is [N, F, 3], we apply linear across the F dimension
        Uv = self.U(V.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv = self.V_lin(V.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]

        # Inner product <Uv, Vv> gives angular info
        inner = (Uv * Vv).sum(dim=-1)  # [N, F]

        # Norm of Vv for scalar gating
        Vv_norm = Vv.norm(dim=-1).clamp(min=1e-8)  # [N, F]

        # Scalar update: uses both s and angular inner product
        scalar_input = torch.cat([s, Vv_norm], dim=-1)  # [N, 2F]
        scalar_out = self.scalar_mlp(scalar_input)  # [N, 3F]
        a_ss, a_sv, a_vv = scalar_out.chunk(3, dim=-1)  # each [N, F]

        # Update scalar: residual + inner product contribution
        ds = a_ss + a_sv * inner  # [N, F]

        # Update vector: scalar-gated
        dV = a_vv.unsqueeze(-1) * Uv  # [N, F, 3]

        return s + ds, V + dV


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on time embedding.

    Applies LayerNorm then modulates with learned scale and shift
    from the conditioning signal (time embedding):
        out = scale * LayerNorm(x) + shift
    where scale, shift = Linear(cond)
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, F] features to normalize
            cond: [N, cond_dim] conditioning signal (broadcast from batch)
        Returns:
            [N, F] normalized and modulated features
        """
        scale_shift = self.proj(cond)  # [N, 2F]
        scale, shift = scale_shift.chunk(2, dim=-1)  # each [N, F]
        return (1 + scale) * self.norm(x) + shift


class PaiNNLayer(nn.Module):
    """Single PaiNN layer: message passing + update + AdaLayerNorm."""

    def __init__(self, hidden_dim: int, num_rbf: int = 20, cutoff: float = 5.0, cond_dim: int = 256):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, num_rbf, cutoff)
        self.update = PaiNNUpdate(hidden_dim)
        self.norm_s = AdaLayerNorm(hidden_dim, cond_dim)

    def forward(
        self,
        s: torch.Tensor,
        V: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple:
        """
        Args:
            s: [N, F] scalar features
            V: [N, F, 3] vector features
            pos: [N, 3] atom positions (fixed, not updated)
            edge_index: [2, E]
            cond: [N, cond_dim] time + condition embedding
        Returns:
            s_out, V_out
        """
        # Message passing with residual
        ds, dV = self.message(s, V, pos, edge_index)
        s = s + ds
        V = V + dV

        # Scalar-vector interaction
        s, V = self.update(s, V)

        # AdaLayerNorm on scalar features (conditioned on time)
        s = self.norm_s(s, cond)

        return s, V


class FullerenePaiNNModel(nn.Module):
    """PaiNN-based diffusion model for fullerene coordinate generation.

    Key differences from EGNN model:
      1. Maintains scalar + vector features (angular awareness)
      2. Fixed input positions (no per-layer coordinate updates)
      3. Output via equivariant linear combination of vector features
      4. CFG dropout: randomly drops C conditioning during training
      5. AdaLayerNorm for time conditioning at every layer

    Forward signature matches EGNN model for drop-in replacement:
        forward(pos, edge_index, t, C, batch) -> noise_pred [N, 3]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_rbf: int = 20,
        cutoff: float = 5.0,
        time_embed_dim: int = 128,
        C_embed_dim: int = 64,
        C_fourier_features: int = 16,
        max_C: int = 720,
        cfg_drop_prob: float = 0.1,
        # Phase 3 scaffolding
        num_atom_types: int = 1,
        use_pbc: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.cfg_drop_prob = cfg_drop_prob
        self.num_atom_types = num_atom_types
        self.use_pbc = use_pbc

        # Conditioning dimension = hidden_dim (time + C combined)
        cond_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Carbon count embedding (continuous Fourier)
        self.C_embed = ContinuousCEmbedding(
            embed_dim=C_embed_dim,
            fourier_features=C_fourier_features,
            C_ref=100.0,
        )

        # Project C_embed to hidden_dim and combine with time
        self.cond_proj = nn.Linear(C_embed_dim + hidden_dim, hidden_dim)

        # Null C embedding for CFG (learned, replaces C_embed when dropped)
        self.null_C_embed = nn.Parameter(torch.randn(C_embed_dim) * 0.02)

        # Atom type embedding (Phase 3: multi-species)
        if num_atom_types > 1:
            self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        else:
            self.atom_embed = None

        # Initialize scalar features from conditioning
        self.scalar_init = nn.Linear(hidden_dim, hidden_dim)

        # PaiNN layers
        self.layers = nn.ModuleList([
            PaiNNLayer(hidden_dim, num_rbf, cutoff, cond_dim)
            for _ in range(num_layers)
        ])

        # Output: equivariant linear combination of vector features
        # noise_pred = einsum('nfc, f -> nc', V, output_weight)
        self.output_weight = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # Output scaling (learnable, initialized to 1)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        C: torch.Tensor,
        batch: torch.Tensor,
        atom_type: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Equivariant noise/velocity prediction via PaiNN vector features.

        Args:
            pos: [N, 3] noisy coordinates
            edge_index: [2, E] graph connectivity
            t: [B] diffusion timestep (integer for DDPM, or continuous for flow matching)
            C: [B] carbon count conditioning
            batch: [N] batch assignment
            atom_type: [N] optional atom type indices (Phase 3)
            cell: [B, 3, 3] optional unit cell for PBC (Phase 3)

        Returns:
            noise_pred: [N, 3] predicted noise or velocity field
        """
        num_nodes = pos.size(0)

        # --- Compute conditioning ---
        t_emb = self.time_embed(t)  # [B, hidden_dim]

        # CFG: randomly drop C conditioning during training
        if self.training and self.cfg_drop_prob > 0:
            C_emb = self.C_embed(C)  # [B, C_embed_dim]
            # Randomly replace with null embedding
            drop_mask = torch.rand(C.size(0), device=C.device) < self.cfg_drop_prob
            null_expanded = self.null_C_embed.unsqueeze(0).expand(C.size(0), -1)  # [B, C_embed_dim]
            C_emb = torch.where(drop_mask.unsqueeze(-1), null_expanded, C_emb)
        else:
            C_emb = self.C_embed(C)  # [B, C_embed_dim]

        # Combine time + C into single conditioning vector
        cond = self.cond_proj(torch.cat([t_emb, C_emb], dim=-1))  # [B, hidden_dim]
        cond_per_node = cond[batch]  # [N, hidden_dim]

        # --- Initialize features ---
        s = self.scalar_init(cond_per_node)  # [N, hidden_dim]

        # Multi-species embedding (Phase 3)
        if self.atom_embed is not None and atom_type is not None:
            s = s + self.atom_embed(atom_type)

        V = torch.zeros(num_nodes, self.hidden_dim, 3, device=pos.device, dtype=pos.dtype)

        # --- PBC minimum-image convention (Phase 3 scaffolding) ---
        pos_msg = pos
        if self.use_pbc and cell is not None:
            pos_msg = self._apply_minimum_image(pos, edge_index, cell, batch)

        # --- PaiNN layers (positions are FIXED, not updated) ---
        for layer in self.layers:
            s, V = layer(s, V, pos_msg, edge_index, cond_per_node)

        # --- Equivariant output ---
        # Linear combination of vector features: sum_f w_f * V[:, f, :]
        noise_pred = torch.einsum('nfc,f->nc', V, self.output_weight)  # [N, 3]
        noise_pred = noise_pred * self.output_scale

        # --- Center of mass projection (translation equivariance) ---
        # Subtract per-molecule mean so output is zero-CoM
        batch_size = batch.max().item() + 1
        com = _scatter(noise_pred, batch, dim=0, dim_size=batch_size, reduce='mean')  # [B, 3]
        noise_pred = noise_pred - com[batch]

        # NaN safety
        if torch.isnan(noise_pred).any():
            logger.warning("NaN in PaiNN output, returning zeros")
            noise_pred = torch.zeros_like(noise_pred)

        return noise_pred

    def forward_cfg(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        C: torch.Tensor,
        batch: torch.Tensor,
        cfg_weight: float = 2.0,
    ) -> torch.Tensor:
        """CFG inference: two forward passes (conditional + unconditional).

        v_cfg = v_uncond + cfg_weight * (v_cond - v_uncond)
              = (1 - cfg_weight) * v_uncond + cfg_weight * v_cond

        Args:
            cfg_weight: guidance strength (1.0 = no guidance, >1.0 = amplify conditioning)
        """
        # Conditional forward pass
        v_cond = self.forward(pos, edge_index, t, C, batch)

        # Unconditional: replace C_emb with null
        t_emb = self.time_embed(t)
        null_C = self.null_C_embed.unsqueeze(0).expand(C.size(0), -1)
        cond_uncond = self.cond_proj(torch.cat([t_emb, null_C], dim=-1))
        cond_per_node = cond_uncond[batch]

        s = self.scalar_init(cond_per_node)
        V = torch.zeros(pos.size(0), self.hidden_dim, 3, device=pos.device, dtype=pos.dtype)

        for layer in self.layers:
            s, V = layer(s, V, pos, edge_index, cond_per_node)

        v_uncond = torch.einsum('nfc,f->nc', V, self.output_weight) * self.output_scale
        batch_size = batch.max().item() + 1
        com = _scatter(v_uncond, batch, dim=0, dim_size=batch_size, reduce='mean')
        v_uncond = v_uncond - com[batch]

        # CFG combination
        return (1 - cfg_weight) * v_uncond + cfg_weight * v_cond

    def _apply_minimum_image(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cell: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply minimum-image convention for periodic boundaries (Phase 3).

        This is a placeholder that returns positions unchanged.
        Full PBC support requires modifying PaiNNMessage to use
        minimum-image relative positions:

            rel_pos = pos[row] - pos[col]
            cell_inv = torch.linalg.inv(cell[batch[row]])
            frac = torch.einsum('ei,eij->ej', rel_pos, cell_inv)
            frac = frac - torch.round(frac)
            rel_pos_mic = torch.einsum('ei,eij->ej', frac, cell[batch[row]])
        """
        # TODO: integrate minimum-image into PaiNNMessage.forward()
        return pos


if __name__ == '__main__':
    """Quick test of PaiNN model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    num_nodes = 60
    pos = torch.randn(num_nodes, 3)
    edges = []
    for i in range(num_nodes):
        for j in [(i + 1) % num_nodes, (i + 2) % num_nodes, (i - 1) % num_nodes]:
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    t = torch.tensor([500])
    C = torch.tensor([60])
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model = FullerenePaiNNModel(hidden_dim=64, num_layers=4, num_rbf=16, cutoff=5.0)
    logger.info("PaiNN model created: %d parameters", sum(p.numel() for p in model.parameters()))

    with torch.no_grad():
        noise_pred = model(pos, edge_index, t, C, batch)
    logger.info("Output shape: %s, mean: %.4f, std: %.4f",
                noise_pred.shape, noise_pred.mean(), noise_pred.std())

    # Test equivariance: rotate input, check output rotates
    theta = torch.tensor(0.7)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    pos_rot = pos @ R.T
    with torch.no_grad():
        noise_rot = model(pos_rot, edge_index, t, C, batch)
        noise_expected = noise_pred @ R.T
    err = (noise_rot - noise_expected).norm() / noise_expected.norm().clamp(min=1e-8)
    logger.info("Equivariance test: relative error = %.6f (should be ~0)", err.item())

    # Test CFG
    with torch.no_grad():
        noise_cfg = model.forward_cfg(pos, edge_index, t, C, batch, cfg_weight=2.0)
    logger.info("CFG output shape: %s, mean: %.4f", noise_cfg.shape, noise_cfg.mean())
