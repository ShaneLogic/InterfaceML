"""
Local GNN-based refinement utilities for interface structures.

This module provides an SE(3)-equivariant GNN that refines local
geometry near an interface (e.g., adsorbate-slab contact region).
It is designed to be safe by default: if no checkpoint is provided,
the model is initialized to produce near-zero displacements.

Architecture overview
---------------------
- **Invariant scalar channel**: node embeddings h ∈ ℝ^H are updated via
  message-passing that uses *scalar* distance and learned radial basis
  functions — fully rotation-invariant.
- **Equivariant vector channel**: per-node vector features v ∈ ℝ^{H×3}
  carry directional information.  Messages are formed by weighting the
  unit direction vectors r̂_ij with invariant edge scalars, preserving
  SO(3) equivariance (EGNN / PaiNN-style).
- **Displacement output**: the final Δx is a *linear* contraction of the
  vector channel (sum over hidden dim), so Δx transforms as a proper
  vector under rotation — making the whole refinement SE(3)-equivariant.

PBC-aware edge construction
---------------------------
When a ``pymatgen.core.Structure`` with a valid lattice is available,
edges are built using ``Structure.get_neighbor_list`` which correctly
handles periodic boundary conditions and minimum-image convention.
A fallback ``torch.cdist`` path is kept for non-periodic / Cartesian
inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime environment dependent
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from pymatgen.core import Structure


@dataclass(frozen=True)
class LocalRefinerConfig:
    """Configuration for the local GNN refiner."""

    cutoff: float = 5.0
    embed_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    max_displacement: float = 0.2
    interface_window: float = 4.0
    num_rbf: int = 20  # radial basis expansion dimension


if nn is None:
    class LocalRefinementGNN:  # type: ignore[too-few-public-methods]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(f"PyTorch is required for LocalRefinementGNN: {_TORCH_IMPORT_ERROR}")
else:
    # -----------------------------------------------------------------
    # Helper: sinusoidal radial basis functions (Gaussian-free, stable)
    # -----------------------------------------------------------------
    class RadialBasis(nn.Module):
        """Sinusoidal radial basis expansion for interatomic distances.

        Maps scalar distance *d* → ℝ^K via  sin(n π d / cutoff) / d ,
        where n = 1 … K.  This is smooth, bounded, and goes to zero at
        the cutoff — no extra envelope needed.
        """

        def __init__(self, num_rbf: int, cutoff: float):
            super().__init__()
            self.num_rbf = num_rbf
            self.cutoff = cutoff
            # pre-compute frequencies  n·π / cutoff
            freq = torch.arange(1, num_rbf + 1, dtype=torch.float32) * math.pi / cutoff
            self.register_buffer("freq", freq)

        def forward(self, dist: torch.Tensor) -> torch.Tensor:
            """dist: (E,) → (E, num_rbf)"""
            d = dist.unsqueeze(-1).clamp(min=1e-8)  # (E, 1)
            return (self.freq * d).sin() / d  # (E, K)

    # -----------------------------------------------------------------
    # SE(3)-equivariant message-passing GNN
    # -----------------------------------------------------------------
    class EquivariantInteractionBlock(nn.Module):
        """One layer of SE(3)-equivariant message passing (PaiNN-style).

        Invariant update:
            m_ij  = MLP_edge([h_i, h_j, RBF(d_ij)])
            h_i  ← h_i + MLP_node(h_i || agg_j m_ij)

        Equivariant update:
            v_i  ← v_i + Σ_j  φ(m_ij) ⊗ r̂_ij          (outer broadcast)
        where φ is a learned scalar-to-vector gate and r̂_ij is the unit
        direction vector — the whole expression transforms equivariantly.
        """

        def __init__(self, hidden_dim: int, num_rbf: int):
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # scalar gate for vector channel (produces per-hidden-dim weight)
            self.vec_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(
            self,
            h: torch.Tensor,          # (N, H) — invariant node scalars
            v: torch.Tensor,          # (N, H, 3) — equivariant node vectors
            edge_index: torch.Tensor,  # (2, E)
            rbf: torch.Tensor,         # (E, K) — radial basis values
            direction: torch.Tensor,   # (E, 3) — unit direction vectors r̂_ij
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if edge_index.numel() == 0:
                return h, v

            row, col = edge_index  # row → source, col → target (messages TO col)

            # --- invariant messages ---
            edge_input = torch.cat([h[row], h[col], rbf], dim=-1)
            msg = self.edge_mlp(edge_input)  # (E, H)

            agg = torch.zeros_like(h)
            agg.index_add_(0, col, msg)
            h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))

            # --- equivariant vector update ---
            gate = self.vec_gate(msg)  # (E, H) — scalar weights per edge
            # gate[:, :, None] * direction[None]  →  (E, H, 3)
            vec_msg = gate.unsqueeze(-1) * direction.unsqueeze(1)  # (E, H, 3)
            vec_agg = torch.zeros_like(v)
            # scatter-add vector messages to target nodes
            vec_agg.index_add_(0, col.unsqueeze(-1).unsqueeze(-1).expand_as(vec_msg)[:, 0, 0],
                               vec_msg)
            # Simpler scatter for 3D: reshape, add, reshape back
            N, H = v.shape[:2]
            vec_agg_flat = torch.zeros(N, H * 3, device=v.device, dtype=v.dtype)
            vec_msg_flat = vec_msg.reshape(-1, H * 3)
            vec_agg_flat.index_add_(0, col, vec_msg_flat)
            v_out = v + vec_agg_flat.reshape(N, H, 3)

            return h_out, v_out

    class LocalRefinementGNN(nn.Module):
        """SE(3)-equivariant message-passing GNN for per-atom displacement
        prediction at material interfaces.

        The model maintains two channels:
        - **Scalar** h ∈ ℝ^H  (invariant under rotation)
        - **Vector** v ∈ ℝ^{H×3} (equivariant — rotates with the system)

        The final displacement Δx ∈ ℝ^3 is read out from the vector
        channel via a learned contraction, guaranteeing that Δx rotates
        consistently with the atomic coordinates.
        """

        def __init__(
            self,
            *,
            embed_dim: int,
            hidden_dim: int,
            num_layers: int,
            extra_dim: int = 2,
            num_rbf: int = 20,
            cutoff: float = 5.0,
        ):
            super().__init__()
            self.cutoff = cutoff

            # --- invariant node embedding ---
            self.embed = nn.Embedding(100, embed_dim)
            self.input_mlp = nn.Sequential(
                nn.Linear(embed_dim + extra_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # --- radial basis ---
            self.rbf = RadialBasis(num_rbf, cutoff)

            # --- SE(3)-equivariant interaction layers ---
            self.layers = nn.ModuleList(
                [EquivariantInteractionBlock(hidden_dim, num_rbf) for _ in range(num_layers)]
            )

            # --- readout: contract vector channel → Δx ∈ ℝ^3 ---
            # We use a scalar gating MLP on h to weight the vector channel,
            # then sum over hidden dim → 3D displacement.
            self.readout_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Safe default: zero-initialize readout so untrained model
            # produces near-zero displacements.
            nn.init.zeros_(self.readout_gate[-1].weight)
            nn.init.zeros_(self.readout_gate[-1].bias)

        def forward(
            self,
            atomic_numbers: torch.Tensor,  # (N,) long
            pos: torch.Tensor,             # (N, 3) float
            edge_index: torch.Tensor,      # (2, E) long
            node_extra: torch.Tensor,      # (N, extra_dim)
            edge_diff: Optional[torch.Tensor] = None,  # (E, 3) — PBC-corrected vectors
        ) -> torch.Tensor:
            """Predict per-atom displacements Δx.

            Parameters
            ----------
            edge_diff : optional (E, 3)
                If provided (e.g. from PBC-aware neighbor list), these are
                used as the direction vectors instead of ``pos[row]-pos[col]``.
            """
            # --- build invariant node features ---
            h = self.embed(atomic_numbers)
            h = torch.cat([h, node_extra], dim=-1)
            h = self.input_mlp(h)

            # --- initialise equivariant vector channel to zero ---
            N, H = h.shape
            v = torch.zeros(N, H, 3, device=h.device, dtype=h.dtype)

            # --- edge geometry ---
            if edge_index.numel() > 0:
                row, col = edge_index
                if edge_diff is not None:
                    rel = edge_diff
                else:
                    rel = pos[row] - pos[col]
                dist = torch.norm(rel, dim=-1).clamp(min=1e-8)
                direction = rel / dist.unsqueeze(-1)
                rbf = self.rbf(dist)
            else:
                rbf = torch.empty(0, self.rbf.num_rbf, device=h.device)
                direction = torch.empty(0, 3, device=h.device)

            # --- message-passing layers ---
            for layer in self.layers:
                h, v = layer(h, v, edge_index, rbf, direction)

            # --- equivariant readout ---
            gate = self.readout_gate(h)  # (N, H) — scalar weights
            # Contract: Δx_i = Σ_d  gate_{i,d} · v_{i,d,:}   →  (N, 3)
            delta = (gate.unsqueeze(-1) * v).sum(dim=1)  # (N, 3)
            return delta


class LocalRefiner:
    """Wrapper for optional local GNN refinement.

    Supports PBC-aware edge construction when a ``pymatgen.Structure``
    with a real lattice is provided (Optimization 2).
    """

    def __init__(
        self,
        *,
        config: Optional[LocalRefinerConfig] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(f"PyTorch is required for LocalRefiner: {_TORCH_IMPORT_ERROR}")
        assert torch is not None

        self.config = config or LocalRefinerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LocalRefinementGNN(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_rbf=self.config.num_rbf,
            cutoff=self.config.cutoff,
        ).to(self.device)
        self.trained = False

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
                self.trained = True
            elif isinstance(state, dict):
                self.model.load_state_dict(state)
                self.trained = True
            else:
                raise RuntimeError("Unsupported checkpoint format for LocalRefiner.")

        self.model.eval()

    def refine_structure(
        self,
        structure: Structure,
        *,
        adsorbate_indices: Sequence[int],
        interface_normal: np.ndarray,
        refine_scope: str = "adsorbate",
    ) -> Structure:
        """Refine coordinates near the interface and return a new Structure.

        Uses PBC-aware neighbor list from pymatgen when the structure has
        a valid periodic lattice, falling back to the non-periodic path
        for molecular / large-box inputs.
        """
        assert torch is not None

        if len(structure) == 0:
            return structure.copy()

        coords = np.asarray(structure.cart_coords, dtype=float)
        pos = torch.tensor(coords, dtype=torch.float32, device=self.device)

        ads_mask = torch.zeros(len(structure), dtype=torch.bool, device=self.device)
        if adsorbate_indices:
            ads_mask[list(adsorbate_indices)] = True
        slab_mask = ~ads_mask

        normal = np.asarray(interface_normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        z = torch.tensor(coords @ normal, dtype=torch.float32, device=self.device)

        if slab_mask.any() and ads_mask.any():
            slab_max = z[slab_mask].max()
            ads_min = z[ads_mask].min()
            interface_z = 0.5 * (slab_max + ads_min)
        else:
            interface_z = z.mean()

        dist_to_interface = z - interface_z
        interface_window = max(float(self.config.interface_window), 1e-6)
        local_mask = dist_to_interface.abs() <= interface_window

        if refine_scope == "all":
            refine_mask = torch.ones_like(ads_mask)
        elif refine_scope == "interface":
            refine_mask = local_mask
        else:
            refine_mask = ads_mask

        node_extra = torch.stack(
            [
                ads_mask.float(),
                (dist_to_interface / interface_window).clamp(-2.0, 2.0),
            ],
            dim=-1,
        )

        atomic_numbers = torch.tensor(
            [site.specie.Z for site in structure],
            dtype=torch.long,
            device=self.device,
        )

        # Build PBC-aware or non-periodic edge graph
        edge_index, edge_diff = _build_radius_graph(
            pos, cutoff=self.config.cutoff, structure=structure, device=self.device,
        )

        with torch.no_grad():
            delta = self.model(atomic_numbers, pos, edge_index, node_extra, edge_diff=edge_diff)
            max_disp = float(self.config.max_displacement)
            if max_disp > 0:
                delta = delta.clamp(-max_disp, max_disp)
            delta = delta * refine_mask.float().unsqueeze(-1)
            pos_refined = pos + delta

        coords_refined = pos_refined.detach().cpu().numpy()
        refined = Structure(
            structure.lattice,
            structure.species,
            coords_refined,
            coords_are_cartesian=True,
            to_unit_cell=False,
            site_properties=structure.site_properties,
        )
        return refined


def _build_radius_graph(
    pos: torch.Tensor,
    *,
    cutoff: float,
    structure: Optional[Structure] = None,
    device: Optional[object] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build a radius graph, with PBC support when a Structure is available.

    Parameters
    ----------
    pos : (N, 3) tensor
        Cartesian coordinates.
    cutoff : float
        Distance cutoff for neighbor search.
    structure : pymatgen Structure, optional
        If provided *and* the lattice is not trivially large (box > 500 Å),
        ``Structure.get_neighbor_list`` is used to correctly handle periodic
        boundary conditions.  Otherwise, falls back to ``torch.cdist``.
    device : torch device, optional

    Returns
    -------
    edge_index : (2, E) long tensor
    edge_diff  : (E, 3) float tensor or None
        PBC-corrected displacement vectors ``r_j - r_i`` (when available).
        This should be passed to the model as ``edge_diff`` so that
        directional information respects minimum-image convention.
    """
    dev = device or pos.device

    if pos.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=dev),
            None,
        )

    # --- PBC-aware path via pymatgen ---
    if structure is not None and len(structure) == pos.shape[0]:
        # Heuristic: skip PBC for very large boxes (molecular in box)
        lattice_params = structure.lattice.abc
        if all(p < 500.0 for p in lattice_params):
            try:
                center_indices, neighbor_indices, images, distances = structure.get_neighbor_list(
                    r=float(cutoff)
                )
                # center_indices[e] → i,  neighbor_indices[e] → j,
                # images[e] → periodic image offset,  distances[e] → |r_j + T - r_i|
                # Compute PBC-corrected displacement vectors
                cart_coords = np.asarray(structure.cart_coords, dtype=np.float64)
                lattice_matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
                # r_j + T - r_i
                diff = (
                    cart_coords[neighbor_indices]
                    + images @ lattice_matrix
                    - cart_coords[center_indices]
                )

                edge_index = torch.tensor(
                    np.stack([center_indices, neighbor_indices], axis=0),
                    dtype=torch.long,
                    device=dev,
                )
                edge_diff = torch.tensor(diff, dtype=torch.float32, device=dev)
                return edge_index, edge_diff
            except Exception:
                pass  # fall through to non-periodic path

    # --- Non-periodic fallback ---
    dist = torch.cdist(pos, pos)
    mask = (dist < float(cutoff)) & (dist > 1e-6)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index, None
