"""
Local GNN-based refinement utilities for interface structures.

This module provides a lightweight, optional GNN that refines local
geometry near an interface (e.g., adsorbate-slab contact region).
It is designed to be safe by default: if no checkpoint is provided,
the model is initialized to produce near-zero displacements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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


class LocalRefinementGNN(nn.Module):
    """Simple message-passing network that predicts per-atom displacements."""

    def __init__(self, *, embed_dim: int, hidden_dim: int, num_layers: int, extra_dim: int = 2):
        super().__init__()
        self.embed = nn.Embedding(100, embed_dim)
        self.input_mlp = nn.Sequential(
            nn.Linear(embed_dim + extra_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # Safe default: start with zero displacement.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        node_extra: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(atomic_numbers)
        h = torch.cat([h, node_extra], dim=-1)
        h = self.input_mlp(h)

        row, col = edge_index
        for layer in self.layers:
            if edge_index.numel() == 0:
                agg = torch.zeros_like(h)
            else:
                rel = pos[row] - pos[col]
                dist = torch.norm(rel, dim=-1, keepdim=True)
                edge_input = torch.cat([h[row], h[col], dist], dim=-1)
                messages = self.edge_mlp(edge_input)
                agg = torch.zeros_like(h)
                agg.index_add_(0, col, messages)
            h = h + layer(torch.cat([h, agg], dim=-1))

        delta = self.delta_mlp(h)
        return delta


class LocalRefiner:
    """Wrapper for optional local GNN refinement."""

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
        """Refine coordinates near the interface and return a new Structure."""
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

        edge_index = _build_radius_graph(pos, cutoff=self.config.cutoff)

        with torch.no_grad():
            delta = self.model(atomic_numbers, pos, edge_index, node_extra)
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


def _build_radius_graph(pos: torch.Tensor, *, cutoff: float) -> torch.Tensor:
    """Build a simple radius graph (fully connected within cutoff)."""
    if pos.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    dist = torch.cdist(pos, pos)
    mask = (dist < float(cutoff)) & (dist > 1e-6)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index
