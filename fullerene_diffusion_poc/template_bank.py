"""Template graph library for fullerene generation.

High-probability fix for unrealistic generations:
- Do NOT invent random 3-regular graphs at generation time.
- Reuse real fullerene adjacency (edge_index) from dataset XYZ templates.

This ensures:
- Degree-3 constraint holds by construction
- Topology resembles real fullerenes for a given carbon count

The diffusion model still generates coordinates; the template only supplies the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import random

import numpy as np
import torch
from torch_geometric.data import Data


def _parse_extended_xyz_1based(filepath: Path) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Parse the repo's extended XYZ format.

    Dataset files are 1-based indexed (see dataset/fullerenes/fullerene_xyz/C60/C60-Ih.xyz).

    Returns:
        coords: (N, 3)
        neighbors: list of 0-based neighbor triples (len N)
    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    neighbors: List[tuple[int, int, int]] = []

    for i in range(n_atoms):
        parts = lines[2 + i].split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        nb1, nb2, nb3 = int(parts[5]), int(parts[6]), int(parts[7])
        coords[i] = [x, y, z]
        neighbors.append((nb1 - 1, nb2 - 1, nb3 - 1))

    return coords, neighbors


def _neighbors_to_edge_index(neighbors: List[tuple[int, int, int]]) -> torch.Tensor:
    edges: list[list[int]] = []
    seen = set()
    for u, (nb1, nb2, nb3) in enumerate(neighbors):
        for v in (nb1, nb2, nb3):
            if v < 0 or v == u:
                continue
            if (u, v) in seen:
                continue
            edges.append([u, v])
            edges.append([v, u])
            seen.add((u, v))
            seen.add((v, u))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


@dataclass
class TemplateBank:
    xyz_dir: Path

    def __post_init__(self) -> None:
        self._cache: Dict[int, List[torch.Tensor]] = {}

    def _load_templates_for_C(self, C: int) -> List[torch.Tensor]:
        folder = self.xyz_dir / f"C{C}"
        if not folder.exists():
            raise FileNotFoundError(f"No folder for C{C}: {folder}")

        xyz_files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".xyz"])
        if not xyz_files:
            raise FileNotFoundError(f"No .xyz templates found in {folder}")

        templates: List[torch.Tensor] = []
        for fp in xyz_files:
            _, neighbors = _parse_extended_xyz_1based(fp)
            edge_index = _neighbors_to_edge_index(neighbors)
            templates.append(edge_index)

        return templates

    def sample_edge_index(self, C: int, *, device: torch.device) -> torch.Tensor:
        if C not in self._cache:
            self._cache[C] = self._load_templates_for_C(C)

        edge_index = random.choice(self._cache[C])
        return edge_index.to(device)

    def make_template_data(self, C: int, *, device: torch.device) -> Data:
        edge_index = self.sample_edge_index(C, device=device)
        # pos is just a placeholder; diffusion starts from noise anyway.
        pos = torch.randn(C, 3, device=device)
        return Data(pos=pos, edge_index=edge_index)
