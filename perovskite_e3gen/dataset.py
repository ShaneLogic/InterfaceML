"""Dataset loader for perovskite crystal structures.

Loads CIF files via pymatgen, builds periodic radius graphs, and prepares
PyTorch Geometric Data objects for training the flow matching model.

Each Data object contains:
  - frac_coords: [N, 3] fractional coordinates
  - atom_types: [N] integer element indices
  - lattice_params: [6] = [a, b, c, alpha, beta, gamma]
  - num_atoms: int
  - composition: dict of element counts (metadata)
  - edge_index: [2, E] radius graph with PBC
  - edge_shift: [E, 3] lattice translation vectors for PBC edges

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from units import (
    element_to_index,
    lattice_params_to_matrix,
    LatticeScaler,
    lattice_params_to_vector,
)

logger = logging.getLogger(__name__)


def _parse_cif(cif_path: str, max_atoms: int = 60) -> Optional[dict]:
    """Parse a CIF file using pymatgen and extract structure data.

    Args:
        cif_path: Path to CIF file.
        max_atoms: Skip structures with more atoms than this.

    Returns:
        dict with frac_coords, atom_types, lattice_params, species, formula
        or None if parsing fails or structure is too large.
    """
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(cif_path)
    except Exception as e:
        logger.debug("Failed to parse %s: %s", cif_path, e)
        return None

    num_atoms = len(struct)
    if num_atoms > max_atoms or num_atoms < 2:
        return None

    # Fractional coordinates
    frac_coords = np.array([site.frac_coords for site in struct])

    # Element types
    species = [str(site.specie) for site in struct]
    atom_types = np.array([element_to_index(s) for s in species])

    # Check for unknown elements
    if 0 in atom_types:
        unknown = [s for s, t in zip(species, atom_types) if t == 0]
        logger.debug("Unknown elements in %s: %s", cif_path, unknown)
        return None

    # Lattice parameters
    lattice = struct.lattice
    lengths = np.array([lattice.a, lattice.b, lattice.c])
    angles = np.array([lattice.alpha, lattice.beta, lattice.gamma])

    # Composition
    comp = struct.composition.as_dict()

    return {
        "frac_coords": frac_coords.astype(np.float32),
        "atom_types": atom_types.astype(np.int64),
        "lattice_lengths": lengths.astype(np.float32),
        "lattice_angles": angles.astype(np.float32),
        "species": species,
        "formula": struct.composition.reduced_formula,
        "num_atoms": num_atoms,
        "composition": comp,
    }


def build_radius_graph_pbc(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float = 5.0,
    max_neighbors: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a radius graph with periodic boundary conditions.

    Uses minimum image convention, checking 27 periodic images (3^3).

    Args:
        frac_coords: [N, 3] fractional coordinates.
        lattice: [3, 3] lattice matrix (rows = lattice vectors).
        cutoff: Distance cutoff in Angstrom.
        max_neighbors: Max neighbors per atom.

    Returns:
        edge_index: [2, E] source-target pairs.
        edge_shift: [E, 3] fractional shift vectors for PBC.
    """
    num_atoms = frac_coords.size(0)

    # Generate all periodic image offsets: [-1, 0, 1]^3
    offsets = torch.tensor(
        [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
        dtype=frac_coords.dtype,
        device=frac_coords.device,
    )  # [27, 3]

    # Cartesian positions for all images
    cart_coords = frac_coords @ lattice  # [N, 3]

    src_list = []
    dst_list = []
    shift_list = []

    for offset in offsets:
        # Shifted fractional coords
        shifted_frac = frac_coords + offset.unsqueeze(0)  # [N, 3]
        shifted_cart = shifted_frac @ lattice  # [N, 3]

        # Pairwise distances: [N, N]
        diff = cart_coords.unsqueeze(1) - shifted_cart.unsqueeze(0)  # [N, N, 3]
        dist = diff.norm(dim=-1)  # [N, N]

        # Mask: within cutoff and not self-loop (when offset is 0)
        is_self = (offset.abs().sum() == 0)
        mask = dist < cutoff
        if is_self:
            mask = mask & ~torch.eye(num_atoms, dtype=torch.bool, device=dist.device)

        # Get edges
        src, dst = torch.where(mask)
        src_list.append(src)
        dst_list.append(dst)
        shift_list.append(offset.unsqueeze(0).expand(src.size(0), -1))

    if not src_list:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=frac_coords.device),
            torch.zeros(0, 3, dtype=frac_coords.dtype, device=frac_coords.device),
        )

    edge_src = torch.cat(src_list)
    edge_dst = torch.cat(dst_list)
    edge_shift = torch.cat(shift_list)

    # Limit neighbors per atom for memory efficiency
    if max_neighbors > 0 and edge_src.numel() > 0:
        # Compute distances for sorting
        cart_src = frac_coords[edge_src] @ lattice
        cart_dst = (frac_coords[edge_dst] + edge_shift) @ lattice
        dists = (cart_src - cart_dst).norm(dim=-1)

        # For each source atom, keep only closest max_neighbors
        keep = torch.ones(edge_src.size(0), dtype=torch.bool, device=edge_src.device)
        for i in range(num_atoms):
            atom_mask = edge_src == i
            if atom_mask.sum() > max_neighbors:
                atom_dists = dists[atom_mask]
                _, sorted_idx = atom_dists.sort()
                atom_indices = atom_mask.nonzero(as_tuple=True)[0]
                discard = atom_indices[sorted_idx[max_neighbors:]]
                keep[discard] = False

        edge_src = edge_src[keep]
        edge_dst = edge_dst[keep]
        edge_shift = edge_shift[keep]

    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    return edge_index, edge_shift


class PerovskiteDataset(Dataset):
    """PyTorch Dataset for perovskite crystal structures.

    Args:
        cif_dirs: List of directories containing CIF files.
        max_atoms: Maximum atoms per unit cell.
        cutoff: Radius cutoff for graph construction.
        lattice_scaler: Optional LatticeScaler for normalization.
    """

    def __init__(
        self,
        cif_dirs: list[str],
        max_atoms: int = 60,
        cutoff: float = 5.0,
        lattice_scaler: Optional[LatticeScaler] = None,
    ):
        self.max_atoms = max_atoms
        self.cutoff = cutoff
        self.lattice_scaler = lattice_scaler
        self.data_list: list[dict] = []

        # Load all CIF files
        for cif_dir in cif_dirs:
            cif_dir = Path(cif_dir)
            if not cif_dir.exists():
                logger.warning("CIF directory not found: %s", cif_dir)
                continue

            cif_files = sorted(cif_dir.glob("*.cif"))
            logger.info("Loading %d CIF files from %s", len(cif_files), cif_dir)

            for cif_path in cif_files:
                parsed = _parse_cif(str(cif_path), max_atoms=max_atoms)
                if parsed is not None:
                    self.data_list.append(parsed)

        logger.info("Loaded %d valid structures (max_atoms=%d)", len(self.data_list), max_atoms)

        # Compute lattice scaler if not provided
        if self.lattice_scaler is None and len(self.data_list) > 0:
            all_params = torch.stack([
                torch.tensor(np.concatenate([d["lattice_lengths"], d["lattice_angles"]]))
                for d in self.data_list
            ])
            self.lattice_scaler = LatticeScaler.from_dataset(all_params)
            logger.info("Lattice scaler: mean=%s, std=%s",
                        self.lattice_scaler.mean.tolist(),
                        self.lattice_scaler.std.tolist())

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        d = self.data_list[idx]

        frac_coords = torch.tensor(d["frac_coords"], dtype=torch.float32)
        atom_types = torch.tensor(d["atom_types"], dtype=torch.long)
        lengths = torch.tensor(d["lattice_lengths"], dtype=torch.float32)
        angles = torch.tensor(d["lattice_angles"], dtype=torch.float32)
        lattice_params = torch.cat([lengths, angles])  # [6]

        # Build lattice matrix for graph construction
        lattice_matrix = lattice_params_to_matrix(
            lengths.unsqueeze(0), angles.unsqueeze(0)
        ).squeeze(0)  # [3, 3]

        # Build radius graph with PBC
        edge_index, edge_shift = build_radius_graph_pbc(
            frac_coords, lattice_matrix, cutoff=self.cutoff
        )

        # Normalize lattice params
        lattice_params_norm = lattice_params.clone()
        if self.lattice_scaler is not None:
            lattice_params_norm = self.lattice_scaler.normalize(lattice_params)

        # Get unique elements for composition conditioning
        unique_types = torch.unique(atom_types)

        data = Data(
            frac_coords=frac_coords,           # [N, 3]
            atom_types=atom_types,              # [N]
            lattice_params=lattice_params,      # [6] raw
            lattice_params_norm=lattice_params_norm,  # [6] normalized
            lattice_matrix=lattice_matrix,      # [3, 3]
            edge_index=edge_index,              # [2, E]
            edge_shift=edge_shift,              # [E, 3]
            num_atoms=torch.tensor(d["num_atoms"], dtype=torch.long),
            num_nodes=d["num_atoms"],           # For PyG batching
        )

        return data


def get_dataloaders(
    config: dict,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from config.

    Args:
        config: Configuration dictionary.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 16)

    # Collect CIF directories
    cif_dirs = []
    mp_dir = data_cfg.get("mp_cif_dir")
    if mp_dir:
        cif_dirs.append(mp_dir)
    hoip_dir = data_cfg.get("hoip_cif_dir")
    if hoip_dir:
        cif_dirs.append(hoip_dir)

    # Load full dataset
    full_dataset = PerovskiteDataset(
        cif_dirs=cif_dirs,
        max_atoms=data_cfg.get("max_atoms", 60),
        cutoff=data_cfg.get("cutoff", 5.0),
    )

    if len(full_dataset) == 0:
        raise RuntimeError("No valid structures loaded! Check CIF directories.")

    # Split
    n = len(full_dataset)
    train_ratio = data_cfg.get("train_ratio", 0.8)
    val_ratio = data_cfg.get("val_ratio", 0.1)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Deterministic split
    seed = config.get("seed", 42)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create subset datasets sharing the lattice scaler
    train_data = [full_dataset[i] for i in train_indices.tolist()]
    val_data = [full_dataset[i] for i in val_indices.tolist()]
    test_data = [full_dataset[i] for i in test_indices.tolist()]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_data), len(val_data), len(test_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Quick test: load a few CIF files and print statistics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    base = Path(__file__).parent
    for key in ["mp_cif_dir", "hoip_cif_dir"]:
        if key in config["data"] and config["data"][key]:
            config["data"][key] = str(base / config["data"][key])

    train_loader, val_loader, test_loader = get_dataloaders(config, num_workers=0)
    logger.info("Train batches: %d, Val batches: %d, Test batches: %d",
                len(train_loader), len(val_loader), len(test_loader))

    # Print first batch
    for batch in train_loader:
        logger.info("Batch keys: %s", list(batch.keys()))
        logger.info("  frac_coords: %s", batch.frac_coords.shape)
        logger.info("  atom_types: %s", batch.atom_types.shape)
        logger.info("  lattice_params: %s", batch.lattice_params.shape)
        logger.info("  edge_index: %s", batch.edge_index.shape)
        logger.info("  batch: %s", batch.batch.shape)
        logger.info("  num_atoms: %s", batch.num_atoms)
        break
