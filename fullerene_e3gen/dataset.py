"""
Dataset loader for Fullerene structures.

Loads XYZ files with explicit 3-neighbor connectivity and prepares
PyTorch Geometric Data objects for training diffusion models.

Key features:
- Parses extended XYZ format (8 columns: element, x, y, z, index, nb1, nb2, nb3)
- Builds edge_index from neighbor information
- Normalizes coordinates (center + optional scale)
- Filters by carbon count range
- Stratified train/val/test split

Author: InterfaceML Project
Date: 2026-01-28
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Data

from synthetic_topology import SyntheticConfig, build_synthetic_dataset

logger = logging.getLogger(__name__)


class FullereneDataset(Dataset):
    """
    PyTorch Dataset for fullerene structures.
    
    Args:
        xyz_dir: Root directory containing C*/ folders with XYZ files
        split_csv: Path to split.csv (contains train/val/test assignments)
        split: One of 'train', 'val', 'test'
        C_range: Tuple (C_min, C_max) to filter carbon counts
        center: If True, center coordinates at origin
        normalize_scale: If True, normalize to unit variance
        transform: Optional PyG transform
    """
    
    def __init__(
        self,
        xyz_dir: str,
        split_csv: str,
        split: str = 'train',
        C_range: Optional[Tuple[int, int]] = None,
        center: bool = True,
        normalize_scale: bool = True,
        global_std: Optional[float] = None,
        transform=None,
    ):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        
        self.xyz_dir = Path(xyz_dir)
        self.split = split
        self.C_range = C_range
        self.center = center
        self.normalize_scale = normalize_scale
        self.transform = transform
        
        # Load split assignments
        self.data_list = self._load_split_list(split_csv)
        
        # Compute global normalization statistics for consistent scaling
        if normalize_scale and global_std is None:
            logger.info("[%s] Computing global normalization statistics...", split.upper())
            self.global_std = self._compute_global_std()
        else:
            self.global_std = global_std
        
        logger.info("[%s] Loaded %d structures", split.upper(), len(self.data_list))
        if C_range:
            logger.info("  C range: [%d, %d]", C_range[0], C_range[1])
        if self.normalize_scale and self.global_std is not None:
            logger.info("  Global std: %.4f (consistent scaling)", self.global_std)
    
    def _load_split_list(self, split_csv: str) -> List[Dict[str, str]]:
        """Load structures belonging to current split."""
        with open(split_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        # Filter by split
        split_rows = [r for r in all_rows if r['split'] == self.split]
        
        # Filter by C range if specified
        if self.C_range:
            C_min, C_max = self.C_range
            split_rows = [
                r for r in split_rows
                if C_min <= int(r['C']) <= C_max
            ]
        
        return split_rows
    
    def _parse_xyz(self, filepath: Path) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Parse extended XYZ file.
        
        Returns:
            coords: (N, 3) array of coordinates
            neighbors: List of (nb1, nb2, nb3) tuples (0-indexed)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        # Skip comment line (line 1)
        
        coords = np.zeros((n_atoms, 3), dtype=np.float32)
        neighbors = []
        
        for i in range(n_atoms):
            parts = lines[2 + i].split()
            # Format: Element x y z index nb1 nb2 nb3
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            nb1, nb2, nb3 = int(parts[5]), int(parts[6]), int(parts[7])
            
            coords[i] = [x, y, z]
            # Convert to 0-indexed
            neighbors.append((nb1 - 1, nb2 - 1, nb3 - 1))
        
        return coords, neighbors
    
    def _build_edge_index(self, neighbors: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Convert neighbor list to edge_index (bidirectional for MessagePassing).
        
        Args:
            neighbors: List of (nb1, nb2, nb3) for each atom
        
        Returns:
            edge_index: [2, E] tensor of edges (bidirectional)
        """
        edges = []
        seen = set()
        
        for u, (nb1, nb2, nb3) in enumerate(neighbors):
            for v in [nb1, nb2, nb3]:
                if v >= 0 and v != u:  # Valid neighbor
                    # Add both directions for MessagePassing
                    if (u, v) not in seen:
                        edges.append([u, v])
                        edges.append([v, u])
                        seen.add((u, v))
                        seen.add((v, u))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, E]
        return edge_index
    
    def _compute_global_std(self) -> float:
        """
        Dummy function for backward compatibility.
        
        With radius normalization, this returns 1.0 (not used).
        Kept to avoid breaking existing code that expects this attribute.
        
        Returns:
            1.0 (dummy value, radius normalization doesn't use global_std)
        """
        return 1.0  # Not used with radius normalization
    
    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        RADIUS NORMALIZATION: Scale molecules to unit sphere.
        
        This is the CORRECT approach for molecular diffusion models:
        - All molecules normalized to radius = 1.0
        - Bond lengths naturally scale: ~0.40 for C60 (1.42 Å / 3.5 Å radius)
        - Compatible with diffusion noise ~ N(0, I)
        - No scale mismatch between training and generation!
        
        Physical intuition:
        - C60: radius ~3.5 Å → normalized to 1.0
        - C-C bond ~1.42 Å → normalized to ~0.40
        - Diffusion noise std=1 matches normalized coords naturally
        """
        coords = coords.copy()
        
        if self.center:
            coords -= coords.mean(axis=0, keepdims=True)
        
        if self.normalize_scale:
            # RADIUS NORMALIZATION (standard for molecular diffusion)
            radii = np.linalg.norm(coords, axis=1)
            mean_radius = radii.mean()
            
            if mean_radius > 1e-6:
                coords /= mean_radius  # Scale to unit radius sphere
            
            # Store normalization factor for this structure (for reference)
            # In practice, we'll use typical radius for denormalization
        
        return coords
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single fullerene structure.
        
        Returns:
            PyG Data object with:
                - pos: [N, 3] coordinates (normalized)
                - edge_index: [2, E] edges
                - C: scalar (carbon count)
                - y: optional target (for supervised tasks)
        """
        row = self.data_list[idx]
        C = int(row['C'])
        filename = row['filename']
        
        # Load XYZ file
        filepath = self.xyz_dir / f"C{C}" / filename
        coords, neighbors = self._parse_xyz(filepath)
        
        # Normalize
        coords = self._normalize_coords(coords)
        
        # Build graph
        edge_index = self._build_edge_index(neighbors)
        
        # Create PyG Data object
        data = Data(
            pos=torch.tensor(coords, dtype=torch.float32),
            edge_index=edge_index,
            C=torch.tensor([C], dtype=torch.long),
            num_nodes=len(coords),
        )
        
        if self.transform:
            data = self.transform(data)
        
        return data


class ListDataset(Dataset):
    """Simple dataset wrapper for a pre-built list of Data objects."""

    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


def get_dataloaders(config: dict, num_workers: int = 4):
    """
    Create train/val/test dataloaders.
    
    Args:
        config: Configuration dictionary
        num_workers: Number of DataLoader workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch_geometric.loader import DataLoader
    
    data_config = config['data']
    train_config = config['training']
    
    C_range = (data_config['C_min'], data_config['C_max'])
    
    # Create train dataset first to compute global normalization statistics
    train_dataset = FullereneDataset(
        xyz_dir=data_config['xyz_dir'],
        split_csv=data_config['split_csv'],
        split='train',
        C_range=C_range,
        center=data_config['center'],
        normalize_scale=data_config['normalize_scale'],
        global_std=None,  # Compute from train set
    )
    
    # Share global_std with val/test for consistent scaling across splits
    global_std = train_dataset.global_std if train_dataset.normalize_scale else None
    
    val_dataset = FullereneDataset(
        xyz_dir=data_config['xyz_dir'],
        split_csv=data_config['split_csv'],
        split='val',
        C_range=C_range,
        center=data_config['center'],
        normalize_scale=data_config['normalize_scale'],
        global_std=global_std,  # Use train statistics
    )
    
    test_dataset = FullereneDataset(
        xyz_dir=data_config['xyz_dir'],
        split_csv=data_config['split_csv'],
        split='test',
        C_range=C_range,
        center=data_config['center'],
        normalize_scale=data_config['normalize_scale'],
        global_std=global_std,  # Use train statistics
    )
    
    # Create loaders
    # Optional synthetic topology augmentation (train-only by default)
    synth_cfg = data_config.get('synthetic_topology', {})
    if synth_cfg.get('enabled', False):
        apply_to = synth_cfg.get('apply_to', 'train')
        C_values = synth_cfg.get('C_values', [])
        per_C = int(synth_cfg.get('per_C', 1))
        max_tries = int(synth_cfg.get('max_tries', 200))
        seed = int(synth_cfg.get('seed', config.get('seed', 20260131)))
        refine = bool(synth_cfg.get('refine', True))

        if C_values:
            synthetic_data = build_synthetic_dataset(
                SyntheticConfig(
                    C_values=C_values,
                    per_C=per_C,
                    max_tries=max_tries,
                    seed=seed,
                    refine=refine,
                )
            )

            if synthetic_data:
                synthetic_dataset = ListDataset(synthetic_data)
                if apply_to in ('train', 'all'):
                    train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
                if apply_to in ('val', 'all'):
                    val_dataset = ConcatDataset([val_dataset, synthetic_dataset])
                if apply_to in ('test', 'all'):
                    test_dataset = ConcatDataset([test_dataset, synthetic_dataset])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Quick test of dataset loading."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = FullereneDataset(
        xyz_dir=config['data']['xyz_dir'],
        split_csv=config['data']['split_csv'],
        split='train',
        C_range=(config['data']['C_min'], config['data']['C_max']),
    )
    
    logger.info("Dataset size: %d", len(dataset))
    
    # Check first sample
    data = dataset[0]
    logger.info("Sample 0:")
    logger.info("  C = %d", data.C.item())
    logger.info("  Positions shape: %s", data.pos.shape)
    logger.info("  Edge index shape: %s", data.edge_index.shape)
    logger.info("  Mean position: %s", data.pos.mean(dim=0))  # Should be ~0 if centered
    logger.info("  Std position: %s", data.pos.std())  # Should be ~1 if normalized
