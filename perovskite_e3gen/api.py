"""API Interface for Perovskite E3Gen model.

Provides a clean programmatic interface for model loading, structure generation,
and evaluation. Designed for integration with web applications and pipelines.

Usage:
    from perovskite_e3gen.api import PerovskiteAPI

    api = PerovskiteAPI("checkpoints/best_model.pt")
    structures = api.generate(A="Ba", B="Ti", X="O", num_samples=10)
    metrics = api.evaluate(structures[0])

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from generate import load_model, build_composition, generate_structure, write_cif, write_poscar
from evaluate import check_validity, check_composition_match, compute_tolerance_factor
from units import LatticeScaler

logger = logging.getLogger(__name__)


class PerovskiteAPI:
    """Main API class for perovskite crystal generation.

    Thread-safe, designed for production use.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        use_ema: bool = True,
    ):
        """Initialize API.

        Args:
            checkpoint_path: Path to trained model checkpoint.
            device: Device ('cuda', 'mps', 'cpu', or None for auto).
            use_ema: Use EMA model weights (recommended).
        """
        if device is None:
            from train import get_device
            device = get_device("auto")

        self.device = device
        self.model = None
        self.config = None
        self.lattice_scaler = None
        self.use_ema = use_ema

        if checkpoint_path:
            self.load_model(checkpoint_path)

        logger.info("PerovskiteAPI initialized on device: %s", self.device)

    def load_model(self, checkpoint_path: str) -> Dict:
        """Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            dict with model info (num_params, config, etc.)
        """
        self.model, self.config, self.lattice_scaler = load_model(
            checkpoint_path, device=self.device, use_ema=self.use_ema
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        info = {
            "num_params": num_params,
            "device": self.device,
            "config": self.config,
            "has_lattice_scaler": self.lattice_scaler is not None,
        }
        logger.info("Model loaded: %d parameters on %s", num_params, self.device)
        return info

    def generate(
        self,
        A: str,
        B: str,
        X: str,
        num_atoms: int = 5,
        num_samples: int = 10,
        num_steps: int = 50,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """Generate perovskite structures.

        Args:
            A: A-site element symbol (e.g., "Ba", "Cs").
            B: B-site element symbol (e.g., "Ti", "Pb").
            X: Anion element symbol (e.g., "O", "I").
            num_atoms: Atoms per unit cell (default 5 for ABX3).
            num_samples: Number of structures to generate.
            num_steps: Flow matching ODE steps.
            seed: Random seed for reproducibility.

        Returns:
            List of structure dicts.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if seed is not None:
            torch.manual_seed(seed)

        atom_types = build_composition(A, B, X, num_atoms)
        structures = []

        for i in range(num_samples):
            struct = generate_structure(
                self.model, atom_types, self.lattice_scaler,
                num_steps=num_steps, device=self.device,
            )
            struct["sample_id"] = i
            struct["formula"] = f"{A}{B}{X}3"
            structures.append(struct)
            logger.info("Generated sample %d/%d: %s", i + 1, num_samples, struct["formula"])

        return structures

    def evaluate(self, structure: Dict) -> Dict:
        """Evaluate a single generated structure.

        Args:
            structure: dict from generate().

        Returns:
            dict of quality metrics.
        """
        metrics = {}

        # Validity check
        validity = check_validity(structure)
        metrics["is_valid"] = validity["is_valid"]
        metrics["min_distance"] = validity["min_distance"]
        if validity["issues"]:
            metrics["issues"] = validity["issues"]

        # Tolerance factor
        tol = compute_tolerance_factor(structure)
        if tol is not None:
            metrics["tolerance_factor"] = tol
            metrics["tolerance_in_range"] = 0.8 <= tol <= 1.0

        # Lattice info
        lp = structure["lattice_params"]
        metrics["lattice_params"] = {
            "a": float(lp[0]), "b": float(lp[1]), "c": float(lp[2]),
            "alpha": float(lp[3]), "beta": float(lp[4]), "gamma": float(lp[5]),
        }

        return metrics

    def save_structure(
        self,
        structure: Dict,
        filepath: str,
        fmt: str = "cif",
    ):
        """Save structure to file.

        Args:
            structure: dict from generate().
            filepath: Output file path.
            fmt: Format - "cif" or "poscar".
        """
        if fmt == "cif":
            write_cif(structure, filepath)
        elif fmt == "poscar":
            write_poscar(structure, filepath)
        else:
            raise ValueError(f"Unknown format: {fmt}. Use 'cif' or 'poscar'.")

    def batch_evaluate(self, structures: List[Dict]) -> Dict:
        """Evaluate a batch of structures and compute aggregate metrics.

        Args:
            structures: List of structure dicts.

        Returns:
            dict of aggregate metrics.
        """
        import numpy as np

        individual = [self.evaluate(s) for s in structures]

        valid_count = sum(1 for m in individual if m["is_valid"])
        min_dists = [m["min_distance"] for m in individual
                     if m["min_distance"] < float("inf")]
        tolerances = [m["tolerance_factor"] for m in individual
                      if "tolerance_factor" in m]

        aggregate = {
            "num_structures": len(structures),
            "validity_rate": valid_count / max(1, len(structures)),
            "avg_min_distance": float(np.mean(min_dists)) if min_dists else None,
        }

        if tolerances:
            aggregate["tolerance_mean"] = float(np.mean(tolerances))
            aggregate["tolerance_std"] = float(np.std(tolerances))
            aggregate["tolerance_in_range_rate"] = float(np.mean(
                [0.8 <= t <= 1.0 for t in tolerances]
            ))

        return aggregate
