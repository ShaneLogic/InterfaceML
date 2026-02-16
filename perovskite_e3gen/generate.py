"""Generation script for perovskite crystal structures.

Samples new perovskite structures using a trained flow matching model,
conditioned on a target composition (A-site, B-site, X-site elements).

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt \
        --A Ba --B Ti --X O --num_atoms 5 --num_samples 10

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from model import PerovskitePaiNNModel
from flow_matching import CrystalFlowMatcher
from dataset import build_radius_graph_pbc
from units import (
    element_to_index,
    index_to_element,
    LatticeScaler,
    lattice_params_to_matrix,
    lattice_matrix_to_params,
    wrap_frac_coords,
)

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    use_ema: bool = True,
) -> tuple[PerovskitePaiNNModel, dict, Optional[LatticeScaler]]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
        use_ema: Use EMA weights (recommended for generation).

    Returns:
        model, config, lattice_scaler
    """
    state = torch.load(checkpoint_path, map_location=device)
    config = state["config"]
    model_cfg = config["model"]

    model = PerovskitePaiNNModel(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 6),
        num_rbf=model_cfg.get("num_rbf", 20),
        cutoff=model_cfg.get("cutoff", 5.0),
        time_embed_dim=model_cfg.get("time_embed_dim", 128),
        num_elements=model_cfg.get("num_elements", 100),
        element_embed_dim=model_cfg.get("element_embed_dim", 64),
        composition_embed_dim=model_cfg.get("composition_embed_dim", 128),
    ).to(device)

    if use_ema and "ema_state" in state:
        model.load_state_dict(state["ema_state"])
        logger.info("Loaded EMA model weights")
    else:
        model.load_state_dict(state["model_state"])
        logger.info("Loaded model weights")

    model.eval()

    lattice_scaler = None
    if "lattice_scaler" in state:
        lattice_scaler = LatticeScaler.from_state_dict(state["lattice_scaler"])

    return model, config, lattice_scaler


def build_composition(
    A: str, B: str, X: str, num_atoms: int = 5,
) -> torch.Tensor:
    """Build atom type tensor for ABX3 composition.

    For num_atoms=5: 1 A-site + 1 B-site + 3 X-site atoms.
    For num_atoms=10: 2A + 2B + 6X (double perovskite cell).
    For num_atoms=20: 4A + 4B + 12X (2x2x2 supercell).

    Args:
        A: A-site element symbol.
        B: B-site element symbol.
        X: X-site (anion) element symbol.
        num_atoms: Total atoms in unit cell.

    Returns:
        [N] tensor of atom type indices.
    """
    a_idx = element_to_index(A)
    b_idx = element_to_index(B)
    x_idx = element_to_index(X)

    if a_idx == 0 or b_idx == 0 or x_idx == 0:
        raise ValueError(f"Unknown element: A={A}({a_idx}), B={B}({b_idx}), X={X}({x_idx})")

    # ABX3 stoichiometry: ratio 1:1:3
    formula_atoms = 5  # 1 + 1 + 3
    num_formulas = max(1, num_atoms // formula_atoms)

    types = []
    for _ in range(num_formulas):
        types.extend([a_idx, b_idx, x_idx, x_idx, x_idx])

    # Truncate or pad if needed
    types = types[:num_atoms]
    while len(types) < num_atoms:
        types.append(x_idx)  # Pad with anion

    return torch.tensor(types, dtype=torch.long)


@torch.no_grad()
def generate_structure(
    model: PerovskitePaiNNModel,
    atom_types: torch.Tensor,
    lattice_scaler: Optional[LatticeScaler] = None,
    num_steps: int = 50,
    device: str = "cpu",
) -> dict:
    """Generate a single perovskite structure.

    Args:
        model: Trained model.
        atom_types: [N] atom type indices for target composition.
        lattice_scaler: For denormalizing lattice parameters.
        num_steps: Flow matching ODE steps.
        device: Device.

    Returns:
        dict with frac_coords, lattice_params, species, lattice_matrix
    """
    model.eval()
    num_atoms = atom_types.size(0)
    atom_types = atom_types.to(device)
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

    # Initialize from noise
    frac_t = torch.rand(num_atoms, 3, device=device)
    lattice_t = torch.randn(1, 6, device=device)

    dt = 1.0 / num_steps

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_batch = torch.full((1,), t_val, device=device)

        # Build lattice matrix for graph construction
        # During generation, use a default cubic lattice initially,
        # then transition to predicted lattice
        if lattice_scaler is not None:
            lattice_params_raw = lattice_scaler.denormalize(lattice_t.squeeze(0))
        else:
            lattice_params_raw = lattice_t.squeeze(0).clone()
            # Ensure positive lengths
            lattice_params_raw[:3] = lattice_params_raw[:3].abs().clamp(min=2.0)
            # Ensure reasonable angles
            lattice_params_raw[3:] = lattice_params_raw[3:].clamp(min=60.0, max=120.0)

        lengths = lattice_params_raw[:3].unsqueeze(0)
        angles = lattice_params_raw[3:].unsqueeze(0)
        lattice_matrix = lattice_params_to_matrix(lengths, angles)  # [1, 3, 3]

        # Build radius graph with PBC
        edge_index, edge_shift = build_radius_graph_pbc(
            frac_t, lattice_matrix.squeeze(0), cutoff=model.cutoff
        )

        # Ensure edges exist (fallback to fully connected if graph is empty)
        if edge_index.size(1) == 0:
            edges = []
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        edges.append([i, j])
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            edge_shift = torch.zeros(edge_index.size(1), 3, device=device)

        # Model forward
        coord_vel, lattice_vel, type_logits = model(
            frac_t, atom_types, lattice_matrix, t_batch,
            edge_index, edge_shift, batch,
        )

        # Euler steps
        frac_t = wrap_frac_coords(frac_t - dt * coord_vel)
        lattice_t = lattice_t - dt * lattice_vel

        if torch.isnan(frac_t).any() or torch.isnan(lattice_t).any():
            logger.warning("NaN at step %d/%d — aborting", step, num_steps)
            break

    # Post-processing
    frac_coords = wrap_frac_coords(frac_t)

    # Denormalize lattice
    if lattice_scaler is not None:
        lattice_params_final = lattice_scaler.denormalize(lattice_t.squeeze(0))
    else:
        lattice_params_final = lattice_t.squeeze(0)

    # Clamp to physical values
    lattice_params_final[:3] = lattice_params_final[:3].abs().clamp(min=2.0, max=30.0)
    lattice_params_final[3:] = lattice_params_final[3:].clamp(min=30.0, max=170.0)

    # Build final lattice matrix
    lengths_f = lattice_params_final[:3].unsqueeze(0)
    angles_f = lattice_params_final[3:].unsqueeze(0)
    lattice_matrix_final = lattice_params_to_matrix(lengths_f, angles_f).squeeze(0)

    # Resolve species
    species = [index_to_element(idx.item()) for idx in atom_types]

    return {
        "frac_coords": frac_coords.cpu().numpy(),
        "lattice_params": lattice_params_final.cpu().numpy(),
        "lattice_matrix": lattice_matrix_final.cpu().numpy(),
        "species": species,
        "atom_types": atom_types.cpu().numpy(),
        "num_atoms": num_atoms,
    }


def write_cif(structure: dict, filepath: str):
    """Write structure to CIF file.

    Args:
        structure: dict from generate_structure().
        filepath: Output CIF file path.
    """
    try:
        from pymatgen.core import Structure, Lattice
        lattice = Lattice.from_parameters(*structure["lattice_params"])
        struct = Structure(
            lattice, structure["species"], structure["frac_coords"],
            coords_are_cartesian=False,
        )
        struct.to(filename=filepath)
        logger.info("Wrote CIF: %s", filepath)
    except ImportError:
        # Fallback: write minimal CIF manually
        _write_cif_manual(structure, filepath)


def write_poscar(structure: dict, filepath: str):
    """Write structure to POSCAR file."""
    from pymatgen.core import Structure, Lattice
    lattice = Lattice.from_parameters(*structure["lattice_params"])
    struct = Structure(
        lattice, structure["species"], structure["frac_coords"],
        coords_are_cartesian=False,
    )
    struct.to(filename=filepath, fmt="poscar")
    logger.info("Wrote POSCAR: %s", filepath)


def _write_cif_manual(structure: dict, filepath: str):
    """Write minimal CIF without pymatgen."""
    lp = structure["lattice_params"]
    lines = [
        "data_generated",
        f"_cell_length_a   {lp[0]:.4f}",
        f"_cell_length_b   {lp[1]:.4f}",
        f"_cell_length_c   {lp[2]:.4f}",
        f"_cell_angle_alpha   {lp[3]:.4f}",
        f"_cell_angle_beta    {lp[4]:.4f}",
        f"_cell_angle_gamma   {lp[5]:.4f}",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number   1",
        "loop_",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
    ]
    for sp, fc in zip(structure["species"], structure["frac_coords"]):
        lines.append(f"  {sp}  {fc[0]:.6f}  {fc[1]:.6f}  {fc[2]:.6f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote CIF (manual): %s", filepath)


def generate_samples(
    checkpoint_path: str,
    A: str, B: str, X: str,
    num_atoms: int = 5,
    num_samples: int = 10,
    num_steps: int = 50,
    output_dir: str = "generated",
    output_format: str = "cif",
    device: str = "auto",
    seed: Optional[int] = None,
) -> list[dict]:
    """Generate multiple perovskite structures.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        A, B, X: Element symbols for A-site, B-site, anion.
        num_atoms: Atoms per unit cell.
        num_samples: Number of structures to generate.
        num_steps: Flow matching ODE steps.
        output_dir: Directory for output files.
        output_format: "cif", "poscar", or "xyz".
        device: Device string.
        seed: Random seed.

    Returns:
        List of structure dicts.
    """
    from train import get_device
    device = get_device(device)

    if seed is not None:
        torch.manual_seed(seed)

    model, config, lattice_scaler = load_model(checkpoint_path, device)
    atom_types = build_composition(A, B, X, num_atoms)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    structures = []
    formula = f"{A}{B}{X}3"

    for i in range(num_samples):
        logger.info("Generating sample %d/%d: %s (%d atoms)", i + 1, num_samples, formula, num_atoms)

        struct = generate_structure(
            model, atom_types, lattice_scaler,
            num_steps=num_steps, device=device,
        )
        structures.append(struct)

        # Write output
        filename = f"{formula}_sample_{i:03d}"
        if output_format == "cif":
            write_cif(struct, str(output_path / f"{filename}.cif"))
        elif output_format == "poscar":
            write_poscar(struct, str(output_path / f"{filename}.vasp"))
        else:
            write_cif(struct, str(output_path / f"{filename}.cif"))

    logger.info("Generated %d structures in %s", num_samples, output_path)
    return structures


def main():
    parser = argparse.ArgumentParser(description="Generate perovskite structures")
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint")
    parser.add_argument("--A", required=True, help="A-site element (e.g., Ba, Cs)")
    parser.add_argument("--B", required=True, help="B-site element (e.g., Ti, Pb)")
    parser.add_argument("--X", required=True, help="Anion element (e.g., O, I)")
    parser.add_argument("--num_atoms", type=int, default=5, help="Atoms per unit cell")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of structures")
    parser.add_argument("--output_dir", default="generated_perovskites", help="Output directory")
    parser.add_argument("--sampling_steps", type=int, default=50, help="Flow ODE steps")
    parser.add_argument("--output_format", default="cif", choices=["cif", "poscar"],
                        help="Output format")
    parser.add_argument("--device", default="auto", help="Device: auto/cuda/mps/cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    generate_samples(
        checkpoint_path=args.checkpoint,
        A=args.A, B=args.B, X=args.X,
        num_atoms=args.num_atoms,
        num_samples=args.num_samples,
        num_steps=args.sampling_steps,
        output_dir=args.output_dir,
        output_format=args.output_format,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
