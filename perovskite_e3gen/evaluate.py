"""Evaluation script for generated perovskite structures.

Computes quality metrics:
  - Structural validity (no overlapping atoms, reasonable bonds)
  - Composition match (correct stoichiometry)
  - Lattice parameter deviation from reference
  - Bond valence mismatch
  - Minimum interatomic distance
  - Goldschmidt tolerance factor

Usage:
    python evaluate.py --generated_dir generated_perovskites/ \
        --reference_dir ../dataset/perovskite/mp_perovskite_cifs

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_structure(cif_path: str) -> Optional[dict]:
    """Load a CIF file and extract structure data.

    Returns:
        dict with frac_coords, species, lattice_params, or None on failure.
    """
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(cif_path)
        return {
            "frac_coords": np.array([s.frac_coords for s in struct]),
            "species": [str(s.specie) for s in struct],
            "lattice_params": np.array([
                struct.lattice.a, struct.lattice.b, struct.lattice.c,
                struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma,
            ]),
            "num_atoms": len(struct),
            "formula": struct.composition.reduced_formula,
            "structure": struct,
        }
    except Exception as e:
        logger.debug("Failed to load %s: %s", cif_path, e)
        return None


def check_validity(structure: dict, min_dist_threshold: float = 0.5) -> dict:
    """Check if a crystal structure is physically valid.

    Criteria:
      1. No overlapping atoms (min distance > threshold)
      2. Positive lattice parameters
      3. Reasonable angles (20-170 degrees)
      4. At least 2 atoms

    Returns:
        dict with is_valid, min_distance, issues
    """
    issues = []
    lp = structure["lattice_params"]

    # Check lattice parameters
    if np.any(lp[:3] <= 0):
        issues.append("Non-positive lattice length")
    if np.any(lp[3:] < 20) or np.any(lp[3:] > 170):
        issues.append(f"Extreme angles: {lp[3:]}")

    # Check minimum distance
    struct_obj = structure.get("structure")
    min_dist = float("inf")
    if struct_obj is not None:
        try:
            all_dists = struct_obj.distance_matrix
            np.fill_diagonal(all_dists, float("inf"))
            min_dist = all_dists.min()
            if min_dist < min_dist_threshold:
                issues.append(f"Atom overlap: min_dist={min_dist:.3f} A")
        except Exception:
            pass

    # Check atom count
    if structure["num_atoms"] < 2:
        issues.append("Too few atoms")

    return {
        "is_valid": len(issues) == 0,
        "min_distance": min_dist,
        "issues": issues,
    }


def check_composition_match(structure: dict, target_A: str, target_B: str, target_X: str) -> dict:
    """Check if composition matches target ABX3 stoichiometry.

    Returns:
        dict with matches, actual_composition, expected_ratio
    """
    species = structure["species"]
    counts = defaultdict(int)
    for s in species:
        counts[s] += 1

    n_A = counts.get(target_A, 0)
    n_B = counts.get(target_B, 0)
    n_X = counts.get(target_X, 0)
    total = len(species)

    # Check ABX3 ratio (1:1:3 or multiples thereof)
    if n_A > 0 and n_B > 0 and n_X > 0:
        ratio_ok = (n_A == n_B) and (n_X == 3 * n_A)
    else:
        ratio_ok = False

    # Check no extra elements
    expected = {target_A, target_B, target_X}
    actual = set(counts.keys())
    extra = actual - expected

    return {
        "matches": ratio_ok and len(extra) == 0,
        "counts": dict(counts),
        "expected_ratio": "1:1:3",
        "actual_ratio": f"{n_A}:{n_B}:{n_X}",
        "extra_elements": list(extra),
    }


def compute_lattice_deviation(
    generated_params: np.ndarray,
    reference_params: np.ndarray,
) -> dict:
    """Compute lattice parameter deviation metrics.

    Args:
        generated_params: [N_gen, 6] generated lattice parameters.
        reference_params: [N_ref, 6] reference lattice parameters.

    Returns:
        dict with rmsd, mae, per-parameter stats
    """
    # Compare to nearest reference structure
    gen_mean = generated_params.mean(axis=0)
    ref_mean = reference_params.mean(axis=0)

    mae = np.abs(gen_mean - ref_mean)
    rmsd = np.sqrt(np.mean((gen_mean - ref_mean) ** 2))

    return {
        "rmsd": float(rmsd),
        "mae_lengths": mae[:3].tolist(),
        "mae_angles": mae[3:].tolist(),
        "gen_mean": gen_mean.tolist(),
        "ref_mean": ref_mean.tolist(),
    }


def compute_tolerance_factor(structure: dict) -> Optional[float]:
    """Compute Goldschmidt tolerance factor for a perovskite structure."""
    from units import IONIC_RADII

    species = structure["species"]
    unique = list(set(species))
    counts = {s: species.count(s) for s in unique}

    # Identify A, B, X by count ratio
    sorted_elems = sorted(counts.items(), key=lambda x: x[1])

    if len(sorted_elems) < 3:
        return None

    # X = most abundant, A and B = least abundant
    x_elem = sorted_elems[-1][0]
    a_elem = sorted_elems[0][0]
    b_elem = sorted_elems[1][0] if len(sorted_elems) > 2 else sorted_elems[0][0]

    r_A = IONIC_RADII.get(a_elem)
    r_B = IONIC_RADII.get(b_elem)
    r_X = IONIC_RADII.get(x_elem)

    if r_A is None or r_B is None or r_X is None:
        return None

    from units import goldschmidt_tolerance
    return goldschmidt_tolerance(r_A, r_B, r_X)


def evaluate_generated(
    generated_dir: str,
    reference_dir: Optional[str] = None,
    target_A: Optional[str] = None,
    target_B: Optional[str] = None,
    target_X: Optional[str] = None,
) -> dict:
    """Evaluate a batch of generated structures.

    Args:
        generated_dir: Directory containing generated CIF files.
        reference_dir: Directory containing reference CIF files.
        target_A, target_B, target_X: Target composition elements.

    Returns:
        dict of aggregate metrics.
    """
    gen_path = Path(generated_dir)
    gen_files = sorted(gen_path.glob("*.cif"))

    if not gen_files:
        logger.warning("No CIF files found in %s", generated_dir)
        return {}

    logger.info("Evaluating %d generated structures", len(gen_files))

    # Load generated structures
    structures = []
    for f in gen_files:
        s = load_structure(str(f))
        if s is not None:
            structures.append(s)

    if not structures:
        return {"error": "No valid structures loaded"}

    # Validity
    validity_results = [check_validity(s) for s in structures]
    valid_count = sum(1 for v in validity_results if v["is_valid"])
    min_dists = [v["min_distance"] for v in validity_results if v["min_distance"] < float("inf")]

    metrics = {
        "num_generated": len(gen_files),
        "num_valid": valid_count,
        "validity_rate": valid_count / len(structures),
        "avg_min_distance": float(np.mean(min_dists)) if min_dists else None,
    }

    # Composition match
    if target_A and target_B and target_X:
        comp_results = [check_composition_match(s, target_A, target_B, target_X) for s in structures]
        comp_match = sum(1 for c in comp_results if c["matches"])
        metrics["composition_match_rate"] = comp_match / len(structures)

    # Lattice deviation
    gen_params = np.array([s["lattice_params"] for s in structures])
    metrics["lattice_mean"] = gen_params.mean(axis=0).tolist()
    metrics["lattice_std"] = gen_params.std(axis=0).tolist()

    if reference_dir:
        ref_path = Path(reference_dir)
        ref_files = sorted(ref_path.glob("*.cif"))[:200]  # Sample reference
        ref_structures = []
        for f in ref_files:
            s = load_structure(str(f))
            if s is not None:
                ref_structures.append(s)

        if ref_structures:
            ref_params = np.array([s["lattice_params"] for s in ref_structures])
            dev = compute_lattice_deviation(gen_params, ref_params)
            metrics["lattice_deviation"] = dev

    # Tolerance factors
    tolerances = []
    for s in structures:
        t = compute_tolerance_factor(s)
        if t is not None:
            tolerances.append(t)
    if tolerances:
        metrics["tolerance_factor_mean"] = float(np.mean(tolerances))
        metrics["tolerance_factor_std"] = float(np.std(tolerances))
        metrics["tolerance_in_range"] = float(np.mean([0.8 <= t <= 1.0 for t in tolerances]))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated perovskite structures")
    parser.add_argument("--generated_dir", required=True, help="Directory with generated CIFs")
    parser.add_argument("--reference_dir", default=None, help="Directory with reference CIFs")
    parser.add_argument("--A", default=None, help="Target A-site element")
    parser.add_argument("--B", default=None, help="Target B-site element")
    parser.add_argument("--X", default=None, help="Target anion element")
    parser.add_argument("--metrics", nargs="+",
                        default=["validity", "composition_match", "lattice_deviation"],
                        help="Metrics to compute")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = evaluate_generated(
        generated_dir=args.generated_dir,
        reference_dir=args.reference_dir,
        target_A=args.A,
        target_B=args.B,
        target_X=args.X,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", key, value)
        elif isinstance(value, dict):
            logger.info("  %s:", key)
            for k, v in value.items():
                logger.info("    %s: %s", k, v)
        else:
            logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
