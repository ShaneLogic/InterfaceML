"""Step 5: Filter DFT results and prepare retraining datasets."""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def prepare_retraining_data(
    results_path: Path,
    existing_data_dir: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """Filter + convert DFT results into retraining dataset.

    Args:
        results_path: Path to results.json from parse step.
        existing_data_dir: Directory with existing training data.
        output_dir: Directory for retraining outputs.
        config: Full pipeline configuration dict.

    Returns:
        dict with keys: perovskite_cifs, interface_structures, num_accepted,
        num_rejected, stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    retrain_cfg = config.get("retraining", {})

    energy_filter = retrain_cfg.get("energy_filter", -1.0)
    force_threshold = retrain_cfg.get("force_threshold", 0.5)
    min_structures = retrain_cfg.get("min_structures", 50)

    # Load results
    results = json.loads(results_path.read_text())

    # Filter
    accepted = []
    rejected = []
    for r in results:
        reason = _check_acceptance(r, energy_filter, force_threshold)
        if reason is None:
            accepted.append(r)
        else:
            r["reject_reason"] = reason
            rejected.append(r)

    logger.info(
        "Filtering: %d accepted, %d rejected out of %d total",
        len(accepted),
        len(rejected),
        len(results),
    )

    # Warn if below minimum
    if len(accepted) < min_structures:
        logger.warning(
            "Only %d accepted structures (minimum: %d). "
            "Consider relaxing filters or running more DFT.",
            len(accepted),
            min_structures,
        )

    # Extract and save structures
    perovskite_cif_dir = output_dir / "perovskite_cifs"
    interface_data_dir = output_dir / "interface_data"
    perovskite_cif_dir.mkdir(parents=True, exist_ok=True)
    interface_data_dir.mkdir(parents=True, exist_ok=True)

    perovskite_count = 0
    interface_count = 0

    for r in accepted:
        opt_xyz = r.get("optimized_xyz")
        if not opt_xyz or not Path(opt_xyz).exists():
            continue

        job_id = r["job_id"]

        # Copy optimized interface structure
        dst_xyz = interface_data_dir / f"{job_id}_optimized.xyz"
        shutil.copy2(opt_xyz, dst_xyz)
        interface_count += 1

        # Extract perovskite slab from optimized interface
        if retrain_cfg.get("retrain_perovskite", True):
            try:
                cif_path = perovskite_cif_dir / f"{job_id}_slab.cif"
                _extract_perovskite_slab(opt_xyz, cif_path, r)
                perovskite_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to extract perovskite slab from %s: %s",
                    job_id, exc,
                )

    # Compute statistics
    stats = _compute_stats(accepted, rejected)

    # Write retrain config
    retrain_config = _build_retrain_config(
        config,
        perovskite_cif_dir,
        interface_data_dir,
        existing_data_dir,
        stats,
    )
    retrain_config_path = output_dir / "retrain_config.yaml"
    retrain_config_path.write_text(yaml.dump(retrain_config, default_flow_style=False))

    # Save accepted/rejected lists
    (output_dir / "accepted.json").write_text(
        json.dumps(accepted, indent=2, default=str)
    )
    (output_dir / "rejected.json").write_text(
        json.dumps(rejected, indent=2, default=str)
    )

    summary = {
        "perovskite_cifs": str(perovskite_cif_dir),
        "interface_structures": str(interface_data_dir),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "perovskite_cifs_extracted": perovskite_count,
        "interface_structures_saved": interface_count,
        "stats": stats,
    }

    logger.info(
        "Prepared retraining data: %d perovskite CIFs, %d interface structures",
        perovskite_count,
        interface_count,
    )

    return summary


def _check_acceptance(
    result: dict,
    energy_filter: float,
    force_threshold: float,
) -> Optional[str]:
    """Check if a result passes quality filters.

    Returns None if accepted, or a rejection reason string.
    """
    if not result.get("converged"):
        return "not_converged"

    if result.get("total_energy_eV") is None:
        return "no_energy"

    binding = result.get("binding_energy_eV")
    if binding is not None and binding > energy_filter:
        return f"binding_energy_too_high ({binding:.4f} > {energy_filter})"

    max_force = result.get("max_force_Ha_bohr")
    if max_force is not None and max_force > force_threshold:
        return f"force_too_high ({max_force:.6f} > {force_threshold})"

    return None


def _extract_perovskite_slab(
    optimized_xyz: str,
    output_cif: Path,
    result: dict,
) -> None:
    """Extract the perovskite slab portion from an optimized interface.

    Uses layer splitting to separate the fullerene (molecular) layer
    from the perovskite slab layers.
    """
    from interfaceml.core.io import load_structure

    structure = load_structure(optimized_xyz)

    # Identify carbon atoms (fullerene) vs non-carbon (perovskite slab)
    # This is a heuristic — fullerene is pure carbon, perovskite has mixed elements
    carbon_indices = []
    slab_indices = []
    for i, site in enumerate(structure):
        if str(site.specie) == "C":
            carbon_indices.append(i)
        else:
            slab_indices.append(i)

    if not slab_indices:
        raise ValueError("No non-carbon atoms found — cannot extract slab")

    # Extract slab structure
    slab_structure = structure.copy()
    # Remove carbon atoms (in reverse order to preserve indices)
    for idx in sorted(carbon_indices, reverse=True):
        slab_structure.remove_sites([idx])

    # Write as CIF
    slab_structure.to(filename=str(output_cif))


def _compute_stats(accepted: List[dict], rejected: List[dict]) -> dict:
    """Compute summary statistics."""
    stats = {
        "total": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / max(len(accepted) + len(rejected), 1),
    }

    energies = [r["total_energy_eV"] for r in accepted if r.get("total_energy_eV")]
    if energies:
        stats["energy_mean_eV"] = float(np.mean(energies))
        stats["energy_std_eV"] = float(np.std(energies))
        stats["energy_min_eV"] = float(np.min(energies))
        stats["energy_max_eV"] = float(np.max(energies))

    forces = [r["max_force_Ha_bohr"] for r in accepted if r.get("max_force_Ha_bohr")]
    if forces:
        stats["max_force_mean"] = float(np.mean(forces))
        stats["max_force_max"] = float(np.max(forces))

    # Rejection reasons
    reasons = {}
    for r in rejected:
        reason = r.get("reject_reason", "unknown")
        # Group parametric reasons
        base_reason = reason.split("(")[0].strip()
        reasons[base_reason] = reasons.get(base_reason, 0) + 1
    stats["rejection_reasons"] = reasons

    return stats


def _build_retrain_config(
    config: dict,
    perovskite_cif_dir: Path,
    interface_data_dir: Path,
    existing_data_dir: Path,
    stats: dict,
) -> dict:
    """Build a retraining configuration file."""
    retrain_cfg = config.get("retraining", {})

    retrain_config = {
        "data": {
            "existing_data_dir": str(existing_data_dir),
            "new_perovskite_cifs": str(perovskite_cif_dir),
            "new_interface_data": str(interface_data_dir),
            "num_new_structures": stats["accepted"],
        },
        "training": {
            "retrain_perovskite": retrain_cfg.get("retrain_perovskite", True),
            "retrain_fullerene": retrain_cfg.get("retrain_fullerene", False),
            "fine_tune": True,
            "learning_rate": 1e-4,
            "epochs": 50,
        },
        "round": config["pipeline"].get("round", 0),
    }

    return retrain_config
