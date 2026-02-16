"""Step 4: Parse CP2K output files to extract energies, forces, and optimized geometries."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HA_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177249


def parse_dft_results(
    dft_jobs_dir: Path,
    output_path: Path,
) -> List[dict]:
    """Parse CP2K outputs and collect results.

    Args:
        dft_jobs_dir: Directory containing job subdirectories.
        output_path: Path to write results.json.

    Returns:
        List of result dicts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job_dirs = sorted(
        d for d in dft_jobs_dir.iterdir()
        if d.is_dir() and (d / "cp2k.inp").exists()
    )

    if not job_dirs:
        logger.warning("No job directories found in %s", dft_jobs_dir)
        return []

    results = []
    for job_dir in job_dirs:
        job_id = job_dir.name
        result = _parse_single_job(job_dir, job_id)
        results.append(result)

    # Write results
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Parsed %d jobs, wrote results to %s", len(results), output_path)

    # Write human-readable summary
    summary_path = output_path.parent / "summary.txt"
    _write_summary(results, summary_path)

    return results


def _parse_single_job(job_dir: Path, job_id: str) -> dict:
    """Parse a single CP2K job directory."""
    result = {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "converged": False,
        "total_energy_Ha": None,
        "total_energy_eV": None,
        "max_force_Ha_bohr": None,
        "n_scf_steps": None,
        "n_geo_steps": None,
        "n_atoms": None,
        "optimized_xyz": None,
        "status": "not_started",
    }

    # Load metadata if available
    metadata_path = job_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            result["perovskite_formula"] = _extract_formula(
                metadata.get("perovskite_path", "")
            )
            result["fullerene_C"] = _extract_carbon_count(
                metadata.get("fullerene_path", "")
            )
            result["n_atoms"] = metadata.get("n_atoms")
            result["metadata"] = metadata
        except Exception:
            pass

    # Check if cp2k.out exists
    cp2k_out = job_dir / "cp2k.out"
    if not cp2k_out.exists():
        result["status"] = "not_started"
        return result

    result["status"] = "started"

    try:
        out_text = cp2k_out.read_text(errors="replace")
    except Exception as exc:
        result["status"] = "read_error"
        result["error"] = str(exc)
        return result

    # Parse total energy
    energy = _parse_total_energy(out_text)
    if energy is not None:
        result["total_energy_Ha"] = energy
        result["total_energy_eV"] = energy * HA_TO_EV

    # Parse convergence
    result["converged"] = _check_convergence(out_text)

    # Parse SCF steps
    result["n_scf_steps"] = _count_scf_steps(out_text)

    # Parse geometry optimization steps
    result["n_geo_steps"] = _count_geo_steps(out_text)

    # Parse max force
    max_force = _parse_max_force(out_text)
    if max_force is not None:
        result["max_force_Ha_bohr"] = max_force

    # Look for optimized trajectory
    traj_files = list(job_dir.glob("*-pos-1.xyz"))
    if not traj_files:
        traj_files = list(job_dir.glob("*-pos-*.xyz"))

    if traj_files:
        traj_file = traj_files[0]
        opt_xyz = job_dir / "optimized.xyz"
        _extract_last_frame(traj_file, opt_xyz)
        if opt_xyz.exists():
            result["optimized_xyz"] = str(opt_xyz)

    # Set final status
    if result["converged"]:
        result["status"] = "converged"
    elif energy is not None:
        result["status"] = "completed_not_converged"
    else:
        result["status"] = "failed"

    return result


def _parse_total_energy(text: str) -> Optional[float]:
    """Extract the final total energy from CP2K output."""
    # CP2K prints: ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:     -1234.567890
    matches = re.findall(
        r"ENERGY\|\s+Total FORCE_EVAL.*?energy.*?:\s+([-\d.Ee+]+)",
        text,
    )
    if matches:
        return float(matches[-1])  # Last occurrence = final energy
    return None


def _check_convergence(text: str) -> bool:
    """Check if geometry optimization converged."""
    return "GEOMETRY OPTIMIZATION COMPLETED" in text


def _count_scf_steps(text: str) -> Optional[int]:
    """Count total SCF iterations."""
    matches = re.findall(r"SCF\s+run\s+converged\s+in\s+(\d+)\s+steps", text)
    if matches:
        return sum(int(m) for m in matches)
    return None


def _count_geo_steps(text: str) -> Optional[int]:
    """Count geometry optimization steps."""
    matches = re.findall(r"OPTIMIZATION STEP:\s+(\d+)", text)
    if matches:
        return int(matches[-1])
    return None


def _parse_max_force(text: str) -> Optional[float]:
    """Extract the final maximum force."""
    # CP2K prints: Max. step size     =     0.0012345678
    # or: MAX GRADIENT       =     0.0012345678
    matches = re.findall(
        r"Max(?:imum)?\s+(?:gradient|force)\s*[:=]\s*([-\d.Ee+]+)",
        text,
        re.IGNORECASE,
    )
    if matches:
        return float(matches[-1])

    # Alternative pattern
    matches = re.findall(
        r"Convergence.*?RMS\s+gradient\s*[:=]\s*([-\d.Ee+]+)",
        text,
        re.IGNORECASE,
    )
    if matches:
        return float(matches[-1])

    return None


def _extract_last_frame(trajectory_path: Path, output_path: Path) -> None:
    """Extract the last frame from a CP2K trajectory XYZ file."""
    try:
        text = trajectory_path.read_text()
    except Exception:
        return

    lines = text.strip().splitlines()
    if not lines:
        return

    # Find the last frame by scanning backwards for atom count line
    frames = []
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        frame_end = i + n_atoms + 2  # count + comment + atoms
        if frame_end <= len(lines):
            frames.append((i, frame_end))
        i = frame_end

    if frames:
        start, end = frames[-1]
        output_path.write_text("\n".join(lines[start:end]) + "\n")


def _extract_formula(path: str) -> Optional[str]:
    """Extract perovskite formula from filename."""
    if not path:
        return None
    name = Path(path).stem
    # Try patterns like BaTiO3_sample_001
    match = re.match(r"([A-Z][a-z]?[A-Z][a-z]?[A-Z][a-z]?\d*)", name)
    if match:
        return match.group(1)
    return name


def _extract_carbon_count(path: str) -> Optional[int]:
    """Extract carbon count from fullerene filename."""
    if not path:
        return None
    match = re.search(r"C(\d+)", Path(path).stem)
    if match:
        return int(match.group(1))
    return None


def _write_summary(results: List[dict], summary_path: Path) -> None:
    """Write a human-readable summary of DFT results."""
    total = len(results)
    converged = sum(1 for r in results if r["converged"])
    failed = sum(1 for r in results if r["status"] == "failed")
    not_started = sum(1 for r in results if r["status"] == "not_started")
    energies = [r["total_energy_eV"] for r in results if r["total_energy_eV"] is not None]

    lines = [
        "=" * 60,
        "DFT Results Summary",
        "=" * 60,
        f"Total jobs:      {total}",
        f"Converged:       {converged}",
        f"Failed:          {failed}",
        f"Not started:     {not_started}",
        "",
    ]

    if energies:
        lines.extend([
            f"Energy range:    {min(energies):.4f} to {max(energies):.4f} eV",
            f"Mean energy:     {np.mean(energies):.4f} eV",
            "",
        ])

    lines.append("-" * 60)
    lines.append(f"{'Job ID':<30s} {'Status':<15s} {'Energy (eV)':>15s} {'Force':>10s}")
    lines.append("-" * 60)

    for r in results:
        energy_str = f"{r['total_energy_eV']:.4f}" if r["total_energy_eV"] else "N/A"
        force_str = f"{r['max_force_Ha_bohr']:.6f}" if r["max_force_Ha_bohr"] else "N/A"
        lines.append(
            f"{r['job_id']:<30s} {r['status']:<15s} {energy_str:>15s} {force_str:>10s}"
        )

    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote summary to %s", summary_path)
