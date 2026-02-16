"""Step 3: Generate CP2K input files and SLURM submission scripts."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# CP2K basis set + potential mappings for common interface elements
# Format: element -> (basis_set, potential)
_DEFAULT_BASIS_POTENTIAL = {
    # Fullerene
    "C": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q4"),
    "H": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q1"),
    # Perovskite A-site
    "Ba": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q10"),
    "Cs": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "Rb": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "K": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "Na": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "Ma": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q1"),  # Methylammonium proxy
    # Perovskite B-site
    "Ti": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q12"),
    "Pb": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q4"),
    "Sn": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q4"),
    "Ge": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q4"),
    "Zr": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q12"),
    # Perovskite X-site
    "O": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q6"),
    "I": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q7"),
    "Br": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q7"),
    "Cl": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q7"),
    "F": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q7"),
    "N": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q5"),
    "S": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q6"),
}

# Hartree to eV conversion
HA_TO_EV = 27.211386245988


def prepare_dft_calculations(
    interface_records: List[dict],
    output_base: Path,
    config: dict,
) -> List[dict]:
    """Generate CP2K input + SLURM script for each interface.

    Args:
        interface_records: List of metadata dicts from build_interfaces step.
        output_base: Base directory for DFT jobs (e.g., round_dir/dft_jobs/).
        config: Full pipeline configuration dict.

    Returns:
        List of job dicts with paths to generated files.
    """
    output_base.mkdir(parents=True, exist_ok=True)
    dft_cfg = config["dft"]
    slurm_cfg = config["slurm"]

    # Load templates
    template_dir = Path(__file__).resolve().parent / "templates"
    geo_opt_template = (template_dir / "cp2k_geo_opt.inp").read_text()
    slurm_template = (template_dir / "slurm_cp2k.sh").read_text()

    # Filter to successful interfaces only
    successful = [r for r in interface_records if r.get("status") == "success"]
    if not successful:
        logger.warning("No successful interfaces to prepare DFT for")
        return []

    jobs = []
    for idx, record in enumerate(successful, 1):
        poscar_path = Path(record["output_path"])
        if not poscar_path.exists():
            logger.warning("POSCAR not found: %s", poscar_path)
            continue

        # Create job directory
        perovskite_name = Path(record.get("perovskite_path", "unknown")).stem
        fullerene_name = Path(record.get("fullerene_path", "unknown")).stem
        job_name = f"{idx:04d}_{perovskite_name}_{fullerene_name}"
        # Truncate long names
        if len(job_name) > 80:
            job_name = job_name[:80]
        job_dir = output_base / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            job_info = _prepare_single_job(
                poscar_path=poscar_path,
                job_dir=job_dir,
                job_name=job_name,
                geo_opt_template=geo_opt_template,
                slurm_template=slurm_template,
                dft_cfg=dft_cfg,
                slurm_cfg=slurm_cfg,
                record=record,
            )
            jobs.append(job_info)
        except Exception as exc:
            logger.error("Failed to prepare job %s: %s", job_name, exc)
            jobs.append({
                "job_id": job_name,
                "status": "prep_failed",
                "error": str(exc),
            })

    # Generate submit_all.sh
    _write_submit_all(output_base, jobs)

    logger.info("Prepared %d DFT jobs in %s", len(jobs), output_base)
    return jobs


def _prepare_single_job(
    poscar_path: Path,
    job_dir: Path,
    job_name: str,
    geo_opt_template: str,
    slurm_template: str,
    dft_cfg: dict,
    slurm_cfg: dict,
    record: dict,
) -> dict:
    """Prepare CP2K input and SLURM script for a single interface."""
    from interfaceml.core.io import load_structure

    structure = load_structure(poscar_path)

    # Write structure.xyz for CP2K
    xyz_path = job_dir / "structure.xyz"
    _write_cp2k_xyz(structure, xyz_path)

    # Get unique elements
    elements = sorted(set(str(site.specie) for site in structure))

    # Build KIND blocks
    kind_blocks = _build_kind_blocks(elements, dft_cfg)

    # Build cell vectors
    lattice = structure.lattice
    cell_a = f"{lattice.matrix[0][0]:.10f} {lattice.matrix[0][1]:.10f} {lattice.matrix[0][2]:.10f}"
    cell_b = f"{lattice.matrix[1][0]:.10f} {lattice.matrix[1][1]:.10f} {lattice.matrix[1][2]:.10f}"
    cell_c = f"{lattice.matrix[2][0]:.10f} {lattice.matrix[2][1]:.10f} {lattice.matrix[2][2]:.10f}"

    # Build dispersion block
    dispersion_block = _build_dispersion_block(dft_cfg.get("dispersion", "DFT-D3"))

    # Build constraint block for fixed atoms
    constraint_block = _build_constraint_block(poscar_path, structure)

    # Fill CP2K template
    cp2k_input = geo_opt_template.format(
        project_name=job_name.replace("-", "_"),
        cutoff=dft_cfg.get("cutoff", 400),
        rel_cutoff=dft_cfg.get("rel_cutoff", 60),
        max_scf=dft_cfg.get("max_scf", 300),
        functional=dft_cfg.get("functional", "PBE"),
        dispersion_block=dispersion_block,
        cell_a=cell_a,
        cell_b=cell_b,
        cell_c=cell_c,
        kind_blocks=kind_blocks,
        geo_opt_max_iter=dft_cfg.get("geo_opt_max_iter", 200),
        geo_opt_convergence=dft_cfg.get("geo_opt_convergence", 3.0e-3),
        constraint_block=constraint_block,
    )

    cp2k_path = job_dir / "cp2k.inp"
    cp2k_path.write_text(cp2k_input)

    # Fill SLURM template
    account_line = ""
    if slurm_cfg.get("account"):
        account_line = f"#SBATCH --account={slurm_cfg['account']}"

    slurm_input = slurm_template.format(
        job_name=job_name,
        partition=slurm_cfg.get("partition", "gpu"),
        nodes=slurm_cfg.get("nodes", 1),
        ntasks_per_node=slurm_cfg.get("ntasks_per_node", 48),
        time=slurm_cfg.get("time", "24:00:00"),
        account_line=account_line,
        cp2k_module=slurm_cfg.get("cp2k_module", "cp2k/2024.1"),
        cp2k_binary=slurm_cfg.get("cp2k_binary", "cp2k.psmp"),
    )

    slurm_path = job_dir / "submit.sh"
    slurm_path.write_text(slurm_input)

    # Save metadata
    metadata = {
        "job_id": job_name,
        "source_poscar": str(poscar_path),
        "n_atoms": len(structure),
        "elements": elements,
        "cell_params": {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
        },
        "perovskite_path": record.get("perovskite_path"),
        "fullerene_path": record.get("fullerene_path"),
        "interface_record": record,
    }
    metadata_path = job_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    return {
        "job_id": job_name,
        "job_dir": str(job_dir),
        "cp2k_input": str(cp2k_path),
        "slurm_script": str(slurm_path),
        "xyz_file": str(xyz_path),
        "n_atoms": len(structure),
        "elements": elements,
        "status": "prepared",
    }


def _write_cp2k_xyz(structure, filepath: Path) -> None:
    """Write structure in XYZ format for CP2K."""
    n_atoms = len(structure)
    lines = [str(n_atoms), ""]
    for site in structure:
        elem = str(site.specie)
        x, y, z = site.coords
        lines.append(f"{elem:>4s} {x:16.10f} {y:16.10f} {z:16.10f}")
    filepath.write_text("\n".join(lines) + "\n")


def _build_kind_blocks(elements: List[str], dft_cfg: dict) -> str:
    """Build CP2K &KIND blocks for each element."""
    basis_set = dft_cfg.get("basis_set", "DZVP-MOLOPT-SR-GTH")
    potential_prefix = dft_cfg.get("potential", "GTH-PBE")

    blocks = []
    for elem in elements:
        if elem in _DEFAULT_BASIS_POTENTIAL:
            bs, pot = _DEFAULT_BASIS_POTENTIAL[elem]
        else:
            bs = basis_set
            pot = f"{potential_prefix}-q4"  # Fallback
            logger.warning(
                "Element %s not in default mapping, using %s / %s",
                elem, bs, pot,
            )

        blocks.append(
            f"    &KIND {elem}\n"
            f"      BASIS_SET {bs}\n"
            f"      POTENTIAL {pot}\n"
            f"    &END KIND"
        )
    return "\n".join(blocks)


def _build_dispersion_block(dispersion: Optional[str]) -> str:
    """Build CP2K dispersion correction block."""
    if not dispersion:
        return ""

    dispersion = dispersion.upper()
    if dispersion == "DFT-D3":
        return (
            "      &VDW_POTENTIAL\n"
            "        POTENTIAL_TYPE PAIR_POTENTIAL\n"
            "        &PAIR_POTENTIAL\n"
            "          TYPE DFTD3\n"
            "          PARAMETER_FILE_NAME dftd3.dat\n"
            "          REFERENCE_FUNCTIONAL PBE\n"
            "        &END PAIR_POTENTIAL\n"
            "      &END VDW_POTENTIAL"
        )
    elif dispersion == "DFT-D3(BJ)":
        return (
            "      &VDW_POTENTIAL\n"
            "        POTENTIAL_TYPE PAIR_POTENTIAL\n"
            "        &PAIR_POTENTIAL\n"
            "          TYPE DFTD3(BJ)\n"
            "          PARAMETER_FILE_NAME dftd3.dat\n"
            "          REFERENCE_FUNCTIONAL PBE\n"
            "        &END PAIR_POTENTIAL\n"
            "      &END VDW_POTENTIAL"
        )
    else:
        logger.warning("Unknown dispersion type: %s", dispersion)
        return ""


def _build_constraint_block(
    poscar_path: Path,
    structure,
) -> str:
    """Build CP2K CONSTRAINT block for fixed atoms from selective dynamics."""
    fixed_indices = _get_fixed_atoms_from_poscar(poscar_path)
    if not fixed_indices:
        return ""

    # CP2K uses 1-based atom indices in LIST
    fixed_1based = [i + 1 for i in fixed_indices]

    # Build fixed atoms list string (CP2K format)
    list_str = " ".join(str(i) for i in fixed_1based)

    return (
        "  &CONSTRAINT\n"
        "    &FIXED_ATOMS\n"
        f"      LIST {list_str}\n"
        "    &END FIXED_ATOMS\n"
        "  &END CONSTRAINT"
    )


def _get_fixed_atoms_from_poscar(poscar_path: Path) -> List[int]:
    """Extract fixed atom indices (0-based) from POSCAR selective dynamics."""
    try:
        lines = poscar_path.read_text().splitlines()
    except Exception:
        return []

    # Find selective dynamics line
    sd_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("selective"):
            sd_line_idx = i
            break

    if sd_line_idx is None:
        return []

    # Coordinate lines start after the selective dynamics line
    coord_start = sd_line_idx + 1
    fixed = []
    atom_idx = 0
    for line in lines[coord_start:]:
        parts = line.split()
        if len(parts) < 6:
            break
        # Selective dynamics flags are the last 3 columns (T/F)
        flags = parts[-3:]
        if all(f.upper() == "F" for f in flags):
            fixed.append(atom_idx)
        atom_idx += 1

    return fixed


def _write_submit_all(output_base: Path, jobs: List[dict]) -> None:
    """Write a master script to submit all SLURM jobs."""
    prepared = [j for j in jobs if j.get("status") == "prepared"]
    if not prepared:
        return

    lines = [
        "#!/bin/bash",
        f"# Submit all {len(prepared)} DFT jobs",
        f"# Generated for active learning round",
        "",
        "SCRIPT_DIR=$(cd \"$(dirname \"$0\")\" && pwd)",
        "",
    ]

    for job in prepared:
        job_dir = Path(job["job_dir"]).name
        lines.append(f'echo "Submitting {job["job_id"]}..."')
        lines.append(f'cd "$SCRIPT_DIR/{job_dir}" && sbatch submit.sh && cd "$SCRIPT_DIR"')
        lines.append("")

    lines.append(f'echo "Submitted {len(prepared)} jobs"')

    submit_all_path = output_base / "submit_all.sh"
    submit_all_path.write_text("\n".join(lines) + "\n")
    submit_all_path.chmod(0o755)

    logger.info("Wrote %s", submit_all_path)
