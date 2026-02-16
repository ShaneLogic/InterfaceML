"""Step 2: Build fullerene/perovskite interface structures."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def build_interfaces(
    fullerene_dir: Path,
    perovskite_dir: Path,
    output_dir: Path,
    config: dict,
) -> List[dict]:
    """Build interfaces using batch_adsorbate_builder.

    Args:
        fullerene_dir: Directory containing generated fullerene .xyz files.
        perovskite_dir: Directory containing generated perovskite .cif files.
        output_dir: Directory for output interface POSCAR files.
        config: Full pipeline configuration dict.

    Returns:
        List of metadata dicts from metadata.jsonl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    iface_cfg = config["interface"]

    miller = ",".join(str(m) for m in iface_cfg["miller"])
    xy_grid = ",".join(str(g) for g in iface_cfg.get("xy_grid", [2, 2]))

    # Build command for batch_adsorbate_builder
    builder_script = (
        Path(__file__).resolve().parent.parent
        / "build_heterojunctions"
        / "batch_adsorbate_builder.py"
    )

    cmd = [
        sys.executable,
        str(builder_script),
        "--perovskite_dir", str(perovskite_dir),
        "--fullerene_dir", str(fullerene_dir),
        "--output_dir", str(output_dir),
        "--miller", miller,
        "--slab_thickness", str(iface_cfg.get("slab_thickness", 18.0)),
        "--vacuum", str(iface_cfg.get("vacuum", 20.0)),
        "--distance", str(iface_cfg.get("distance", 3.2)),
        "--orientation_samples", str(iface_cfg.get("orientation_samples", 4)),
        "--xy_grid", xy_grid,
        "--seed", str(config["pipeline"].get("seed", 42)),
    ]

    max_interfaces = iface_cfg.get("max_interfaces")
    if max_interfaces:
        cmd.extend(["--max_pairs", str(max_interfaces)])

    logger.info("Running batch_adsorbate_builder: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    if result.returncode != 0:
        logger.error("batch_adsorbate_builder failed:\n%s", result.stderr)
        raise RuntimeError(
            f"batch_adsorbate_builder exited with code {result.returncode}: "
            f"{result.stderr[:500]}"
        )

    if result.stdout:
        logger.info("Builder output:\n%s", result.stdout[:1000])

    # Read metadata
    metadata_path = output_dir / "metadata.jsonl"
    records = _read_metadata(metadata_path)

    # Apply selective dynamics (fix bottom layers)
    fix_layers = config.get("dft", {}).get("fix_bottom_layers", 2)
    if fix_layers and fix_layers > 0:
        records = _apply_selective_dynamics(records, fix_layers)

    successful = [r for r in records if r.get("status") == "success"]
    logger.info(
        "Built %d interfaces (%d successful, %d filtered/failed)",
        len(records),
        len(successful),
        len(records) - len(successful),
    )

    return records


def _read_metadata(metadata_path: Path) -> List[dict]:
    """Read JSONL metadata file."""
    records = []
    if not metadata_path.exists():
        logger.warning("No metadata file found at %s", metadata_path)
        return records

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _apply_selective_dynamics(records: List[dict], n_fix_layers: int) -> List[dict]:
    """Apply selective dynamics to fix bottom layers of perovskite slab."""
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent / "build_heterojunctions"),
    )
    from fix_interface_layers import print_fixed_atoms_by_z_layers

    from interfaceml.core.io import load_structure, write_poscar

    for record in records:
        if record.get("status") != "success":
            continue

        poscar_path = record.get("output_path")
        if not poscar_path or not Path(poscar_path).exists():
            continue

        try:
            structure = load_structure(poscar_path)

            fixed_indices = print_fixed_atoms_by_z_layers(
                str(poscar_path),
                n_fix_layers=n_fix_layers,
                tol=None,
                one_based_output=False,
                gap_cut=True,
            )

            # Build selective dynamics flags: fixed atoms get (F,F,F)
            fixed_set = set(fixed_indices)
            selective_dynamics = [
                (False, False, False) if i in fixed_set else (True, True, True)
                for i in range(len(structure))
            ]

            write_poscar(
                structure,
                poscar_path,
                selective_dynamics=selective_dynamics,
                comment=f"Interface with {len(fixed_indices)} fixed atoms",
            )

            record["n_fixed_atoms"] = len(fixed_indices)

        except Exception as exc:
            logger.warning(
                "Failed to apply selective dynamics to %s: %s",
                poscar_path,
                exc,
            )

    return records
