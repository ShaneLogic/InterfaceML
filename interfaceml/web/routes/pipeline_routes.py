"""
Routes for the DFT Pipeline tab.

Provides endpoints to:
  A. Build perovskite/fullerene interfaces + generate CP2K DFT inputs as a ZIP.
  B. Generate DFT inputs from an already-built POSCAR (standalone).
"""

from __future__ import annotations

import logging
import math
import secrets
import tempfile
import zipfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

bp = Blueprint("pipeline", __name__)

# ---------------------------------------------------------------------------
# Download endpoint for generated ZIP files
# ---------------------------------------------------------------------------

_zip_store: dict = {}  # run_id -> path on disk


@bp.route("/api/pipeline/download/<run_id>", methods=["GET"])
def pipeline_download(run_id: str):
    """Serve a previously generated ZIP file."""
    zip_path = _zip_store.get(run_id)
    if not zip_path or not Path(zip_path).exists():
        return jsonify({"status": "error", "error": "ZIP not found or expired"}), 404
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"dft_jobs_{run_id}.zip",
    )


# ---------------------------------------------------------------------------
# Endpoint A: Build interfaces + generate DFT inputs
# ---------------------------------------------------------------------------


@bp.route("/api/pipeline/build-and-prep", methods=["POST"])
def pipeline_build_and_prep():
    """Build perovskite/fullerene interfaces then generate CP2K + SLURM files.

    Returns a JSON summary and a download URL for the ZIP archive.
    """
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"status": "error", "error": "Core modules not available"}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "error": "A JSON object is required"}), 400
    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])

    try:
        config = _build_config_from_request(data)
        # ---- 1. Load perovskite ----
        perovskite_filename = data.get("perovskite_filename")
        if not perovskite_filename:
            return jsonify({"status": "error", "error": "perovskite_filename is required"}), 400

        perovskite_path = _resolve_upload(upload_folder, perovskite_filename)
        if not perovskite_path.exists():
            return jsonify({"status": "error", "error": "Perovskite file not found"}), 404

        perovskite = core_status.io.load_structure(perovskite_path)

        # ---- 2. Load / generate fullerene(s) ----
        fullerene_source = data.get("fullerene_source", "upload")
        fullerene_structures = []
        fullerene_paths: list[Path] = []

        if fullerene_source == "ai":
            ai_status = current_app.extensions.get("ai_status")
            if not ai_status or not ai_status.available:
                return jsonify(
                    {"status": "error", "error": "AI module not available for fullerene generation"}
                ), 503

            model_key = data.get("ai_model") or ai_status.default_model
            api = ai_status.get_api(model_key)
            if api is None:
                return jsonify(
                    {"status": "error", "error": f'AI model "{model_key}" not available'}
                ), 503

            num_carbon = int(data.get("ai_num_carbon", 60))
            num_samples = int(data.get("ai_num_samples", 1))
            run_id = secrets.token_hex(6)

            results = api.generate(num_carbon=num_carbon, num_samples=num_samples)
            for idx, result in enumerate(results):
                fname = f"pipeline_ai_{run_id}_C{num_carbon}_{idx:03d}.xyz"
                fpath = upload_folder / fname
                api.save_structure(result["positions"], result["edges"], str(fpath), format="xyz")
                fullerene_paths.append(fpath)
                fullerene_structures.append(core_status.io.load_structure(fpath))
        else:
            fullerene_filename = data.get("fullerene_filename")
            if not fullerene_filename:
                return jsonify({"status": "error", "error": "fullerene_filename is required"}), 400
            fpath = _resolve_upload(upload_folder, fullerene_filename)
            if not fpath.exists():
                return jsonify({"status": "error", "error": "Fullerene file not found"}), 404
            fullerene_paths.append(fpath)
            fullerene_structures.append(core_status.io.load_structure(fpath))

        # ---- 3. Build interfaces ----
        from interfaceml.core.adsorbate import build_adsorbate_interface

        miller = _parse_int_tuple(data.get("miller"), length=3, default=(0, 0, 1))
        slab_thickness = float(data.get("slab_thickness", 18.0))
        vacuum = float(data.get("vacuum", 20.0))
        separation = float(data.get("separation", 3.2))
        supercell_raw = data.get("supercell", "auto")
        termination_raw = data.get("termination", "auto")
        termination = (
            None if str(termination_raw).lower() in ("auto", "any") else str(termination_raw)
        )
        fix_bottom_layers = int(data.get("fix_bottom_layers", 2))

        supercell_xy = None
        if supercell_raw and str(supercell_raw).lower() != "auto":
            parts = str(supercell_raw).replace("x", ",").split(",")
            if len(parts) == 2:
                supercell_xy = (int(parts[0]), int(parts[1]))

        run_id = secrets.token_hex(8)
        tmp_dir = Path(tempfile.mkdtemp(prefix="pipeline_", dir=str(upload_folder)))
        interface_records: list[dict] = []

        for idx, (full_struct, full_path) in enumerate(zip(fullerene_structures, fullerene_paths)):
            try:
                _bottom, _top, combined, choice = build_adsorbate_interface(
                    base_structure=perovskite,
                    adsorbate_structure=full_struct,
                    miller=miller,
                    slab_thickness=slab_thickness,
                    vacuum=vacuum,
                    separation=separation,
                    supercell_xy=supercell_xy,
                    termination=termination,
                )

                # Write POSCAR with selective dynamics
                poscar_name = f"interface_{idx:03d}.vasp"
                poscar_path = tmp_dir / poscar_name

                sd = _selective_dynamics_bottom_layers(
                    combined,
                    fix_bottom_layers,
                )
                core_status.io.write_poscar(combined, poscar_path, selective_dynamics=sd)

                interface_records.append(
                    {
                        "status": "success",
                        "output_path": str(poscar_path),
                        "perovskite_path": str(perovskite_path),
                        "fullerene_path": str(full_path),
                        "n_atoms": len(combined),
                        "supercell": f"{choice.nx}x{choice.ny}",
                    }
                )
            except Exception as exc:
                logger.error("Failed to build interface %d: %s", idx, exc)
                interface_records.append(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "perovskite_path": str(perovskite_path),
                        "fullerene_path": str(full_path),
                    }
                )

        # ---- 4. Generate DFT inputs ----
        from active_learning.dft_inputs import prepare_dft_calculations

        dft_dir = tmp_dir / "dft_jobs"
        jobs = prepare_dft_calculations(interface_records, dft_dir, config)

        if not any(job.get("status") == "prepared" for job in jobs):
            return jsonify(
                {
                    "status": "error",
                    "error": "No DFT jobs could be prepared",
                    "jobs_summary": jobs,
                    "interfaces": interface_records,
                }
            ), 422

        # ---- 5. ZIP everything ----
        zip_path = tmp_dir / f"dft_pipeline_{run_id}.zip"
        _zip_directory(dft_dir, zip_path)
        _zip_store[run_id] = str(zip_path)

        n_success = sum(1 for r in interface_records if r.get("status") == "success")
        n_prepared = sum(1 for j in jobs if j.get("status") == "prepared")

        jobs_summary = []
        for j in jobs:
            jobs_summary.append(
                {
                    "job_id": j.get("job_id", "?"),
                    "n_atoms": j.get("n_atoms"),
                    "elements": j.get("elements"),
                    "status": j.get("status"),
                    "error": j.get("error"),
                }
            )

        return jsonify(
            {
                "status": "success",
                "n_interfaces": n_success,
                "n_jobs": n_prepared,
                "download_url": f"/api/pipeline/download/{run_id}",
                "jobs_summary": jobs_summary,
            }
        )

    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Pipeline build-and-prep failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Endpoint B: Standalone DFT prep from existing POSCAR
# ---------------------------------------------------------------------------


@bp.route("/api/pipeline/dft-prep", methods=["POST"])
def pipeline_dft_prep():
    """Generate CP2K + SLURM files for an already-built structure."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"status": "error", "error": "Core modules not available"}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "error": "A JSON object is required"}), 400
    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])

    try:
        config = _build_config_from_request(data)
        structure_filename = data.get("structure_filename")
        if not structure_filename:
            return jsonify({"status": "error", "error": "structure_filename is required"}), 400

        structure_path = _resolve_upload(upload_folder, structure_filename)
        if not structure_path.exists():
            return jsonify({"status": "error", "error": "Structure file not found"}), 404

        run_id = secrets.token_hex(8)
        tmp_dir = Path(tempfile.mkdtemp(prefix="dftprep_", dir=str(upload_folder)))

        # Build a single interface record
        interface_records = [
            {
                "status": "success",
                "output_path": str(structure_path),
                "perovskite_path": str(structure_path),
                "fullerene_path": "standalone",
            }
        ]

        from active_learning.dft_inputs import prepare_dft_calculations

        dft_dir = tmp_dir / "dft_jobs"
        jobs = prepare_dft_calculations(interface_records, dft_dir, config)

        if not any(job.get("status") == "prepared" for job in jobs):
            return jsonify(
                {
                    "status": "error",
                    "error": "No DFT jobs could be prepared",
                    "jobs_summary": jobs,
                }
            ), 422

        zip_path = tmp_dir / f"dft_prep_{run_id}.zip"
        _zip_directory(dft_dir, zip_path)
        _zip_store[run_id] = str(zip_path)

        n_prepared = sum(1 for j in jobs if j.get("status") == "prepared")
        jobs_summary = []
        for j in jobs:
            jobs_summary.append(
                {
                    "job_id": j.get("job_id", "?"),
                    "n_atoms": j.get("n_atoms"),
                    "elements": j.get("elements"),
                    "status": j.get("status"),
                    "error": j.get("error"),
                }
            )

        return jsonify(
            {
                "status": "success",
                "n_interfaces": 1,
                "n_jobs": n_prepared,
                "download_url": f"/api/pipeline/download/{run_id}",
                "jobs_summary": jobs_summary,
            }
        )

    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Pipeline dft-prep failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_upload(upload_folder: Path, filename: str) -> Path:
    """Resolve and validate a filename within the upload folder."""
    safe_name = secure_filename(str(filename))
    resolved = (upload_folder / safe_name).resolve()
    try:
        resolved.relative_to(upload_folder.resolve())
    except ValueError as exc:
        raise ValueError(f"Invalid filename: {filename}") from exc
    return resolved


def _parse_int_tuple(value, *, length: int = 3, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == length:
        try:
            return tuple(int(x) for x in value)
        except Exception:
            return default
    return default


def _build_config_from_request(data: dict) -> dict:
    """Convert flat request params into the nested config dict expected by
    ``prepare_dft_calculations``."""
    dft_raw = data.get("dft", {})
    slurm_raw = data.get("slurm", {})
    if not isinstance(dft_raw, dict) or not isinstance(slurm_raw, dict):
        raise ValueError("dft and slurm must be JSON objects")

    config = {
        "dft": {
            "functional": dft_raw.get("functional", "PBE"),
            "cutoff": int(dft_raw.get("cutoff", 400)),
            "rel_cutoff": int(dft_raw.get("rel_cutoff", 60)),
            "max_scf": int(dft_raw.get("max_scf", 300)),
            "dispersion": dft_raw.get("dispersion", "DFT-D3"),
            "geo_opt_max_iter": int(dft_raw.get("geo_opt_max_iter", 200)),
            "geo_opt_convergence": float(dft_raw.get("geo_opt_convergence", 3e-3)),
        },
        "slurm": {
            "partition": slurm_raw.get("partition", "gpu"),
            "nodes": int(slurm_raw.get("nodes", 1)),
            "ntasks_per_node": int(slurm_raw.get("ntasks_per_node", 48)),
            "time": slurm_raw.get("time", "24:00:00"),
            "cp2k_module": slurm_raw.get("cp2k_module", "cp2k/2024.1"),
            "cp2k_binary": slurm_raw.get("cp2k_binary", "cp2k.psmp"),
            "account": slurm_raw.get("account"),
        },
    }
    for section, names in (
        ("dft", ("cutoff", "rel_cutoff", "max_scf", "geo_opt_max_iter", "geo_opt_convergence")),
        ("slurm", ("nodes", "ntasks_per_node")),
    ):
        for name in names:
            value = config[section][name]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} must be a positive finite number")
    return config


def _selective_dynamics_bottom_layers(
    structure,
    n_fix_layers: int,
    tol: float = 1.5,
) -> list[tuple[bool, bool, bool]] | None:
    """Build selective dynamics flags: fix bottom N z-layers, relax the rest."""
    if n_fix_layers <= 0:
        return None

    import numpy as np

    from interfaceml.core import layering

    n_hat = layering.interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n_hat)

    # Cluster heights into layers using a simple tolerance-based approach
    sorted_h = np.sort(heights)
    layer_boundaries = [sorted_h[0]]
    for h in sorted_h[1:]:
        if h - layer_boundaries[-1] > tol:
            layer_boundaries.append(h)

    if len(layer_boundaries) <= n_fix_layers:
        # Fix all atoms
        threshold = sorted_h[-1] + 1.0
    else:
        threshold = layer_boundaries[n_fix_layers]

    sd = []
    for h in heights:
        if h < threshold - 0.01:
            sd.append((False, False, False))  # Fixed
        else:
            sd.append((True, True, True))  # Free
    return sd


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    """Create a ZIP archive of an entire directory tree."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                zf.write(file_path, arcname)
