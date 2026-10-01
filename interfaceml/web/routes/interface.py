"""
Routes for interface building, layer fixing, and splitting.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from interfaceml.web.utils import compress_ranges, format_ranges

bp = Blueprint("interface", __name__)


def _matching_mode(data):
    mode = str(data.get("matching_mode", data.get("strain_mode", "bidirectional"))).lower()
    if mode == "film_only":
        mode = "forward"
    if mode not in ("bidirectional", "forward"):
        raise ValueError(
            "Matching mode must be bidirectional or forward; the film is strained to the substrate."
        )
    return mode


@bp.route("/api/list-interface-terminations", methods=["POST"])
def list_interface_terminations():
    """Enumerate substrate/film termination labels for a substrate+film+miller pair."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"error": "Core modules not available"}), 500

    data = request.json or {}
    substrate_file = data.get("base_file") or data.get("substrate_file")
    film_file = data.get("film_file")
    if not substrate_file or not Path(substrate_file).exists():
        return jsonify({"error": "Invalid or missing substrate file"}), 400
    if not film_file or not Path(film_file).exists():
        return jsonify({"error": "Invalid or missing film file"}), 400

    def _parse_miller(raw, default=(0, 0, 1)):
        try:
            tup = tuple(int(x) for x in raw)
            return tup if len(tup) == 3 else default
        except Exception:
            return default

    sub_miller = _parse_miller(data.get("miller_base") or data.get("miller_substrate", [0, 0, 1]))
    film_miller = _parse_miller(data.get("miller_film", [0, 0, 1]))

    try:
        max_area = float(data.get("max_area", 400))
        max_length_tol = float(data.get("max_length_tol", 0.03))
        max_angle_tol = float(data.get("max_angle_tol", 0.01))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid numeric parameter: {exc}"}), 400

    try:
        matching_mode = _matching_mode(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    bidirectional = matching_mode == "bidirectional"

    try:
        substrate = core_status.io.load_structure(substrate_file)
        film = core_status.io.load_structure(film_file)
    except Exception as exc:
        return jsonify({"error": f"Failed to load structures: {exc}"}), 400

    from interfaceml.core.interfaces_builder import list_interface_terminations as _list_terms

    try:
        info = _list_terms(
            substrate,
            film,
            substrate_miller=sub_miller,
            film_miller=film_miller,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
            bidirectional=bidirectional,
        )
    except Exception as exc:
        return jsonify({"status": "failed", "error": f"Termination scan failed: {exc}"}), 500

    return jsonify({"status": "success", **info})


@bp.route("/api/build-interface", methods=["POST"])
def build_interface():
    """Build coherent slab-slab heterojunction(s) via Zur-McGill matching."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"error": "Core modules not available"}), 500

    data = request.json or {}
    substrate_file = data.get("base_file") or data.get("substrate_file")
    film_file = data.get("film_file")
    if not substrate_file or not Path(substrate_file).exists():
        return jsonify({"error": "Invalid or missing substrate file"}), 400
    if not film_file or not Path(film_file).exists():
        return jsonify({"error": "Invalid or missing film file"}), 400

    def _parse_miller(raw, default=(0, 0, 1)):
        try:
            tup = tuple(int(x) for x in raw)
            return tup if len(tup) == 3 else default
        except Exception:
            return default

    sub_miller = _parse_miller(data.get("miller_base") or data.get("miller_substrate", [0, 0, 1]))
    film_miller = _parse_miller(data.get("miller_film", [0, 0, 1]))

    try:
        max_area = float(data.get("max_area", 400))
        gap = float(data.get("gap", data.get("distance", 2.5)))
        vacuum = float(data.get("vacuum", 20.0))
        max_length_tol = float(data.get("max_length_tol", 0.03))
        max_angle_tol = float(data.get("max_angle_tol", 0.01))
        film_thickness = int(data.get("film_thickness", 1))
        substrate_thickness = int(data.get("substrate_thickness", 1))
        top_k = int(data.get("top_k", 3))
        max_termination_pairs = int(data.get("max_termination_pairs", 3))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid numeric parameter: {exc}"}), 400

    try:
        matching_mode = _matching_mode(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    bidirectional = matching_mode == "bidirectional"

    # Optional explicit termination pair (Phase 2)
    sub_term = (data.get("substrate_termination") or "").strip() or None
    film_term = (data.get("film_termination") or "").strip() or None
    explicit_termination = (sub_term, film_term) if sub_term and film_term else None

    # Phase 3: lateral registry + twist
    try:
        xy_shift_x = float(data.get("xy_shift_x", 0.0))
        xy_shift_y = float(data.get("xy_shift_y", 0.0))
        twist_deg = float(data.get("twist_deg", 0.0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid registry/twist value: {exc}"}), 400
    xy_units = str(data.get("xy_units", "angstrom")).lower()
    if xy_units not in ("angstrom", "fractional"):
        xy_units = "angstrom"

    try:
        substrate = core_status.io.load_structure(substrate_file)
        film = core_status.io.load_structure(film_file)
    except Exception as exc:
        return jsonify({"error": f"Failed to load structures: {exc}"}), 400

    from interfaceml.core.interfaces_builder import build_coherent_interfaces

    try:
        candidates = build_coherent_interfaces(
            substrate,
            film,
            substrate_miller=sub_miller,
            film_miller=film_miller,
            max_area=max_area,
            max_length_tol=max_length_tol,
            max_angle_tol=max_angle_tol,
            bidirectional=bidirectional,
            gap=gap,
            vacuum=vacuum,
            film_thickness=film_thickness,
            substrate_thickness=substrate_thickness,
            top_k=top_k,
            max_termination_pairs=max_termination_pairs,
            termination=explicit_termination,
        )
    except Exception as exc:
        return jsonify({"status": "failed", "error": f"Interface build failed: {exc}"}), 500

    if not candidates:
        return jsonify(
            {
                "status": "failed",
                "error": (
                    f"No coherent matches found within tolerance "
                    f"(max_area={max_area} Å², length_tol={max_length_tol}, angle_tol={max_angle_tol}). "
                    f"Try relaxing tolerances or raising max_area."
                ),
            }
        ), 200

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    sub_stem = Path(substrate_file).stem
    film_stem = Path(film_file).stem
    results = []
    from interfaceml.core.interfaces_builder import apply_registry_twist

    apply_transform = abs(xy_shift_x) > 1e-9 or abs(xy_shift_y) > 1e-9 or abs(twist_deg) > 1e-9

    for i, c in enumerate(candidates):
        ifc = c["interface"]
        if apply_transform:
            try:
                ifc = apply_registry_twist(
                    ifc,
                    xy_shift=(xy_shift_x, xy_shift_y),
                    xy_units=xy_units,
                    twist_deg=twist_deg,
                )
            except Exception as exc:
                results.append(
                    {"status": "failed", "rank": i + 1, "error": f"registry/twist failed: {exc}"}
                )
                continue
        out_name = f"{sub_stem}_{film_stem}_{run_id}_match{i:02d}.vasp"
        out_path = upload_folder / out_name
        try:
            core_status.io.write_poscar(
                ifc,
                out_path,
                comment=(
                    f"{sub_stem}+{film_stem} match #{i + 1} vm={c['von_mises_strain']:.4f}"
                    + (
                        f" shift=({xy_shift_x},{xy_shift_y}){xy_units[0]} twist={twist_deg}deg"
                        if apply_transform
                        else ""
                    )
                ),
            )
        except Exception as exc:
            results.append({"status": "failed", "rank": i + 1, "error": str(exc)})
            continue

        term_label = (
            ", ".join(str(t) for t in c["termination"])
            if isinstance(c["termination"], (tuple, list))
            else str(c["termination"])
        )
        results.append(
            {
                "status": "success",
                "rank": i + 1,
                "adsorbate": f"match #{i + 1}",
                "n_atoms": c["n_atoms"],
                "supercell": f"sub {c.get('substrate_transformation')}  film {c.get('film_transformation')}",
                "strain_pct": round(float(c["von_mises_strain"]) * 100.0, 3)
                if c["von_mises_strain"] == c["von_mises_strain"]
                else None,
                "area": round(float(c["area"]), 2),
                "termination": term_label,
                "output_file": str(out_path),
                "filename": out_name,
                "download_url": f"/api/download/{out_name}",
            }
        )

    n_ok = sum(1 for r in results if r["status"] == "success")
    return jsonify(
        {
            "status": "success" if n_ok else "failed",
            "run_id": run_id,
            "stack_mode": "separate",
            "substrate": sub_stem,
            "film": film_stem,
            "substrate_miller": list(sub_miller),
            "film_miller": list(film_miller),
            "max_area": max_area,
            "matching_mode": matching_mode,
            "strained_layer": "film",
            "warnings": [
                "Geometric film rotation applied. Periodic commensurability after rotation has not been validated."
            ]
            if abs(twist_deg) > 1e-9
            else [],
            "n_built": n_ok,
            "n_total": len(candidates),
            "results": results,
            "message": (
                f"Built {n_ok} interface geometry candidate(s) with film rotation"
                if abs(twist_deg) > 1e-9
                else f"Built {n_ok} coherent interface match(es); ranked by von Mises strain"
            ),
        }
    )


def _z_rotation_matrix(deg: float) -> np.ndarray:
    """Rotation about z-axis (interface normal for c-axis slabs)."""
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _parse_xy_frac(raw, default=(0.5, 0.5)):
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]) % 1.0, float(raw[1]) % 1.0)
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
        if len(parts) == 2:
            return (float(parts[0]) % 1.0, float(parts[1]) % 1.0)
    raise ValueError(f"xy_frac must be two floats, got {raw!r}")


@bp.route("/api/build-adsorbate", methods=["POST"])
def build_adsorbate():
    """Build a perovskite/adsorbate interface from uploaded structures.

    Supports:
      - rotation_deg: in-plane rotation of each adsorbate around z (deg)
      - xy_frac: [x, y] fractional placement of adsorbate in slab a/b basis
      - stack_mode: 'separate' (one interface per adsorbate, default) or
                    'stacked' (all adsorbates stacked on a single slab,
                    each separated by `distance`)
    """
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"error": "Core modules not available"}), 500

    from interfaceml.core.adsorbate import (
        Lattice,
        SlabGenerator,
        _parse_target_atoms,
        _select_slab_by_termination,
        auto_supercell_xy,
        build_adsorbate_interface,
        prepare_adsorbate_layer,
        stack_structures,
    )

    data = request.json or {}
    base_file = data.get("base_file")
    adsorbate_files = data.get("adsorbate_files") or []

    if not base_file or not Path(base_file).exists():
        return jsonify({"error": "Invalid or missing base_file"}), 400
    if not adsorbate_files:
        return jsonify({"error": "At least one adsorbate file is required"}), 400

    miller_raw = data.get("miller", [0, 0, 1])
    try:
        miller = tuple(int(x) for x in miller_raw)
        if len(miller) != 3:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "miller must be 3 integers"}), 400

    termination_raw = str(data.get("termination", "auto"))
    termination = None if termination_raw.lower() in ("auto", "any", "") else termination_raw
    termination_atoms = _parse_target_atoms(data.get("termination_atoms"))

    supercell_raw = str(data.get("supercell", "auto"))
    supercell_xy = None
    if supercell_raw.lower() != "auto":
        parts = supercell_raw.replace("x", ",").split(",")
        if len(parts) == 2:
            try:
                supercell_xy = (int(parts[0]), int(parts[1]))
            except ValueError:
                return jsonify({"error": f"Invalid supercell: {supercell_raw}"}), 400

    try:
        separation = float(data.get("distance", 3.2))
        slab_thickness = float(data.get("slab_thickness", 18.0))
        vacuum = float(data.get("vacuum", 20.0))
        buffer = float(data.get("buffer", 10.0))
        layer_tol = float(data.get("layer_tol", 1.5))
        rotation_deg = float(data.get("rotation_deg", 0.0))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Numeric param invalid: {exc}"}), 400

    try:
        xy_frac = _parse_xy_frac(data.get("xy_frac"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    stack_mode = str(data.get("stack_mode", "separate")).lower()
    if stack_mode not in ("separate", "stacked"):
        return jsonify({"error": f"stack_mode must be separate|stacked, got {stack_mode!r}"}), 400

    rotation = _z_rotation_matrix(rotation_deg) if abs(rotation_deg) > 1e-9 else None

    try:
        perovskite = core_status.io.load_structure(base_file)
    except Exception as exc:
        return jsonify({"error": f"Failed to load base: {exc}"}), 400

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"

    if stack_mode == "separate":
        results = []
        for idx, ads_path in enumerate(adsorbate_files):
            ads_path = Path(ads_path)
            if not ads_path.exists():
                results.append(
                    {"status": "failed", "adsorbate": str(ads_path), "error": "File not found"}
                )
                continue
            try:
                adsorbate_struct = core_status.io.load_structure(ads_path)
                _bottom, _top, combined, choice = build_adsorbate_interface(
                    base_structure=perovskite,
                    adsorbate_structure=adsorbate_struct,
                    miller=miller,
                    slab_thickness=slab_thickness,
                    vacuum=vacuum,
                    separation=separation,
                    supercell_xy=supercell_xy,
                    buffer=buffer,
                    xy_frac=xy_frac,
                    termination=termination,
                    termination_atoms=termination_atoms,
                    layer_tol=layer_tol,
                    rotation=rotation,
                )

                out_name = f"{Path(base_file).stem}_{ads_path.stem}_{run_id}_{idx:02d}.vasp"
                out_path = upload_folder / out_name
                core_status.io.write_poscar(
                    combined,
                    out_path,
                    comment=f"{Path(base_file).stem} + {ads_path.stem} ({choice.nx}x{choice.ny})",
                )
                results.append(
                    {
                        "status": "success",
                        "adsorbate": ads_path.name,
                        "n_atoms": len(combined),
                        "supercell": f"{choice.nx}x{choice.ny}",
                        "termination": getattr(choice, "termination", None),
                        "rotation_deg": rotation_deg,
                        "xy_frac": list(xy_frac),
                        "output_file": str(out_path),
                        "filename": out_name,
                        "download_url": f"/api/download/{out_name}",
                    }
                )
            except Exception as exc:
                results.append({"status": "failed", "adsorbate": ads_path.name, "error": str(exc)})

        n_ok = sum(1 for r in results if r["status"] == "success")
        return jsonify(
            {
                "status": "success" if n_ok else "failed",
                "run_id": run_id,
                "stack_mode": stack_mode,
                "n_built": n_ok,
                "n_total": len(adsorbate_files),
                "results": results,
                "message": f"Built {n_ok}/{len(adsorbate_files)} adsorbate interface(s)",
            }
        ), (200 if n_ok else 500)

    # stacked mode: load all adsorbates, build one slab, stack iteratively
    adsorbate_structs = []
    bad = []
    for ads_path in adsorbate_files:
        p = Path(ads_path)
        if not p.exists():
            bad.append({"adsorbate": str(p), "error": "File not found"})
            continue
        try:
            adsorbate_structs.append((p, core_status.io.load_structure(p)))
        except Exception as exc:
            bad.append({"adsorbate": p.name, "error": str(exc)})

    if not adsorbate_structs:
        return jsonify(
            {
                "status": "failed",
                "run_id": run_id,
                "stack_mode": stack_mode,
                "errors": bad,
                "message": "No valid adsorbate files",
            }
        ), 400

    try:
        slabs = SlabGenerator(
            perovskite,
            miller_index=miller,
            min_slab_size=slab_thickness,
            min_vacuum_size=0.0,
            center_slab=True,
        ).get_slabs()
        if not slabs:
            return jsonify({"error": f"SlabGenerator returned no slabs for miller={miller}"}), 500

        slab, term_info = _select_slab_by_termination(
            slabs, target=termination, target_atoms=termination_atoms, layer_tol=layer_tol
        )

        if supercell_xy is None:
            # Pick supercell sized for largest adsorbate to avoid overlap
            best_choice = None
            for _, ads_struct in adsorbate_structs:
                c = auto_supercell_xy(slab, ads_struct, buffer=buffer)
                if best_choice is None or (c.nx * c.ny) > (best_choice.nx * best_choice.ny):
                    best_choice = c
            nx, ny = best_choice.nx, best_choice.ny
            choice_diameter = best_choice.diameter
        else:
            nx, ny = supercell_xy
            choice_diameter = 0.0

        slab_super = slab.copy()
        slab_super.make_supercell([nx, ny, 1])

        combined = slab_super
        stacked_info = []
        for ads_path, ads_struct in adsorbate_structs:
            top_lattice = Lattice(np.array(combined.lattice.matrix))
            ads_layer = prepare_adsorbate_layer(
                ads_struct,
                top_lattice,
                xy_frac=xy_frac,
                rotation=rotation,
            )
            _bot, _top, combined = stack_structures(
                combined,
                ads_layer,
                separation=separation,
                vacuum=vacuum,
            )
            stacked_info.append(
                {
                    "adsorbate": ads_path.name,
                    "n_atoms": len(ads_struct),
                }
            )

        out_name = f"{Path(base_file).stem}_stacked_{run_id}.vasp"
        out_path = upload_folder / out_name
        core_status.io.write_poscar(
            combined,
            out_path,
            comment=f"{Path(base_file).stem} + {len(stacked_info)} adsorbates stacked ({nx}x{ny})",
        )

        return jsonify(
            {
                "status": "success",
                "run_id": run_id,
                "stack_mode": stack_mode,
                "n_built": 1,
                "n_total": len(adsorbate_files),
                "n_stacked": len(stacked_info),
                "n_atoms": len(combined),
                "supercell": f"{nx}x{ny}",
                "diameter": choice_diameter,
                "termination": term_info.termination,
                "rotation_deg": rotation_deg,
                "xy_frac": list(xy_frac),
                "output_file": str(out_path),
                "filename": out_name,
                "download_url": f"/api/download/{out_name}",
                "stack_order": stacked_info,
                "load_errors": bad,
                "message": f"Stacked {len(stacked_info)} adsorbates on slab",
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "status": "failed",
                "run_id": run_id,
                "stack_mode": stack_mode,
                "error": str(exc),
                "load_errors": bad,
            }
        ), 500


@bp.route("/api/fix-layers", methods=["POST"])
def fix_layers():
    """Add selective dynamics to a structure."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"error": "Core modules not available"}), 500

    data = request.json
    structure_file = data.get("structure_file")

    if not structure_file or not Path(structure_file).exists():
        return jsonify({"error": "Invalid structure file"}), 400

    try:
        structure = core_status.io.load_structure(structure_file)
        mode = data.get("mode", "by_z_layers")
        n_fix = data.get("n_fix_layers", 3)
        include_mols = data.get("include_molecules", True)

        if mode == "by_z_layers":
            layers, _ = core_status.layering.split_layers_by_z(structure, gap_cut=True)

            if n_fix > len(layers):
                return jsonify(
                    {"error": f"Only {len(layers)} layers found, cannot fix {n_fix}"}
                ), 400

            fixed_indices = []
            for layer in layers[:n_fix]:
                fixed_indices.extend(layer)

            if include_mols:
                fixed_indices = core_status.layering.include_whole_molecules(
                    structure, fixed_indices
                )

            selective_dynamics = []
            fixed_set = set(fixed_indices)
            for i in range(len(structure)):
                if i in fixed_set:
                    selective_dynamics.append((False, False, False))
                else:
                    selective_dynamics.append((True, True, True))

            output_filename = Path(structure_file).stem + "_fixed.vasp"
            output_path = Path(current_app.config["UPLOAD_FOLDER"]) / output_filename
            core_status.io.write_poscar(
                structure, output_path, selective_dynamics=selective_dynamics
            )

            return jsonify(
                {
                    "status": "success",
                    "n_layers_total": len(layers),
                    "n_layers_fixed": n_fix,
                    "n_atoms_fixed": len(fixed_indices),
                    "output_file": str(output_path),
                    "download_url": f"/api/download/{output_filename}",
                    "fixed_indices": sorted(fixed_indices),
                }
            )

        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/split-layers", methods=["POST"])
def split_layers():
    """Split a multi-layer structure into separate layer files."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available or not core_status.splitting_available:
        return jsonify({"error": "Core splitting module not available"}), 500

    data = request.json
    structure_file = data.get("structure_file")
    n_interfaces = data.get("n_interfaces", 1)

    if not structure_file or not Path(structure_file).exists():
        return jsonify({"error": "Invalid structure file"}), 400

    try:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"

        structure = core_status.io.load_structure(structure_file)

        input_hash = hashlib.sha256(Path(structure_file).read_bytes()).hexdigest()[:12]

        min_gap = float(data.get("min_gap", 1.5))
        use_smart = data.get("use_smart_detection", True)

        layers = core_status.splitting.split_structure_into_layers(
            structure, n_interfaces=n_interfaces, min_gap=min_gap, use_smart_detection=use_smart
        )

        base_name = Path(structure_file).stem
        layer_info = []

        normal = core_status.layering.interface_normal_unit(structure)
        all_heights = np.dot(structure.cart_coords, normal)

        for i, layer_struct in enumerate(layers, start=1):
            layer_type = core_status.splitting.identify_layer_type(layer_struct)
            output_filename = f"{base_name}_{run_id}_layer{i}_{layer_type.replace(' ', '_')}.vasp"
            output_path = Path(current_app.config["UPLOAD_FOLDER"]) / output_filename

            layer_indices = []
            for site_idx, site in enumerate(structure):
                if any(
                    np.allclose(site.frac_coords, layer_struct[j].frac_coords)
                    for j in range(len(layer_struct))
                ):
                    layer_indices.append(site_idx)

            layer_heights = all_heights[layer_indices] if layer_indices else [0]
            z_min = float(np.min(layer_heights))
            z_max = float(np.max(layer_heights))

            core_status.io.write_poscar(
                layer_struct,
                output_path,
                comment=f"{base_name} - Layer {i}/{len(layers)}: {layer_type}",
            )

            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()[:12]

            layer_info.append(
                {
                    "layer_number": i,
                    "n_atoms": len(layer_struct),
                    "composition": layer_struct.composition.formula,
                    "layer_type": layer_type,
                    "z_range": f"{z_min:.2f} - {z_max:.2f} Å",
                    "filename": output_filename,
                    "download_url": f"/api/download/{output_filename}?v={run_id}",
                    "file_hash": output_hash,
                }
            )

        splitting_file = getattr(core_status.splitting, "__file__", None)
        splitting_hash = None
        try:
            if splitting_file and Path(splitting_file).exists():
                splitting_hash = hashlib.sha256(Path(splitting_file).read_bytes()).hexdigest()[:12]
        except Exception:
            splitting_hash = None

        return jsonify(
            {
                "status": "success",
                "run_id": run_id,
                "input_file": str(structure_file),
                "input_hash": input_hash,
                "splitting_file": splitting_file,
                "splitting_hash": splitting_hash,
                "n_interfaces": n_interfaces,
                "n_layers": len(layers),
                "layers": layer_info,
                "message": f"Successfully split into {len(layers)} layers",
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@bp.route("/api/pdos-layers", methods=["POST"])
def pdos_layers():
    """Generate layer-resolved atom indices for CP2K PDOS input."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({"error": "Core modules not available"}), 500

    if not core_status.splitting_available:
        return jsonify({"error": "Core splitting module not available"}), 500

    data = request.json or {}
    structure_file = data.get("structure_file")

    if not structure_file or not Path(structure_file).exists():
        return jsonify({"error": "Invalid structure file"}), 400

    try:
        structure = core_status.io.load_structure(structure_file)
        n_interfaces = int(data.get("n_interfaces", 1))
        min_gap = float(data.get("min_gap", 1.5))
        use_smart = bool(data.get("use_smart_detection", True))

        layers_struct = core_status.splitting.split_structure_into_layers(
            structure,
            n_interfaces=n_interfaces,
            min_gap=min_gap,
            use_smart_detection=use_smart,
        )

        heights = np.dot(
            np.asarray(structure.cart_coords, dtype=float),
            core_status.layering.interface_normal_unit(structure),
        )

        layers: list[list[int]] = []
        for layer_struct in layers_struct:
            layer_indices = []
            for site_idx, site in enumerate(structure):
                if any(
                    np.allclose(site.frac_coords, layer_struct[j].frac_coords)
                    for j in range(len(layer_struct))
                ):
                    layer_indices.append(site_idx)
            layers.append(sorted(set(layer_indices)))

        pdos_filename = data.get("pdos_filename") or f"{Path(structure_file).stem}_PDOS"
        nlumo = int(data.get("nlumo", 250))

        layer_info = []
        ldos_blocks = []
        for idx, layer in enumerate(layers, start=1):
            indices_0 = sorted(int(i) for i in layer)
            indices_1 = [i + 1 for i in indices_0]
            ranges_1 = format_ranges(compress_ranges(indices_1))
            z_min = float(np.min(heights[indices_0])) if indices_0 else 0.0
            z_max = float(np.max(heights[indices_0])) if indices_0 else 0.0

            layer_info.append(
                {
                    "layer_number": idx,
                    "n_atoms": len(indices_0),
                    "indices_0": indices_0,
                    "indices_1": indices_1,
                    "ranges_1": ranges_1,
                    "z_range": f"{z_min:.2f} - {z_max:.2f} Å",
                }
            )

            ldos_blocks.append(
                "\n".join(
                    [
                        "&LDOS",
                        f"  LIST {ranges_1}",
                        "&END LDOS",
                    ]
                )
            )

        pdos_block = "\n".join(
            [
                "&PDOS",
                f"  FILENAME {pdos_filename}",
                f"  NLUMO {nlumo}",
                "  COMPONENTS",
                *ldos_blocks,
                "&END PDOS",
            ]
        )

        return jsonify(
            {
                "status": "success",
                "input_file": str(structure_file),
                "n_layers": len(layers),
                "n_interfaces": n_interfaces,
                "min_gap": min_gap,
                "use_smart_detection": use_smart,
                "layers": layer_info,
                "cp2k_pdos": pdos_block,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
