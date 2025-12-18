#!/usr/bin/env python3
"""
auto_interface_builder.py

Given two bulk structures (CIF/POSCAR), this script:
 - finds/optimizes primitive cells,
 - searches for low-strain commensurate interface supercells (in-plane),
 - builds slabs for given Miller indices,
 - applies strain to one or both materials per user choice,
 - stacks them with an initial separation and vacuum,
 - writes POSCARs for each slab (strained) and the combined interface.

Author: Xuan-Yan (worked example tuned for MAPbI3 / TiO2 workflows)
Requirements: pymatgen, numpy
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Set, Tuple, Optional

# Optional heavy dependencies (needed for build workflow).
# Split-layer workflow can run without them for CIF inputs.
try:
    import numpy as np  # type: ignore
    from pymatgen.core import Lattice, Structure  # type: ignore
    from pymatgen.core.surface import SlabGenerator  # type: ignore
    from pymatgen.transformations.standard_transformations import SupercellTransformation  # type: ignore
    # from pymatgen.analysis.interfaces import InterfaceMatcher
    from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder, Interface  # type: ignore
    from pymatgen.io.vasp import Poscar  # type: ignore
    _HAS_DEPS = True
except ModuleNotFoundError:
    np = None  # type: ignore
    Lattice = None  # type: ignore
    Structure = None  # type: ignore
    SlabGenerator = None  # type: ignore
    SupercellTransformation = None  # type: ignore
    CoherentInterfaceBuilder = None  # type: ignore
    Interface = None  # type: ignore
    Poscar = None  # type: ignore
    _HAS_DEPS = False


def _require_build_deps() -> None:
    if not _HAS_DEPS:
        raise RuntimeError(
            "This workflow requires numpy + pymatgen, but they are not available in the current Python environment. "
            "Install them (e.g. `pip install numpy pymatgen`) or run this script inside your conda environment."
        )


def _extend_matrix_2d_to_3d(mat_2d: np.ndarray) -> np.ndarray:
    """
    Promote a 2x2 transformation matrix to a 3x3 supercell matrix by keeping the
    original in-plane mapping and leaving the out-of-plane direction unchanged.
    """
    mat_3d = np.eye(3)
    mat_3d[0:2, 0:2] = np.array(mat_2d)
    return np.rint(mat_3d).astype(int)


def _matrix_key(film_trans: np.ndarray, sub_trans: np.ndarray) -> Tuple[int, ...]:
    """Create a hashable key describing a pair of 2x2 transformation matrices."""
    film_int = tuple(np.rint(film_trans).astype(int).flatten().tolist())
    sub_int = tuple(np.rint(sub_trans).astype(int).flatten().tolist())
    return film_int + sub_int


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    dot_val = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norms < 1e-8:
        return 0.0
    cos_theta = np.clip(dot_val / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _compute_match_metrics(match_obj) -> Dict[str, np.ndarray | float]:
    """Extract length/angle mismatch information from a ZSLMatch."""
    film_vectors = np.array(match_obj.film_vectors)
    sub_vectors = np.array(match_obj.substrate_vectors)

    film_lengths = np.linalg.norm(film_vectors, axis=1)
    sub_lengths = np.linalg.norm(sub_vectors, axis=1)
    avg_lengths = (film_lengths + sub_lengths) / 2.0
    # Relative mismatch per in-plane vector (signed)
    length_components = np.divide(
        (film_lengths - sub_lengths),
        np.where(avg_lengths < 1e-8, 1.0, avg_lengths),
    )

    film_angle = _angle_between(film_vectors[0], film_vectors[1])
    sub_angle = _angle_between(sub_vectors[0], sub_vectors[1])
    angle_mismatch = abs(film_angle - sub_angle)

    return {
        "film_lengths": film_lengths,
        "substrate_lengths": sub_lengths,
        "length_components": length_components,
        "max_length_mismatch": float(np.max(np.abs(length_components))),
        "film_angle": film_angle,
        "substrate_angle": sub_angle,
        "angle_mismatch": angle_mismatch,
    }


def _generate_zsl_parameter_grid(
    base_tol: float,
    max_area: float,
    allow_bidirectional: bool = True,
) -> List[Dict[str, float | bool]]:
    """
    Generate a prioritized list of parameter dictionaries for ZSLGenerator searches.
    Parameters with smaller area, tighter tolerances, and bidirectional matching are
    considered first to encourage compact supercells.
    """
    base_tol = max(0.01, min(base_tol, 0.15))
    length_candidates = {
        base_tol,
        min(0.15, base_tol * 1.35),
        min(0.15, base_tol * 1.7),
        min(0.15, base_tol + 0.02),
        min(0.15, 0.12 if base_tol < 0.12 else base_tol),
    }
    length_values = sorted(length_candidates)

    area_candidates = {
        max_area,
        max_area * 1.3,
        max_area * 1.7,
        max_area * 2.2,
    }
    area_values = sorted({float(max(10.0, a)) for a in area_candidates})

    bidi_options: Iterable[bool] = [True, False] if allow_bidirectional else [False]

    param_grid: List[Dict[str, float | bool]] = []
    for length_tol in length_values:
        angle_tol = min(0.1, max(0.015, length_tol / 3.0))
        for area in area_values:
            for bidi in bidi_options:
                param_grid.append(
                    {
                        "max_area": float(area),
                        "max_length_tol": float(length_tol),
                        "max_angle_tol": float(angle_tol),
                        "bidirectional": bool(bidi),
                    }
                )

    param_grid.sort(key=lambda p: (p["max_area"], p["max_length_tol"], 0 if p["bidirectional"] else 1))
    return param_grid


def _safe_structure_label(path_str: str) -> str:
    """
    Derive a readable label from an input structure path by stripping directories,
    removing the file suffix, and replacing spaces with underscores.
    """
    path = Path(path_str)
    label = path.stem  # Removes the final suffix such as .cif, .vasp, etc.
    return label.replace(" ", "_")


def _reorder_structure_for_poscar(struct: Structure) -> Structure:
    """
    Group sites by species so that POSCAR headers list each element once in the
    order they first appear. Within each species block, maintain ascending
    fractional z for readability.
    """
    struct_copy = struct.copy()
    species_order: List[str] = []
    for site in struct_copy:
        symbol = site.species_string
        if symbol not in species_order:
            species_order.append(symbol)
    order_index = {symbol: idx for idx, symbol in enumerate(species_order)}
    struct_copy.sort(key=lambda site: (order_index[site.species_string], site.frac_coords[2]))
    return struct_copy

def load_structure(path):
    _require_build_deps()
    s = Structure.from_file(path)
    return s


def _read_cif_preserve_frac(path: Path) -> Structure:
    """
    Read a (simple) CIF and preserve *raw* fractional coordinates as written.

    Motivation
    ----------
    For differential charge workflows, we sometimes need to split a heterojunction
    into independent layers while keeping their relative positions inside the cell
    unchanged. Many structure readers wrap fractional coords into [0,1), which can
    change how the split layers appear when viewed inside the unit cell. This
    reader keeps the original values (including negatives / >1).

    Notes
    -----
    This is intentionally lightweight: it supports CIFs that provide
    _cell_length_{a,b,c}, _cell_angle_{alpha,beta,gamma}, and an atom loop that
    includes _atom_site_label and _atom_site_fract_{x,y,z}. If parsing fails, the
    caller should fall back to pymatgen's Structure.from_file.
    """
    _require_build_deps()
    text = path.read_text(errors="ignore").splitlines()

    def _get_float(key: str) -> float:
        for line in text:
            if line.strip().startswith(key):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
        raise ValueError(f"Missing CIF key: {key}")

    a = _get_float("_cell_length_a")
    b = _get_float("_cell_length_b")
    c = _get_float("_cell_length_c")
    alpha = _get_float("_cell_angle_alpha")
    beta = _get_float("_cell_angle_beta")
    gamma = _get_float("_cell_angle_gamma")

    # Find atom loop headers
    header_start = None
    for i, line in enumerate(text):
        if line.strip() == "loop_":
            # Check whether subsequent lines include the headers we need
            headers: List[str] = []
            j = i + 1
            while j < len(text) and text[j].strip().startswith("_"):
                headers.append(text[j].strip())
                j += 1
            need = {
                "_atom_site_label",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            }
            if need.issubset(set(headers)):
                header_start = i
                header_lines = headers
                data_start = j
                break

    if header_start is None:
        raise ValueError("Could not find CIF atom loop with fractional coordinates.")

    # Map header -> column index
    col_index = {h: idx for idx, h in enumerate(header_lines)}
    idx_label = col_index["_atom_site_label"]
    idx_x = col_index["_atom_site_fract_x"]
    idx_y = col_index["_atom_site_fract_y"]
    idx_z = col_index["_atom_site_fract_z"]

    species: List[str] = []
    frac_coords: List[List[float]] = []

    # Read until next loop_/data_/header or EOF
    for k in range(data_start, len(text)):
        line = text[k].strip()
        if not line:
            continue
        if line.startswith("loop_") or line.startswith("data_") or line.startswith("_"):
            break
        parts = line.split()
        if len(parts) <= max(idx_label, idx_x, idx_y, idx_z):
            continue

        label = parts[idx_label]
        # Strip any trailing digits (e.g., "Ti1" -> "Ti") while keeping symbols like "Cl"
        elem = "".join([ch for ch in label if ch.isalpha()])
        if not elem:
            elem = label

        x = float(parts[idx_x])
        y = float(parts[idx_y])
        z = float(parts[idx_z])
        species.append(elem)
        frac_coords.append([x, y, z])

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return Structure(lattice, species, frac_coords, coords_are_cartesian=False, to_unit_cell=False)


def _load_structure_preserve_positions(path_str: str) -> Structure:
    """Load structure while preserving as-written fractional coords for CIF when possible."""
    _require_build_deps()
    path = Path(path_str)
    if path.suffix.lower() == ".cif":
        try:
            return _read_cif_preserve_frac(path)
        except Exception:
            # Fall back to pymatgen parser if CIF isn't in the expected simple format
            return Structure.from_file(str(path))
    return Structure.from_file(str(path))


def _split_heterojunction_into_layers(
    struct: Structure,
    layer1_elements: Set[str] | None = None,
    layer2_elements: Set[str] | None = None,
) -> Tuple[Structure, Structure, str, str]:
    """
    Split a heterojunction structure into two independent layers *without*
    changing the lattice or coordinates.

    Returns (layer1, layer2, layer1_label, layer2_label).
    """
    # Auto-detect for common oxide/perovskite interfaces (e.g., TiO2 + organo-lead halide)
    if layer1_elements is None and layer2_elements is None:
        present = {str(el) for el in struct.composition.elements}
        if "Ti" in present and "O" in present:
            layer1_elements = {"Ti", "O"}
            layer2_elements = present - layer1_elements
        else:
            # Conservative fallback: use the first element as layer1, rest as layer2
            # (user should ideally pass --layer1_elements/--layer2_elements for robustness)
            elems = sorted(present)
            layer1_elements = {elems[0]} if elems else set()
            layer2_elements = set(elems[1:]) if len(elems) > 1 else set()

    layer1_elements = set(layer1_elements or [])
    layer2_elements = set(layer2_elements or [])

    # If user provides only one side, infer the other
    present = {str(el) for el in struct.composition.elements}
    if layer1_elements and not layer2_elements:
        layer2_elements = present - layer1_elements
    if layer2_elements and not layer1_elements:
        layer1_elements = present - layer2_elements

    l1_species: List[str] = []
    l1_frac: List[List[float]] = []
    l2_species: List[str] = []
    l2_frac: List[List[float]] = []

    for site in struct:
        sym = site.species_string
        # species_string may include oxidation; for safety use element symbol-like alpha part
        elem = "".join([ch for ch in sym if ch.isalpha()]) or sym
        if elem in layer1_elements:
            l1_species.append(site.species_string)
            l1_frac.append(site.frac_coords.tolist())
        elif elem in layer2_elements:
            l2_species.append(site.species_string)
            l2_frac.append(site.frac_coords.tolist())
        else:
            # If an element isn't in either set, assign it to layer2 by default
            l2_species.append(site.species_string)
            l2_frac.append(site.frac_coords.tolist())

    layer1 = Structure(struct.lattice, l1_species, l1_frac, coords_are_cartesian=False, to_unit_cell=False)
    layer2 = Structure(struct.lattice, l2_species, l2_frac, coords_are_cartesian=False, to_unit_cell=False)

    label1 = "Layer 1"
    label2 = "Layer 2"
    if layer1_elements:
        label1 = f"Layer 1 ({','.join(sorted(layer1_elements))})"
    if layer2_elements:
        label2 = f"Layer 2 ({','.join(sorted(layer2_elements))})"
    return layer1, layer2, label1, label2


def _parse_cif_atom_loop(lines: List[str]) -> tuple[
    dict[str, float],
    List[tuple[str, str, float, float, float]],
]:
    """
    Parse a simple CIF (cell + fractional atom loop).

    Returns
    -------
    cell : dict
        Keys: a,b,c,alpha,beta,gamma
    atoms : list of tuples
        (label, elem, fx, fy, fz) in the order they appear in the CIF.
    """

    def _get_float(key: str) -> float:
        for ln in lines:
            if ln.strip().startswith(key):
                parts = ln.split()
                if len(parts) >= 2:
                    return float(parts[1])
        raise ValueError(f"Missing CIF key: {key}")

    cell = {
        "a": _get_float("_cell_length_a"),
        "b": _get_float("_cell_length_b"),
        "c": _get_float("_cell_length_c"),
        "alpha": _get_float("_cell_angle_alpha"),
        "beta": _get_float("_cell_angle_beta"),
        "gamma": _get_float("_cell_angle_gamma"),
    }

    # Find atom loop with fractional coordinates
    header_lines: List[str] = []
    data_start: int | None = None
    for i, ln in enumerate(lines):
        if ln.strip() != "loop_":
            continue
        headers: List[str] = []
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("_"):
            headers.append(lines[j].strip())
            j += 1
        need = {
            "_atom_site_label",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }
        if need.issubset(set(headers)):
            header_lines = headers
            data_start = j
            break
    if data_start is None:
        raise ValueError("Could not find CIF atom loop with fractional coordinates.")

    col_index = {h: idx for idx, h in enumerate(header_lines)}
    idx_label = col_index["_atom_site_label"]
    idx_x = col_index["_atom_site_fract_x"]
    idx_y = col_index["_atom_site_fract_y"]
    idx_z = col_index["_atom_site_fract_z"]

    atoms: List[tuple[str, str, float, float, float]] = []
    for k in range(data_start, len(lines)):
        line = lines[k].strip()
        if not line:
            continue
        if line.startswith("loop_") or line.startswith("data_") or line.startswith("_"):
            break
        parts = line.split()
        if len(parts) <= max(idx_label, idx_x, idx_y, idx_z):
            continue
        label = parts[idx_label]
        elem = "".join([ch for ch in label if ch.isalpha()]) or label
        fx = float(parts[idx_x])
        fy = float(parts[idx_y])
        fz = float(parts[idx_z])
        atoms.append((label, elem, fx, fy, fz))

    return cell, atoms


def _lattice_vectors_from_cell(cell: dict[str, float]) -> List[List[float]]:
    """Convert CIF cell parameters to lattice vectors (Å)."""
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    alpha = cell["alpha"]
    beta = cell["beta"]
    gamma = cell["gamma"]

    ar = math.radians(alpha)
    br = math.radians(beta)
    gr = math.radians(gamma)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gr), b * math.sin(gr), 0.0
    cx = c * math.cos(br)
    sin_g = math.sin(gr)
    if abs(sin_g) < 1e-12:
        raise ValueError("Invalid gamma angle; sin(gamma) ~ 0.")
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))
    return [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]


def _write_poscar_simple(
    path: Path,
    comment: str,
    lattice_vecs: List[List[float]],
    atoms: List[tuple[str, str, float, float, float]],
) -> None:
    """Write a minimal POSCAR with Direct coordinates, grouping atoms by element."""
    order: List[str] = []
    counts: Dict[str, int] = {}
    for _label, elem, *_ in atoms:
        if elem not in counts:
            counts[elem] = 0
            order.append(elem)
        counts[elem] += 1

    grouped: List[tuple[str, str, float, float, float]] = []
    for elem in order:
        grouped.extend([a for a in atoms if a[1] == elem])

    with path.open("w", encoding="utf-8") as f:
        f.write(comment.strip() + "\n")
        f.write("1.0\n")
        for v in lattice_vecs:
            f.write(f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")
        f.write(" ".join(order) + "\n")
        f.write(" ".join(str(counts[e]) for e in order) + "\n")
        f.write("Direct\n")
        for _label, _elem, fx, fy, fz in grouped:
            f.write(f"{fx:.16f} {fy:.16f} {fz:.16f}\n")


def _write_cif_simple(
    path: Path,
    data_name: str,
    cell: dict[str, float],
    atoms: List[tuple[str, str, float, float, float]],
    header_comment: str | None = None,
) -> None:
    """
    Write a simple CIF preserving fractional coordinates as provided (no wrapping).
    """
    with path.open("w", encoding="utf-8") as f:
        if header_comment:
            f.write(f"# {header_comment.strip()}\n")
        f.write(f"data_{data_name}\n")
        f.write(f"_cell_length_a   {cell['a']:.6f}\n")
        f.write(f"_cell_length_b   {cell['b']:.6f}\n")
        f.write(f"_cell_length_c   {cell['c']:.6f}\n")
        f.write(f"_cell_angle_alpha  {cell['alpha']:.6f}\n")
        f.write(f"_cell_angle_beta   {cell['beta']:.6f}\n")
        f.write(f"_cell_angle_gamma  {cell['gamma']:.6f}\n")
        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("x,y,z\n")
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        for label, _elem, fx, fy, fz in atoms:
            f.write(f"{label} {fx:.8f} {fy:.8f} {fz:.8f}\n")


def split_layers_to_folder(input_path_str: str, out_base_dir: str | None = None, layer1_elements: str | None = None, layer2_elements: str | None = None) -> Path:
    """
    Split a heterojunction model file into independent layers and write to:
      <out_base_dir or input_parent>/<input_stem>/
    along with a copy of the original input file.

    The written layer structures keep the original lattice and fractional coordinates unchanged.
    """
    input_path = Path(input_path_str)
    if out_base_dir is None:
        out_dir = input_path.parent / input_path.stem
    else:
        out_dir = Path(out_base_dir) / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy original file alongside outputs
    shutil.copy2(str(input_path), str(out_dir / input_path.name))

    stem = input_path.stem
    poscar1 = out_dir / f"{stem}_layer1.vasp"
    poscar2 = out_dir / f"{stem}_layer2.vasp"
    combined_path = out_dir / f"{stem}.vasp"

    if _HAS_DEPS:
        # Load with best-effort position preservation
        struct = _load_structure_preserve_positions(str(input_path))

        l1_set = {s.strip() for s in (layer1_elements or "").split(",") if s.strip()} or None
        l2_set = {s.strip() for s in (layer2_elements or "").split(",") if s.strip()} or None
        layer1, layer2, label1, label2 = _split_heterojunction_into_layers(struct, l1_set, l2_set)

        comment1 = f"{label1} extracted from {input_path.name}; lattice+fractional coords unchanged."
        comment2 = f"{label2} extracted from {input_path.name}; lattice+fractional coords unchanged."

        Poscar(layer1, comment=comment1, sort_structure=False).write_file(poscar1)
        Poscar(layer2, comment=comment2, sort_structure=False).write_file(poscar2)
        Poscar(struct, comment=f"Original structure from {input_path.name}; lattice+fractional coords unchanged.", sort_structure=False).write_file(combined_path)

        # Also provide CIF outputs for each layer (preserve fractional coords; no wrapping)
        cell = {
            "a": float(struct.lattice.a),
            "b": float(struct.lattice.b),
            "c": float(struct.lattice.c),
            "alpha": float(struct.lattice.alpha),
            "beta": float(struct.lattice.beta),
            "gamma": float(struct.lattice.gamma),
        }
        # Keep site order as in the Structure objects
        def _atoms_from_struct(s, prefix: str) -> List[tuple[str, str, float, float, float]]:
            out: List[tuple[str, str, float, float, float]] = []
            for i, site in enumerate(s):
                elem = "".join([ch for ch in site.species_string if ch.isalpha()]) or site.species_string
                label = f"{prefix}{i+1}"
                fx, fy, fz = site.frac_coords.tolist()
                out.append((label, elem, float(fx), float(fy), float(fz)))
            return out

        cif1 = out_dir / f"{stem}_layer1.cif"
        cif2 = out_dir / f"{stem}_layer2.cif"
        # Keep the original CIF copy untouched; write combined CIF with a distinct name.
        cif_combined = out_dir / f"{stem}_combined.cif"
        _write_cif_simple(cif1, f"{stem}_layer1", cell, _atoms_from_struct(layer1, "L1"), header_comment=comment1)
        _write_cif_simple(cif2, f"{stem}_layer2", cell, _atoms_from_struct(layer2, "L2"), header_comment=comment2)
        _write_cif_simple(cif_combined, stem, cell, _atoms_from_struct(struct, "A"), header_comment=f"Original structure from {input_path.name}; lattice+fractional coords unchanged.")
    else:
        # CIF-only fallback without numpy/pymatgen: parse CIF, split by element sets, write POSCARs.
        if input_path.suffix.lower() != ".cif":
            raise RuntimeError(
                "Split-layer mode without numpy/pymatgen currently supports CIF inputs only. "
                "Either install dependencies or provide a CIF."
            )

        lines = input_path.read_text(errors="ignore").splitlines()
        cell, all_atoms = _parse_cif_atom_loop(lines)
        lattice_vecs = _lattice_vectors_from_cell(cell)

        present = {elem for (_label, elem, *_xyz) in all_atoms}
        l1_set = {s.strip() for s in (layer1_elements or "").split(",") if s.strip()}
        l2_set = {s.strip() for s in (layer2_elements or "").split(",") if s.strip()}
        if not l1_set and not l2_set:
            if "Ti" in present and "O" in present:
                l1_set = {"Ti", "O"}
                l2_set = present - l1_set
            else:
                elems = sorted(present)
                l1_set = {elems[0]} if elems else set()
                l2_set = set(elems[1:]) if len(elems) > 1 else set()
        if l1_set and not l2_set:
            l2_set = present - l1_set
        if l2_set and not l1_set:
            l1_set = present - l2_set

        label1 = f"Layer 1 ({','.join(sorted(l1_set))})" if l1_set else "Layer 1"
        label2 = f"Layer 2 ({','.join(sorted(l2_set))})" if l2_set else "Layer 2"

        layer1_atoms = [a for a in all_atoms if a[1] in l1_set]
        layer2_atoms = [a for a in all_atoms if a[1] in l2_set or a[1] not in l1_set]

        comment1 = f"{label1} extracted from {input_path.name}; lattice+fractional coords unchanged."
        comment2 = f"{label2} extracted from {input_path.name}; lattice+fractional coords unchanged."

        _write_poscar_simple(poscar1, comment1, lattice_vecs, layer1_atoms)
        _write_poscar_simple(poscar2, comment2, lattice_vecs, layer2_atoms)
        _write_poscar_simple(combined_path, f"Original structure from {input_path.name}; lattice+fractional coords unchanged.", lattice_vecs, all_atoms)

        # Also provide CIF outputs
        cif1 = out_dir / f"{stem}_layer1.cif"
        cif2 = out_dir / f"{stem}_layer2.cif"
        # Keep the original CIF copy untouched; write combined CIF with a distinct name.
        cif_combined = out_dir / f"{stem}_combined.cif"
        _write_cif_simple(cif1, f"{stem}_layer1", cell, layer1_atoms, header_comment=comment1)
        _write_cif_simple(cif2, f"{stem}_layer2", cell, layer2_atoms, header_comment=comment2)
        _write_cif_simple(cif_combined, stem, cell, all_atoms, header_comment=f"Original structure from {input_path.name}; lattice+fractional coords unchanged.")

    print("Wrote split layers to:", out_dir)
    print(" - Original:", out_dir / input_path.name)
    print(" - Layer 1 :", poscar1)
    print(" - Layer 2 :", poscar2)
    print(" - Combined:", combined_path)
    cif_combined_path = out_dir / f"{stem}_combined.cif"
    if cif_combined_path.exists():
        print(" - CIF combined:", cif_combined_path)
    return out_dir

def get_primitive(struct):
    """Return primitive cell if found, else the original structure."""
    try:
        prim = struct.get_primitive_structure()
        # sometimes get_primitive returns same; ensure lattice volume not pathologically changed
        return prim
    except Exception as e:
        print("Warning: get_primitive_structure failed, returning original. Err:", e)
        return struct

def build_slab(struct, miller, min_slab_size, vacuum, center_slab=True):
    """Construct a slab using SlabGenerator. Returns the first termination slab."""
    sg = SlabGenerator(struct, miller_index=tuple(map(int, miller)), 
                       min_slab_size=min_slab_size, min_vacuum_size=vacuum, center_slab=center_slab)
    slabs = sg.get_slabs()
    if len(slabs) == 0:
        raise RuntimeError(f"No slabs generated for miller {miller}")
    return slabs[0]


def _interface_normal_unit_from_lattice(lattice: Lattice) -> np.ndarray:
    """
    Compute the unit normal of the slab surface from the in-plane lattice vectors.

    For a well-formed slab, a and b span the surface plane, and c is (approximately)
    along the surface normal. We compute n = (a x b) / |a x b| and orient it to have
    a positive dot product with c.
    """
    a_vec = np.array(lattice.matrix[0], dtype=float)
    b_vec = np.array(lattice.matrix[1], dtype=float)
    c_vec = np.array(lattice.matrix[2], dtype=float)
    n = np.cross(a_vec, b_vec)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise ValueError("Degenerate in-plane lattice vectors; cannot define surface normal.")
    n = n / norm
    if float(np.dot(n, c_vec)) < 0.0:
        n *= -1.0
    return n


def _element_symbol(site_species_string: str) -> str:
    """Extract element-like symbol from a species string (e.g., 'Pb2+' -> 'Pb')."""
    sym = "".join([ch for ch in site_species_string if ch.isalpha()]) or site_species_string
    return sym


def _top_layer_elements(struct: Structure, normal_unit: np.ndarray, layer_thickness: float = 1.5) -> Set[str]:
    """
    Return a set of element symbols found in the top-most atomic layer.

    The "top layer" is defined as sites whose projection along the surface normal
    is within `layer_thickness` Å of the maximum projection.
    """
    if len(struct) == 0:
        return set()
    proj = np.dot(np.asarray(struct.cart_coords, dtype=float), normal_unit)
    max_p = float(np.max(proj))
    mask = proj >= (max_p - float(layer_thickness))
    elems = {_element_symbol(struct[i].species_string) for i in np.where(mask)[0].tolist()}
    return elems


def _organic_components(struct: Structure) -> List[List[int]]:
    """
    Identify connected components for organic fragments (C/N/H only).

    This is a lightweight, distance-cutoff based connectivity finder designed to
    detect whether A-site organic cations (e.g., MA/FA) are intact at a surface termination.

    Returns
    -------
    components : list of list of int
        Each component is a list of site indices in `struct`.
    """
    _require_build_deps()
    # Bond cutoffs (Å): generous to avoid missing connectivity due to relaxation.
    cutoffs: Dict[Tuple[str, str], float] = {
        ("C", "N"): 1.75,
        ("N", "C"): 1.75,
        ("C", "H"): 1.25,
        ("H", "C"): 1.25,
        ("N", "H"): 1.25,
        ("H", "N"): 1.25,
    }

    organic_idx: List[int] = []
    organic_elem: Dict[int, str] = {}
    for i, site in enumerate(struct):
        elem = _element_symbol(site.species_string)
        if elem in {"C", "N", "H"}:
            organic_idx.append(i)
            organic_elem[i] = elem

    if not organic_idx:
        return []

    organic_set = set(organic_idx)
    adj: Dict[int, List[int]] = {i: [] for i in organic_idx}

    # Build adjacency by local neighbor search.
    # Use a single radius that covers the largest cutoff.
    radius = 2.2
    for i in organic_idx:
        elem_i = organic_elem[i]
        neighs = struct.get_neighbors(struct[i], r=radius)
        for nb in neighs:
            j = int(nb.index)
            if j not in organic_set or j == i:
                continue
            elem_j = organic_elem[j]
            key = (elem_i, elem_j)
            dmax = cutoffs.get(key)
            if dmax is None:
                continue
            if float(nb.nn_distance) <= float(dmax):
                adj[i].append(j)

    # Connected components (BFS)
    visited: Set[int] = set()
    comps: List[List[int]] = []
    for i in organic_idx:
        if i in visited:
            continue
        queue = [i]
        visited.add(i)
        comp: List[int] = []
        while queue:
            u = queue.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        comps.append(comp)
    return comps


def _classify_surface_organic_components(
    struct: Structure,
    normal_unit: np.ndarray,
    surface_window: float = 6.0,
    max_component_thickness: float = 4.5,
) -> Dict[str, int]:
    """
    Count intact organic A-site cations (MA/FA) and surface fragments near the top surface.

    The "top surface region" is defined as sites within `surface_window` Å of the
    maximum projection along the surface normal.
    """
    _require_build_deps()
    if len(struct) == 0:
        return {"fa_intact": 0, "fa_fragments": 0}

    proj = np.dot(np.asarray(struct.cart_coords, dtype=float), normal_unit)
    slab_top = float(np.max(proj))

    def in_top_region(indices: List[int]) -> bool:
        return any(float(proj[i]) >= slab_top - float(surface_window) for i in indices)

    comps = _organic_components(struct)
    fa_intact = 0
    ma_intact = 0
    fragments = 0
    for comp in comps:
        if not in_top_region(comp):
            continue
        elems = [_element_symbol(struct[i].species_string) for i in comp]
        c = elems.count("C")
        n = elems.count("N")
        h = elems.count("H")
        comp_max = float(np.max([proj[i] for i in comp]))
        comp_min = float(np.min([proj[i] for i in comp]))
        thickness = comp_max - comp_min

        # FA (formamidinium): CH(NH2)2+ -> C1 N2 H5 (often H count may vary slightly after editing/relaxation).
        is_fa = (c == 1) and (n == 2) and (h >= 4)
        # MA (methylammonium): CH3NH3+ -> C1 N1 H6 (allow small deviations).
        is_ma = (c == 1) and (n == 1) and (h >= 4)

        if is_fa and thickness <= float(max_component_thickness):
            fa_intact += 1
        elif is_ma and thickness <= float(max_component_thickness):
            ma_intact += 1
        else:
            fragments += 1
    return {
        "fa_intact": fa_intact,
        "ma_intact": ma_intact,
        "a_intact": fa_intact + ma_intact,
        "a_fragments": fragments,
    }


# Backwards-compatible alias (older internal name).
def _fa_molecule_components(struct: Structure) -> List[List[int]]:
    return _organic_components(struct)


def _classify_surface_fa_components(
    struct: Structure,
    normal_unit: np.ndarray,
    surface_window: float = 6.0,
    max_component_thickness: float = 4.5,
) -> Dict[str, int]:
    # Kept for backwards compatibility; prefer _classify_surface_organic_components.
    stats = _classify_surface_organic_components(
        struct,
        normal_unit,
        surface_window=surface_window,
        max_component_thickness=max_component_thickness,
    )
    return {"fa_intact": int(stats["fa_intact"]), "fa_fragments": int(stats["a_fragments"])}


def _surface_region_indices(struct: Structure, normal_unit: np.ndarray, which: str, window: float) -> List[int]:
    """Return site indices within `window` Å of the chosen surface ('top' or 'bottom')."""
    _require_build_deps()
    which = which.strip().lower()
    if which not in {"top", "bottom"}:
        raise ValueError("which must be 'top' or 'bottom'")
    if len(struct) == 0:
        return []
    proj = np.dot(np.asarray(struct.cart_coords, dtype=float), normal_unit)
    if which == "top":
        ref = float(np.max(proj))
        mask = proj >= (ref - float(window))
    else:
        ref = float(np.min(proj))
        mask = proj <= (ref + float(window))
    return np.where(mask)[0].tolist()


def _surface_layer_elements(struct: Structure, normal_unit: np.ndarray, which: str, thickness: float = 1.5) -> Set[str]:
    """
    Return element symbols in the outermost atomic layer of a surface.

    The layer is defined by a projection window of `thickness` Å from the extreme
    (max for 'top', min for 'bottom') along the slab normal.
    """
    _require_build_deps()
    idx = _surface_region_indices(struct, normal_unit, which=which, window=float(thickness))
    return {_element_symbol(struct[i].species_string) for i in idx}


def _surface_signature(struct: Structure, normal_unit: np.ndarray, which: str, window: float = 4.0) -> Dict[str, int]:
    """
    Compute a lightweight signature for a slab surface region.

    Returns booleans as ints: has_pb, has_i, has_c, has_n, plus fa_intact/fa_fragments.
    """
    idx = _surface_region_indices(struct, normal_unit, which=which, window=float(window))
    elems = {_element_symbol(struct[i].species_string) for i in idx}
    org = _classify_surface_organic_components(struct, normal_unit, surface_window=float(window), max_component_thickness=4.5)
    return {
        "has_pb": int("Pb" in elems),
        "has_i": int("I" in elems),
        "has_c": int("C" in elems),
        "has_n": int("N" in elems),
        "fa_intact": int(org["fa_intact"]),
        "ma_intact": int(org["ma_intact"]),
        "a_intact": int(org["a_intact"]),
        "a_fragments": int(org["a_fragments"]),
    }


def _flip_structure_along_normal(struct: Structure, normal_unit: np.ndarray) -> Structure:
    """
    Mirror a structure along the slab normal to swap top/bottom surfaces.

    This keeps the lattice unchanged and mirrors cartesian coordinates around the
    mid-plane between the minimum and maximum projection along the normal.
    """
    _require_build_deps()
    if len(struct) == 0:
        return struct.copy()
    cart = np.asarray(struct.cart_coords, dtype=float)
    proj = np.dot(cart, normal_unit)
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    delta = (pmin + pmax) - 2.0 * proj
    cart_new = cart + np.outer(delta, normal_unit)
    lat = struct.lattice
    frac_new = [lat.get_fractional_coords(v) for v in cart_new]
    return Structure(lat, struct.species, frac_new, coords_are_cartesian=False, to_unit_cell=True)


def _choose_perovskite_termination_slab(
    bulk: Structure,
    miller: Tuple[int, int, int],
    slab_thickness: float,
    vacuum: float,
    termination: str,
    top_layer_thickness: float = 4.0,
    symmetric: bool = True,
) -> Structure:
    """
    Build a perovskite slab with a user-selected terminating surface.

    Terminology (following common ABX3 perovskite conventions)
    - "PbI" termination: inorganic (B+X) rich surface, expected to expose Pb and I.
    - "FAI" termination: organic (A+X) rich surface, expected to expose C/N (FA) and I.

    Implementation detail
    ---------------------
    We generate all candidate terminations from SlabGenerator and score them based on
    the element set found in the top-most layer. This heuristic is robust enough for
    typical pseudo-cubic perovskite slabs used in DFT input generation.
    """
    _require_build_deps()
    term = termination.strip().upper()
    # Supported terminations for lead-iodide perovskites:
    # - PbI: inorganic termination
    # - FAI: FA + I termination
    # - MAI: MA + I termination
    # - AI: generic A-site organic + I termination (accepts MA and/or FA, useful for mixed-cation perovskites)
    if term not in {"PBI", "FAI", "MAI", "AI"}:
        raise ValueError("termination must be one of: 'PbI', 'FAI', 'MAI', 'AI'")

    sg = SlabGenerator(
        bulk,
        miller_index=tuple(map(int, miller)),
        min_slab_size=float(slab_thickness),
        min_vacuum_size=float(vacuum),
        center_slab=True,
    )
    # Avoid cutting organic cations by discouraging broken C–N, C–H, and N–H bonds.
    # For perovskite slabs intended for interfacial DFT, broken-molecule terminations
    # are unphysical and lead to large artifacts in electronic structure.
    bonds = {
        ("C", "N"): 1.75,
        ("C", "H"): 1.25,
        ("N", "H"): 1.25,
    }
    try:
        slabs = sg.get_slabs(bonds=bonds, max_broken_bonds=0, symmetrize=bool(symmetric))
    except TypeError:
        # Older pymatgen may not support these keyword arguments; fall back.
        slabs = sg.get_slabs()
    # If the strict "no broken bonds" filter eliminates all candidates, fall back to
    # unconstrained terminations and rely on the scoring function to reject fragments.
    if not slabs:
        try:
            slabs = sg.get_slabs(symmetrize=bool(symmetric))
        except TypeError:
            slabs = sg.get_slabs()
    if not slabs:
        raise RuntimeError(f"No slabs generated for miller {miller}")

    def _surface_ok(term_key: str, sig: Dict[str, int]) -> bool:
        """
        Return True if a surface signature matches the requested termination.

        PbI termination: exposed Pb/I, no organic atoms in the surface region.
        FAI termination: exposed FA (C/N/H) + I, no Pb exposure, and no FA fragments.
        """
        if term_key == "PBI":
            # For PbI termination we only require the *outermost layer* to be Pb/I-rich.
            # Organic atoms can appear slightly below the surface depending on molecular orientation.
            # (Symmetry enforcement for PbI is handled via layer-elements, not the 4 Å signature.)
            return (sig["has_pb"] == 1) and (sig["has_i"] == 1)
        if term_key == "FAI":
            return (
                (sig["has_i"] == 1)
                and (sig["has_pb"] == 0)
                and (sig.get("a_fragments", 0) == 0)
                and (sig.get("fa_intact", 0) >= 1)
            )
        if term_key == "MAI":
            return (
                (sig["has_i"] == 1)
                and (sig["has_pb"] == 0)
                and (sig.get("a_fragments", 0) == 0)
                and (sig.get("ma_intact", 0) >= 1)
            )
        # AI (mixed A-site): accept any intact A-site organic cation (MA and/or FA).
        return (
            (sig["has_i"] == 1)
            and (sig["has_pb"] == 0)
            and (sig.get("a_fragments", 0) == 0)
            and (sig.get("a_intact", 0) >= 1)
        )

    def score(slab: Structure) -> Tuple[int, int, int]:
        n = _interface_normal_unit_from_lattice(slab.lattice)
        elems = _top_layer_elements(slab, n, layer_thickness=float(top_layer_thickness))
        has_pb = "Pb" in elems
        has_i = "I" in elems
        has_c = "C" in elems
        has_n = "N" in elems
        has_cn = has_c or has_n
        org = _classify_surface_organic_components(slab, n, surface_window=6.0, max_component_thickness=4.5)
        fa_intact = int(org["fa_intact"])
        ma_intact = int(org["ma_intact"])
        a_intact = int(org["a_intact"])
        a_frag = int(org["a_fragments"])

        # Enforce symmetric slab termination: top and bottom surfaces must both match.
        if symmetric:
            if term == "PBI":
                top_layer = _surface_layer_elements(slab, n, which="top", thickness=1.5)
                bot_layer = _surface_layer_elements(slab, n, which="bottom", thickness=1.5)
                top_ok = ("Pb" in top_layer) and ("I" in top_layer) and ("C" not in top_layer) and ("N" not in top_layer)
                bot_ok = ("Pb" in bot_layer) and ("I" in bot_layer) and ("C" not in bot_layer) and ("N" not in bot_layer)
                if not (top_ok and bot_ok):
                    return (-999, -999, -999)
            else:
                top_sig = _surface_signature(slab, n, which="top", window=4.0)
                bot_sig = _surface_signature(slab, n, which="bottom", window=4.0)
                if not (_surface_ok(term, top_sig) and _surface_ok(term, bot_sig)):
                    return (-999, -999, -999)

        # Higher is better; tuple provides deterministic tie-breaking.
        if term == "PBI":
            # Prefer Pb + I on top, and penalize presence of C/N in the top layer.
            return (
                int(has_pb) + int(has_i),   # target species present
                -int(has_cn),               # avoid organic atoms on PbI termination
                -a_intact,                  # avoid intact organic cations at the top surface
                -a_frag,                    # avoid any surface fragments
            )
        # A-site iodide termination family (FAI/MAI/AI):
        # Prefer organic (C/N/H) + I on top, penalize Pb exposure, maximize intact molecules, avoid fragments.
        if term == "FAI":
            return (
                int(has_i) + int(has_c) + int(has_n),
                fa_intact,
                -a_frag,
                -int(has_pb),
            )
        if term == "MAI":
            return (
                int(has_i) + int(has_c) + int(has_n),
                ma_intact,
                -a_frag,
                -int(has_pb),
            )
        # AI (mixed): maximize total intact A-site organics.
        return (
            int(has_i) + int(has_c) + int(has_n),
            a_intact,
            -a_frag,
            -int(has_pb),
        )

    scored = [(score(s), s) for s in slabs]
    scored.sort(key=lambda x: x[0], reverse=True)
    if symmetric and scored and scored[0][0] == (-999, -999, -999):
        raise RuntimeError(
            f"Could not find a symmetric slab with '{termination}' termination on both surfaces for miller={miller}. "
            "Try increasing slab_thickness, changing miller index, or disable symmetry only for debugging."
        )
    return scored[0][1]


def _load_adsorbate_structure(path: str) -> Structure:
    """
    Load an adsorbate structure (e.g., C60) as a structure object.

    For typical molecule CIFs, the cell is artificial; we use only the atomic positions.
    """
    _require_build_deps()
    return Structure.from_file(path)


def _geometric_center(cart_coords: np.ndarray) -> np.ndarray:
    """Return the geometric center (average position) of a set of cartesian coords."""
    if cart_coords.size == 0:
        return np.zeros(3)
    return np.mean(cart_coords, axis=0)

def _estimate_molecule_diameter(cart_coords: np.ndarray) -> float:
    """
    Estimate a molecule diameter (Å) from cartesian coordinates.

    We approximate the molecule by its maximum distance from the geometric center:
        diameter ~= 2 * max_i ||r_i - r_center||

    This is a fast O(N) heuristic suitable for fullerenes (C60/C70) and similar molecules.
    """
    if cart_coords.size == 0:
        return 0.0
    center = _geometric_center(cart_coords)
    radii = np.linalg.norm(cart_coords - center, axis=1)
    return float(2.0 * np.max(radii))


def _make_inplane_supercell(struct: Structure, nx: int, ny: int) -> Structure:
    """
    Make an in-plane supercell (nx, ny, 1) while keeping the slab normal/vacuum unchanged.
    """
    _require_build_deps()
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    sc_mat = [[int(nx), 0, 0], [0, int(ny), 0], [0, 0, 1]]
    return SupercellTransformation(sc_mat).apply_transformation(struct)


def _suggest_supercell_xy_for_c60(
    slab: Structure,
    c60_diameter: float = 7.1,
    buffer: float = 3.0,
    max_atoms: int = 400,
    c60_atoms: int = 60,
) -> Tuple[int, int]:
    """
    Suggest the smallest (nx, ny) so that the in-plane cell lengths are large enough
    to avoid excessive C60–C60 interactions across periodic images.

    Heuristic
    ---------
    Require a_len >= (c60_diameter + buffer) and b_len >= (c60_diameter + buffer).
    Then check the approximate atom count constraint:
        n_total = n_slab * nx * ny + c60_atoms

    Notes
    -----
    In plane-wave DFT with 3D periodic boundary conditions, the adsorbate is always
    periodic in x/y. Making a larger in-plane supercell reduces spurious interactions
    between periodic images of C60 (unless you intentionally want a dense monolayer).
    """
    _require_build_deps()
    a_len = float(np.linalg.norm(np.asarray(slab.lattice.matrix[0], dtype=float)))
    b_len = float(np.linalg.norm(np.asarray(slab.lattice.matrix[1], dtype=float)))
    target = float(c60_diameter + buffer)
    nx0 = max(1, int(math.ceil(target / max(a_len, 1e-8))))
    ny0 = max(1, int(math.ceil(target / max(b_len, 1e-8))))

    n_slab = len(slab)
    n_total = n_slab * nx0 * ny0 + int(c60_atoms)
    if n_total > int(max_atoms):
        raise RuntimeError(
            f"Auto supercell suggests {nx0}x{ny0}, but estimated atoms={n_total} exceeds max_atoms={max_atoms}. "
            f"Try reducing slab thickness, lowering buffer, or increasing max_atoms."
        )
    return nx0, ny0


def _minimum_image_distance_xy(struct: Structure) -> float:
    """
    Compute the minimum distance between any atom and its periodic images in x/y,
    approximated by the shortest in-plane lattice vector length.

    This is a conservative indicator for whether the in-plane cell is 'large enough'
    for isolated adsorbates; it does not replace a full minimum-image distance check.
    """
    a_len = float(np.linalg.norm(np.asarray(struct.lattice.matrix[0], dtype=float)))
    b_len = float(np.linalg.norm(np.asarray(struct.lattice.matrix[1], dtype=float)))
    return float(min(a_len, b_len))


def build_perovskite_c60_adsorbate(
    base_cif: str,
    c60_cif: str,
    miller: Tuple[int, int, int] = (0, 0, 1),
    slab_thickness: float = 20.0,
    vacuum: float = 20.0,
    termination: str = "PbI",
    adsorbate_distance: float = 3.5,
    adsorbate_xy: Optional[Tuple[float, float]] = None,
    supercell_xy: Optional[Tuple[int, int]] = None,
    auto_supercell: bool = False,
    c60_diameter: float = 7.1,
    c60_buffer: float = 3.0,
    max_atoms: int = 400,
    symmetric_slab: bool = True,
) -> Tuple[Structure, Structure]:
    """
    Build a perovskite slab and place a single adsorbate molecule above the selected surface.

    Parameters
    ----------
    base_cif
        Path to the perovskite bulk CIF (used to generate the slab).
    c60_cif
        Path to the adsorbate CIF (molecule), e.g. C60 or C70.
    miller
        Miller index for the slab cut.
    slab_thickness
        Minimum slab thickness in Å (passed to SlabGenerator).
    vacuum
        Minimum vacuum thickness in Å (passed to SlabGenerator).
    termination
        Termination mode for lead-iodide perovskites:
        - 'PbI': inorganic termination
        - 'FAI': FA + I termination
        - 'MAI': MA + I termination
        - 'AI' : generic A-site organic + I termination (accepts MA and/or FA; useful for mixed-cation perovskites)
    adsorbate_distance
        Target distance (Å) between the top-most slab atom and the lowest atom of the adsorbate
        along the surface normal.
    adsorbate_xy
        Optional (fx, fy) fractional coordinates in the slab cell to place the C60 center
        laterally. If None, defaults to the cell center (0.5, 0.5).
    supercell_xy
        Optional (nx, ny) in-plane supercell to enlarge the slab cell before placing the adsorbate.
        This controls the lateral periodicity of the model.
    auto_supercell
        If True, pick the smallest (nx, ny) so that in-plane lengths exceed
        (c60_diameter + c60_buffer) while keeping atoms <= max_atoms.
    c60_diameter
        Approximate adsorbate diameter in Å (used only for auto_supercell heuristics).
    c60_buffer
        Extra spacing in Å added on top of c60_diameter for auto_supercell.
    max_atoms
        Maximum allowed atom count (used for auto_supercell feasibility checks).

    Returns
    -------
    slab
        The selected-termination slab structure (with vacuum).
    combined
        Combined structure (slab + adsorbate) in a single periodic cell.
    """
    _require_build_deps()
    if adsorbate_xy is None:
        adsorbate_xy = (0.5, 0.5)

    # Delegate to the multi-adsorbate stack builder for consistency.
    return build_perovskite_adsorbate_stack(
        base_cif=base_cif,
        adsorbate_cifs=[c60_cif],
        layer_gaps=[float(adsorbate_distance)],
        miller=miller,
        slab_thickness=slab_thickness,
        vacuum=vacuum,
        termination=termination,
        adsorbate_xy=adsorbate_xy,
        supercell_xy=supercell_xy,
        auto_supercell=auto_supercell,
        adsorbate_diameter=float(c60_diameter),
        adsorbate_buffer=float(c60_buffer),
        max_atoms=max_atoms,
        symmetric_slab=symmetric_slab,
    )


def build_perovskite_adsorbate_stack(
    base_cif: str,
    adsorbate_cifs: List[str],
    layer_gaps: List[float],
    miller: Tuple[int, int, int] = (0, 0, 1),
    slab_thickness: float = 20.0,
    vacuum: float = 20.0,
    termination: str = "PbI",
    adsorbate_xy: Optional[Tuple[float, float]] = None,
    supercell_xy: Optional[Tuple[int, int]] = None,
    auto_supercell: bool = False,
    adsorbate_diameter: float = 0.0,
    adsorbate_buffer: float = 3.0,
    max_atoms: int = 400,
    symmetric_slab: bool = True,
) -> Tuple[Structure, Structure]:
    """
    Build a perovskite slab and place multiple adsorbates above it to form a multilayer stack.

    Example: perovskite (bottom) + C70 (middle) + C60 (top), i.e. 3 layers / 2 interfaces.

    Parameters
    ----------
    base_cif
        Perovskite bulk CIF used to generate the slab.
    adsorbate_cifs
        List of adsorbate CIF paths in stacking order (bottom -> top), e.g. [C70, C60].
    layer_gaps
        List of interlayer gaps in Å. Must have the same length as adsorbate_cifs.
        Interpretation:
          layer_gaps[0] = gap between slab top and adsorbate[0] bottom (along surface normal)
          layer_gaps[i] = gap between adsorbate[i-1] top and adsorbate[i] bottom
    adsorbate_xy
        Lateral placement (fx, fy) for the geometric center of each adsorbate (same for all).
    auto_supercell
        If True, choose an in-plane supercell to reduce periodic-image interactions using:
            target_inplane >= (adsorbate_diameter + adsorbate_buffer)
        If adsorbate_diameter <= 0, the code estimates a diameter from the largest adsorbate.
    """
    _require_build_deps()
    if adsorbate_xy is None:
        adsorbate_xy = (0.5, 0.5)
    if not adsorbate_cifs:
        raise ValueError("adsorbate_cifs must contain at least one CIF path.")
    if len(layer_gaps) != len(adsorbate_cifs):
        raise ValueError("layer_gaps must have the same length as adsorbate_cifs.")

    bulk = load_structure(base_cif)
    slab = _choose_perovskite_termination_slab(
        bulk=bulk,
        miller=miller,
        slab_thickness=slab_thickness,
        vacuum=vacuum,
        termination=termination,
        symmetric=bool(symmetric_slab),
    )

    # Optional in-plane supercelling to control the lateral periodicity.
    if auto_supercell:
        # If user did not specify a diameter, estimate from the largest adsorbate.
        dia = float(adsorbate_diameter)
        if dia <= 1e-6:
            dias: List[float] = []
            for p in adsorbate_cifs:
                mol = _load_adsorbate_structure(p)
                dias.append(_estimate_molecule_diameter(np.asarray(mol.cart_coords, dtype=float)))
            dia = float(max(dias)) if dias else 0.0
            print(f"Estimated adsorbate diameter (max over stack): {dia:.2f} Å")

        # Crude estimate of added atoms from all adsorbates.
        ads_atoms = 0
        for p in adsorbate_cifs:
            ads_atoms += len(_load_adsorbate_structure(p))
        nx, ny = _suggest_supercell_xy_for_c60(
            slab=slab,
            c60_diameter=float(dia),
            buffer=float(adsorbate_buffer),
            max_atoms=int(max_atoms),
            c60_atoms=int(ads_atoms),
        )
        print(f"Auto supercell selected: {nx}x{ny} (target >= {dia + adsorbate_buffer:.2f} Å in-plane)")
        slab = _make_inplane_supercell(slab, nx, ny)
    elif supercell_xy is not None:
        nx, ny = int(supercell_xy[0]), int(supercell_xy[1])
        print(f"Using user supercell: {nx}x{ny}")
        slab = _make_inplane_supercell(slab, nx, ny)

    # Surface normal (after supercell) and slab top reference.
    n = _interface_normal_unit_from_lattice(slab.lattice)
    top_elems = _top_layer_elements(slab, n, layer_thickness=4.0)
    print(f"Selected slab top-layer elements: {sorted(top_elems)} (requested termination: {termination})")
    print(f"In-plane minimum lattice length (a/b): {_minimum_image_distance_xy(slab):.2f} Å (periodic in x/y)")
    slab_proj = np.dot(np.asarray(slab.cart_coords, dtype=float), n) if len(slab) else np.array([0.0])
    current_top = float(np.max(slab_proj))

    # Place adsorbates sequentially along the surface normal.
    placed_adsorbates_cart: List[np.ndarray] = []
    for idx, (ads_path, gap) in enumerate(zip(adsorbate_cifs, layer_gaps)):
        ads = _load_adsorbate_structure(ads_path)
        ads_cart = np.asarray(ads.cart_coords, dtype=float)
        ads_center = _geometric_center(ads_cart)
        ads_cart_centered = ads_cart - ads_center

        # Lateral placement by geometric center.
        delta_xy_cart = slab.lattice.get_cartesian_coords(
            np.array([float(adsorbate_xy[0]), float(adsorbate_xy[1]), 0.0], dtype=float)
        )
        ads_cart_positioned = ads_cart_centered + delta_xy_cart

        # Lift so that the minimum projection sits 'gap' above the current top.
        ads_proj = np.dot(ads_cart_positioned, n)
        ads_min = float(np.min(ads_proj)) if ads_proj.size else 0.0
        lift = (current_top + float(gap)) - ads_min
        ads_cart_positioned = ads_cart_positioned + n * lift

        # Update current top using this adsorbate max.
        ads_proj2 = np.dot(ads_cart_positioned, n)
        current_top = float(np.max(ads_proj2)) if ads_proj2.size else current_top
        placed_adsorbates_cart.append(ads_cart_positioned)
        print(f"Placed adsorbate {idx+1}/{len(adsorbate_cifs)}: {Path(ads_path).name} | gap={gap:.2f} Å")

    # Ensure the cell is tall enough so the top-most adsorbate does not cross the periodic boundary.
    c_len = float(np.linalg.norm(np.array(slab.lattice.matrix[2], dtype=float)))
    clearance = c_len - float(current_top)
    if clearance < 5.0:
        extra = 5.0 - clearance + 2.0  # small buffer
        new_lat = np.array(slab.lattice.matrix, dtype=float)
        new_lat[2] = new_lat[2] + n * extra
        new_lattice = Lattice(new_lat)
        slab_cart = np.asarray(slab.cart_coords, dtype=float)
        slab_frac = [new_lattice.get_fractional_coords(v) for v in slab_cart]
        slab = Structure(new_lattice, slab.species, slab_frac, coords_are_cartesian=False, to_unit_cell=True)
        print(f"Extended vacuum by {extra:.2f} Å to keep clearance above top layer.")

    # Merge slab + all adsorbates into a combined structure in the slab lattice.
    combined = slab.copy()
    lat = combined.lattice
    for ads_path, ads_cart_positioned in zip(adsorbate_cifs, placed_adsorbates_cart):
        ads = _load_adsorbate_structure(ads_path)
        for sp, cart in zip(ads.species, ads_cart_positioned):
            frac = lat.get_fractional_coords(cart)
            combined.append(sp, frac, coords_are_cartesian=False)
    combined = Structure(lat, combined.species, combined.frac_coords, coords_are_cartesian=False, to_unit_cell=True)
    combined = combined.get_sorted_structure()
    return slab, combined

    bulk = load_structure(base_cif)
    slab = _choose_perovskite_termination_slab(
        bulk=bulk,
        miller=miller,
        slab_thickness=slab_thickness,
        vacuum=vacuum,
        termination=termination,
        symmetric=bool(symmetric_slab),
    )

    # If symmetry is disabled (debug mode), keep the adsorption surface preference by flipping.
    if not symmetric_slab:
        n0 = _interface_normal_unit_from_lattice(slab.lattice)
        top_sig = _surface_signature(slab, n0, which="top", window=4.0)
        bot_sig = _surface_signature(slab, n0, which="bottom", window=4.0)

        term_up = termination.strip().upper()
        if term_up == "FAI":
            def _score_fai(sig: Dict[str, int]) -> float:
                return (
                    3.0 * sig["has_i"]
                    + 1.0 * sig["has_c"]
                    + 1.0 * sig["has_n"]
                    + 3.0 * sig["fa_intact"]
                    - 6.0 * sig["fa_fragments"]
                    - 2.0 * sig["has_pb"]
                )
            if _score_fai(bot_sig) > _score_fai(top_sig) + 1e-6:
                slab = _flip_structure_along_normal(slab, n0)
                print("Flipped slab to expose FAI termination on the top surface.")
        else:
            def _score_pbi(sig: Dict[str, int]) -> float:
                return (
                    2.0 * sig["has_pb"]
                    + 2.0 * sig["has_i"]
                    - 2.0 * sig["has_c"]
                    - 2.0 * sig["has_n"]
                    - 3.0 * sig["fa_intact"]
                    - 3.0 * sig["fa_fragments"]
                )
            if _score_pbi(bot_sig) > _score_pbi(top_sig) + 1e-6:
                slab = _flip_structure_along_normal(slab, n0)
                print("Flipped slab to expose PbI termination on the top surface.")

    # Optional in-plane supercelling to control the lateral periodicity.
    if auto_supercell:
        nx, ny = _suggest_supercell_xy_for_c60(
            slab=slab,
            c60_diameter=float(c60_diameter),
            buffer=float(c60_buffer),
            max_atoms=int(max_atoms),
            c60_atoms=60,
        )
        print(f"Auto supercell selected: {nx}x{ny} (target >= {c60_diameter + c60_buffer:.2f} Å in-plane)")
        slab = _make_inplane_supercell(slab, nx, ny)
    elif supercell_xy is not None:
        nx, ny = int(supercell_xy[0]), int(supercell_xy[1])
        print(f"Using user supercell: {nx}x{ny}")
        slab = _make_inplane_supercell(slab, nx, ny)

    # Surface normal and slab top reference (cartesian projection).
    n = _interface_normal_unit_from_lattice(slab.lattice)
    # Use a thicker "top region" to reflect A-site cations sitting above the X layer
    # (as commonly shown in MAI/FAI-terminated surface schematics).
    top_elems = _top_layer_elements(slab, n, layer_thickness=4.0)
    print(f"Selected slab top-layer elements: {sorted(top_elems)} (requested termination: {termination})")
    print(f"In-plane minimum lattice length (a/b): {_minimum_image_distance_xy(slab):.2f} Å (periodic in x/y)")
    slab_proj = np.dot(np.asarray(slab.cart_coords, dtype=float), n) if len(slab) else np.array([0.0])
    slab_top = float(np.max(slab_proj))

    # Load adsorbate and shift it to match requested geometry.
    c60 = _load_adsorbate_structure(c60_cif)
    c60_cart = np.asarray(c60.cart_coords, dtype=float)
    c60_center = _geometric_center(c60_cart)
    c60_cart_centered = c60_cart - c60_center

    # First, place C60 laterally by setting its geometric center to the desired (fx, fy) in the slab cell.
    # Using cartesian translation from fractional coordinates handles non-orthogonal a/b lattices correctly.
    delta_xy_cart = slab.lattice.get_cartesian_coords(
        np.array([float(adsorbate_xy[0]), float(adsorbate_xy[1]), 0.0], dtype=float)
    )
    c60_cart_positioned = c60_cart_centered + delta_xy_cart

    # Then, lift C60 above the slab to enforce the requested minimum distance along the normal.
    c60_proj = np.dot(c60_cart_positioned, n)
    c60_min = float(np.min(c60_proj)) if c60_proj.size else 0.0
    lift = (slab_top + float(adsorbate_distance)) - c60_min
    c60_cart_positioned = c60_cart_positioned + n * lift

    # Ensure the cell is tall enough so the adsorbate does not cross the periodic boundary.
    # If needed, extend the c vector (keeping a and b fixed) to add extra vacuum above.
    c_len = float(np.linalg.norm(np.array(slab.lattice.matrix[2], dtype=float)))
    max_proj_all = float(np.max(np.concatenate([slab_proj, np.dot(c60_cart_positioned, n)])))
    # Keep at least 5 Å clearance to the top boundary.
    clearance = c_len - max_proj_all
    if clearance < 5.0:
        extra = 5.0 - clearance + 2.0  # small buffer
        new_lat = np.array(slab.lattice.matrix, dtype=float)
        new_lat[2] = new_lat[2] + n * extra
        new_lattice = Lattice(new_lat)
        # Rebuild slab in new lattice (preserve cart coords).
        slab_cart = np.asarray(slab.cart_coords, dtype=float)
        slab_frac = [new_lattice.get_fractional_coords(v) for v in slab_cart]
        slab = Structure(new_lattice, slab.species, slab_frac, coords_are_cartesian=False, to_unit_cell=True)
        # Update C60 in the new lattice coordinates.
        c60_cart_positioned = c60_cart_positioned  # unchanged in cart space

    # Merge slab + C60 into a combined structure in the slab lattice.
    combined = slab.copy()
    lat = combined.lattice
    for sp, cart in zip(c60.species, c60_cart_positioned):
        frac = lat.get_fractional_coords(cart)
        combined.append(sp, frac, coords_are_cartesian=False)
    # Wrap to the unit cell for VASP friendliness, then sort for stable output.
    combined = Structure(lat, combined.species, combined.frac_coords, coords_are_cartesian=False, to_unit_cell=True)
    combined = combined.get_sorted_structure()
    return slab, combined

def search_matches(
    structA,
    structB,
    miller_a,
    miller_b,
    tol: float = 0.03,
    max_area: float = 500,
    max_match: int = 20,
    allow_bidirectional: bool = True,
):
    """
    Use CoherentInterfaceBuilder to find candidate in-plane supercells that match.
    Requires bulk structures and Miller indices, not slabs.
    Returns a list of match dicts with strains and transforms.
    """
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator

    param_grid = _generate_zsl_parameter_grid(tol, max_area, allow_bidirectional=allow_bidirectional)
    matches: List[dict] = []
    seen_keys: Set[Tuple[int, ...]] = set()

    for params in param_grid:
        try:
            zslgen = ZSLGenerator(
                max_area=params["max_area"],
                max_length_tol=params["max_length_tol"],
                max_angle_tol=params["max_angle_tol"],
                bidirectional=params["bidirectional"],
            )
        except TypeError:
            # Legacy pymatgen versions may not support all keyword arguments.
            zslgen = ZSLGenerator(max_area=params["max_area"])

        matcher = CoherentInterfaceBuilder(
            substrate_structure=structB,
            film_structure=structA,
            film_miller=miller_a,
            substrate_miller=miller_b,
            zslgen=zslgen,
        )

        zsl_matches = matcher.zsl_matches or []
        if len(zsl_matches) == 0:
            continue

        for idx, match_obj in enumerate(zsl_matches[: max_match or None]):
            film_trans = np.array(match_obj.film_transformation)
            sub_trans = (
                np.array(match_obj.substrate_transformation)
                if hasattr(match_obj, "substrate_transformation")
                else np.eye(2)
            )
            key = _matrix_key(film_trans, sub_trans)
            if key in seen_keys:
                continue

            metrics = _compute_match_metrics(match_obj)
            summary = {
                "area": float(match_obj.match_area),
                "film_transformation": film_trans,
                "substrate_transformation": sub_trans,
                "supercell_a": _extend_matrix_2d_to_3d(film_trans),
                "supercell_b": _extend_matrix_2d_to_3d(sub_trans),
                "det_a": float(abs(np.linalg.det(film_trans))),
                "det_b": float(abs(np.linalg.det(sub_trans))),
                "length_components": metrics["length_components"],
                "length_mismatch": metrics["max_length_mismatch"],
                "angle_mismatch": metrics["angle_mismatch"],
                "angles": (metrics["film_angle"], metrics["substrate_angle"]),
                "film_lengths": metrics["film_lengths"],
                "substrate_lengths": metrics["substrate_lengths"],
                "zsl_params": params,
                "priority": (
                    params["max_area"],
                    params["max_length_tol"],
                    0 if params["bidirectional"] else 1,
                    idx,
                ),
                "match_index": idx,
            }
            matches.append(summary)
            seen_keys.add(key)

    matches.sort(key=lambda m: m["priority"])
    if max_match:
        matches = matches[:max_match]
    return matches

def compute_inplane_norms(latt):
    """Return norms of the first two lattice vectors (in-plane)."""
    mat = np.array(latt.matrix)
    a = np.linalg.norm(mat[0])
    b = np.linalg.norm(mat[1])
    return a, b

def apply_inplane_strain(slab, scale_x, scale_y):
    """
    Return a copy of slab with in-plane lattice vectors scaled by scale_x, scale_y.
    Preserves the c vector magnitude and its relative orientation to maintain slab thickness.
    """
    new_lat = slab.lattice.matrix.copy()
    # Scale in-plane vectors
    new_lat[0] = new_lat[0] * scale_x
    new_lat[1] = new_lat[1] * scale_y
    # Keep c vector unchanged to preserve slab thickness
    # The c vector will be adjusted later in align_and_stack_ordered if needed
    new_lattice = Lattice(new_lat)
    s2 = Structure(new_lattice, slab.species, slab.frac_coords)
    return s2

def align_and_stack_ordered(slab_bottom, slab_top, separation=3.2, vacuum=20.0):
    """
    Align and stack two slabs with proper lattice matching.
    Uses the original slab lattices directly without modification to preserve structure integrity.
    Returns bottom, top, and combined structures, where bottom and top are properly separated.
    
    When vacuum=0 or very small, special care is taken to prevent atoms from crossing
    layer boundaries due to periodic boundary wrapping.
    """
    # Copy slabs to avoid modifying originals
    bottom = slab_bottom.copy()
    top = slab_top.copy()
    
    # Use bottom's lattice as the reference (it should match top after straining)
    # Get in-plane vectors from bottom to determine the interface normal.
    bottom_lat = bottom.lattice.matrix
    a_vec = bottom_lat[0]
    b_vec = bottom_lat[1]

    # Calculate surface normal from in-plane vectors
    ab_normal = np.cross(a_vec, b_vec)
    ab_normal_norm = np.linalg.norm(ab_normal)
    if ab_normal_norm < 1e-10:
        raise ValueError("In-plane vectors are parallel or degenerate")
    ab_normal_unit = ab_normal / ab_normal_norm
    # Ensure the surface normal points in the same general direction as the original c vector.
    if np.dot(ab_normal_unit, bottom_lat[2]) < 0:
        ab_normal_unit *= -1.0

    # Align bottom slab along the interface normal so the minimum projection is 0.
    if len(bottom) > 0:
        bottom_proj = np.dot(bottom.cart_coords, ab_normal_unit)
        min_proj_bottom = bottom_proj.min()
        bottom.translate_sites(
            list(range(len(bottom))),
            -ab_normal_unit * min_proj_bottom,
        )
        bottom_proj = np.dot(bottom.cart_coords, ab_normal_unit)
        max_proj_bottom = bottom_proj.max()
    else:
        max_proj_bottom = 0.0

    # Align top slab so its minimum projection is at the desired separation.
    if len(top) > 0:
        top_proj = np.dot(top.cart_coords, ab_normal_unit)
        min_proj_top = top_proj.min()
        top.translate_sites(
            list(range(len(top))),
            -ab_normal_unit * min_proj_top,
        )
        top.translate_sites(
            list(range(len(top))),
            ab_normal_unit * (max_proj_bottom + separation),
        )
        top_proj = np.dot(top.cart_coords, ab_normal_unit)
        desired_min = max_proj_bottom + separation + 0.1
        min_proj_top_post = top_proj.min()
        if min_proj_top_post < desired_min - 1e-4:
            adjust = desired_min - min_proj_top_post
            top.translate_sites(list(range(len(top))), ab_normal_unit * adjust)
            top_proj = np.dot(top.cart_coords, ab_normal_unit)
        max_proj_top = top_proj.max()
    else:
        max_proj_top = max_proj_bottom + separation

    # Determine combined cell length along the interface normal and construct the new lattice.
    total_z = max_proj_top + vacuum
    
    # Verify top's in-plane vectors match (should be true after straining)
    top_lat = top.lattice.matrix
    if np.linalg.norm(a_vec - top_lat[0]) > 0.01 or np.linalg.norm(b_vec - top_lat[1]) > 0.01:
        print("Warning: In-plane lattice vectors don't match exactly. Using bottom slab vectors.")
        # Create new lattice for top using bottom's in-plane vectors but preserving top's c
        new_top_lat = top_lat.copy()
        new_top_lat[0] = a_vec
        new_top_lat[1] = b_vec
        top_lattice_new = Lattice(new_top_lat)
        # Recalculate fractional coordinates with new lattice
        top_cart_coords = top.cart_coords
        top = Structure(top_lattice_new, top.species, 
                       [top_lattice_new.get_fractional_coords(coord) for coord in top_cart_coords],
                       coords_are_cartesian=False)
    # Create c vector perpendicular to ab plane with correct magnitude
    c_vec = ab_normal_unit * total_z
    
    # Build combined lattice using bottom's in-plane vectors and perpendicular c
    combined_lat_matrix = np.vstack([a_vec, b_vec, c_vec])
    combined_lattice = Lattice(combined_lat_matrix)
    
    # Calculate the interface z-boundary in fractional coordinates
    # The interface is between max_proj_bottom and (max_proj_bottom + separation)
    # We use the midpoint as the boundary
    interface_z_cart = max_proj_bottom + separation / 2.0
    interface_z_frac = interface_z_cart / total_z if total_z > 1e-8 else 0.5
    
    # Convert bottom slab to use combined lattice with layer-aware wrapping
    bottom_cart_coords = bottom.cart_coords
    bottom_frac_coords = []
    for coord in bottom_cart_coords:
        fc = combined_lattice.get_fractional_coords(coord)
        # Wrap x and y to [0, 1)
        fc_x = fc[0] % 1.0
        fc_y = fc[1] % 1.0
        # For z: bottom atoms should stay below interface_z_frac
        fc_z = fc[2] % 1.0
        # If wrapping moved the atom to the top region, shift it back down
        if fc_z > interface_z_frac + 0.1:
            fc_z = fc_z - 1.0
        bottom_frac_coords.append(np.array([fc_x, fc_y, fc_z]))
    
    bottom_separated = Structure(combined_lattice,
                                bottom.species,
                                bottom_frac_coords,
                                coords_are_cartesian=False)
    
    # Convert top slab to use combined lattice with layer-aware wrapping
    top_cart_coords = top.cart_coords
    top_frac_coords = []
    for coord in top_cart_coords:
        fc = combined_lattice.get_fractional_coords(coord)
        # Wrap x and y to [0, 1)
        fc_x = fc[0] % 1.0
        fc_y = fc[1] % 1.0
        # For z: top atoms should stay above interface_z_frac
        fc_z = fc[2] % 1.0
        # If wrapping moved the atom to the bottom region, shift it back up
        if fc_z < interface_z_frac - 0.1:
            fc_z = fc_z + 1.0
        top_frac_coords.append(np.array([fc_x, fc_y, fc_z]))
    
    top_separated = Structure(combined_lattice,
                            top.species,
                            top_frac_coords,
                            coords_are_cartesian=False)
    
    # Create combined structure by adding all sites
    combined = Structure(combined_lattice, [], [])
    
    # Track indices for potential correction
    n_bottom = len(bottom_separated)
    
    # Add bottom slab sites first
    for site in bottom_separated:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
    
    # Add top slab sites
    for site in top_separated:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
    
    # Final verification and correction for small vacuum cases
    if vacuum < 2.0:
        corrections_made = 0
        # Check bottom atoms - should be below interface
        for i in range(n_bottom):
            site = combined[i]
            if site.frac_coords[2] > interface_z_frac + 0.05:
                new_z = site.frac_coords[2] - 1.0
                new_frac = np.array([site.frac_coords[0], site.frac_coords[1], new_z])
                combined.replace(i, site.species_string, new_frac, coords_are_cartesian=False)
                corrections_made += 1
        # Check top atoms - should be above interface
        for i in range(n_bottom, len(combined)):
            site = combined[i]
            if site.frac_coords[2] < interface_z_frac - 0.05:
                new_z = site.frac_coords[2] + 1.0
                new_frac = np.array([site.frac_coords[0], site.frac_coords[1], new_z])
                combined.replace(i, site.species_string, new_frac, coords_are_cartesian=False)
                corrections_made += 1
        
        if corrections_made > 0:
            print(f"Layer crossing prevention: Corrected {corrections_made} atom(s) at periodic boundary")
            # Rebuild separated structures from combined to ensure consistency
            bottom_separated = Structure(combined_lattice,
                                        [combined[i].species_string for i in range(n_bottom)],
                                        [combined[i].frac_coords for i in range(n_bottom)],
                                        coords_are_cartesian=False)
            top_separated = Structure(combined_lattice,
                                     [combined[i].species_string for i in range(n_bottom, len(combined))],
                                     [combined[i].frac_coords for i in range(n_bottom, len(combined))],
                                     coords_are_cartesian=False)
    
    # Sort combined by z-coordinate for better ordering
    combined.sort(key=lambda site: site.frac_coords[2])
    
    # Verify separation: ensure bottom and top are properly separated
    if len(bottom_separated) > 0 and len(top_separated) > 0:
        c_unit = ab_normal_unit
        bottom_proj = np.dot(bottom_separated.cart_coords, c_unit)
        top_proj = np.dot(top_separated.cart_coords, c_unit)
        bottom_proj_max = bottom_proj.max()
        top_proj_min = top_proj.min()
        gap_size = top_proj_min - bottom_proj_max
        if gap_size < 0:
            print(
                f"Warning: Overlap detected along interface normal! "
                f"bottom_max={bottom_proj_max:.2f} Å, top_min={top_proj_min:.2f} Å"
            )
        elif gap_size < 0.5:
            print(f"Warning: Very small gap ({gap_size:.2f} Å) along interface normal.")
    
    return bottom_separated, top_separated, combined

def write_poscars(bottom, top, combined, output_dir: Path, base_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    bottom_for_io = _reorder_structure_for_poscar(bottom)
    top_for_io = _reorder_structure_for_poscar(top)
    combined_for_io = _reorder_structure_for_poscar(combined)

    bottom_path = output_dir / f"{base_name}_bottom_strained.vasp"
    top_path = output_dir / f"{base_name}_top_strained.vasp"
    combined_path = output_dir / f"{base_name}.vasp"

    Poscar(bottom_for_io, sort_structure=False).write_file(bottom_path)
    Poscar(top_for_io, sort_structure=False).write_file(top_path)
    Poscar(combined_for_io, sort_structure=False).write_file(combined_path)

    print(
        "Wrote POSCARs:",
        bottom_path,
        top_path,
        combined_path,
    )

def estimate_atoms_for_match(structA, structB, miller_a, miller_b, supercell_a, supercell_b, 
                             slab_thickness_a, slab_thickness_b):
    """Estimate total number of atoms for a given match."""
    # Build temporary slabs to estimate atom count
    try:
        slabA_temp = build_slab(structA, miller_a, slab_thickness_a, 0, center_slab=False)
        slabB_temp = build_slab(structB, miller_b, slab_thickness_b, 0, center_slab=False)
        
        # Apply supercell transformations
        transA = [[int(supercell_a[0,0]), int(supercell_a[0,1]), int(supercell_a[0,2])],
                  [int(supercell_a[1,0]), int(supercell_a[1,1]), int(supercell_a[1,2])],
                  [int(supercell_a[2,0]), int(supercell_a[2,1]), int(supercell_a[2,2])]]
        transB = [[int(supercell_b[0,0]), int(supercell_b[0,1]), int(supercell_b[0,2])],
                  [int(supercell_b[1,0]), int(supercell_b[1,1]), int(supercell_b[1,2])],
                  [int(supercell_b[2,0]), int(supercell_b[2,1]), int(supercell_b[2,2])]]
        
        slabA_super = SupercellTransformation(transA).apply_transformation(slabA_temp)
        slabB_super = SupercellTransformation(transB).apply_transformation(slabB_temp)
        
        total_atoms = len(slabA_super) + len(slabB_super)
        return total_atoms
    except:
        # If estimation fails, return a large number
        return 10000

def build_interface_from_builder(
    structA,
    structB,
    miller_a,
    miller_b,
    slab_thickness_a,
    slab_thickness_b,
    vacuum,
    sep,
    zsl_param_candidates,
    target_match=None,
    max_atoms: int = 400,
):
    """
    Attempt to construct an ordered interface using CoherentInterfaceBuilder. The search
    iterates over candidate ZSL parameter sets and slab thickness scaling factors until
    an interface within the atom limit is located. If none satisfy the limit, the
    smallest interface found is returned.
    """
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator

    if not zsl_param_candidates:
        zsl_param_candidates = [
            {
                "max_area": 800.0,
                "max_length_tol": 0.08,
                "max_angle_tol": 0.03,
                "bidirectional": True,
            }
        ]

    thickness_factors = [1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.3]
    min_thickness = 3.0

    target_key = None
    if target_match:
        target_key = _matrix_key(
            np.array(target_match["film_transformation"]),
            np.array(target_match["substrate_transformation"]),
        )

    best_candidate = None
    best_over_limit = None

    for factor in thickness_factors:
        current_thickness_a = max(slab_thickness_a * factor, min_thickness)
        current_thickness_b = max(slab_thickness_b * factor, min_thickness)

        for params in zsl_param_candidates:
            try:
                zslgen = ZSLGenerator(
                    max_area=params.get("max_area", 800.0),
                    max_length_tol=params.get("max_length_tol", 0.08),
                    max_angle_tol=params.get("max_angle_tol", 0.03),
                    bidirectional=params.get("bidirectional", True),
                )
            except TypeError:
                zslgen = ZSLGenerator(max_area=params.get("max_area", 800.0))

            builder = CoherentInterfaceBuilder(
                substrate_structure=structB,
                film_structure=structA,
                film_miller=miller_a,
                substrate_miller=miller_b,
                zslgen=zslgen,
            )

            if not builder.zsl_matches:
                continue

            terminations = builder.terminations
            if not terminations:
                continue

            matches = builder.zsl_matches
            for termination in terminations:
                try:
                    interfaces = list(
                        builder.get_interfaces(
                            termination=termination,
                            gap=sep,
                            vacuum_over_film=vacuum,
                            film_thickness=current_thickness_a,
                            substrate_thickness=current_thickness_b,
                            in_layers=False,
                        )
                    )
                except Exception:
                    continue

                for idx, interface in enumerate(interfaces):
                    if idx >= len(matches):
                        break

                    match_obj = matches[idx]
                    match_key = _matrix_key(
                        np.array(match_obj.film_transformation),
                        np.array(match_obj.substrate_transformation),
                    )
                    if target_key and match_key != target_key:
                        continue

                    n_atoms = len(interface)
                    candidate_info = {
                        "interface": interface,
                        "termination": termination,
                        "params": params,
                        "thickness_a": current_thickness_a,
                        "thickness_b": current_thickness_b,
                        "atoms": n_atoms,
                        "match_key": match_key,
                    }

                    if n_atoms <= max_atoms:
                        if best_candidate is None or n_atoms < best_candidate["atoms"]:
                            best_candidate = candidate_info
                    else:
                        if best_over_limit is None or n_atoms < best_over_limit["atoms"]:
                            best_over_limit = candidate_info

        if best_candidate is not None:
            break

    chosen = best_candidate or best_over_limit
    if chosen is None:
        return None, None, None

    interface: "Interface" = chosen["interface"]
    print(f"Using termination: {chosen['termination']}")
    print(f"Interface contains {chosen['atoms']} atoms (target: <= {max_atoms})")
    if chosen["atoms"] > max_atoms:
        print(f"Warning: Atom count ({chosen['atoms']}) exceeds limit ({max_atoms})")
    print(
        f"ZSL parameters: {chosen['params']} | film thickness={chosen['thickness_a']:.2f} Å, "
        f"substrate thickness={chosen['thickness_b']:.2f} Å"
    )

    combined = interface.copy()
    
    # Extract bottom (substrate) and top (film) structures
    bottom = Structure.from_sites(interface.substrate_sites)
    top = Structure.from_sites(interface.film_sites)
    
    # Fix layer crossing when vacuum is small
    if vacuum < 2.0 and len(bottom) > 0 and len(top) > 0:
        # Calculate interface boundary based on z-projections
        lattice = combined.lattice
        c_vec = lattice.matrix[2]
        c_length = np.linalg.norm(c_vec)
        c_unit = c_vec / c_length if c_length > 1e-8 else np.array([0, 0, 1])
        
        # Get z-projections for substrate and film
        substrate_z = np.dot(bottom.cart_coords, c_unit)
        film_z = np.dot(top.cart_coords, c_unit)
        
        # Interface boundary is between max of substrate and min of film
        interface_z_cart = (np.max(substrate_z) + np.min(film_z)) / 2.0
        interface_z_frac = interface_z_cart / c_length if c_length > 1e-8 else 0.5
        
        # Fix substrate (bottom) atoms that crossed to top region
        corrections_made = 0
        for i, site in enumerate(bottom):
            fc = site.frac_coords.copy()
            if fc[2] > interface_z_frac + 0.1:
                fc[2] = fc[2] - 1.0
                bottom.replace(i, site.species_string, fc, coords_are_cartesian=False)
                corrections_made += 1
        
        # Fix film (top) atoms that crossed to bottom region
        for i, site in enumerate(top):
            fc = site.frac_coords.copy()
            if fc[2] < interface_z_frac - 0.1:
                fc[2] = fc[2] + 1.0
                top.replace(i, site.species_string, fc, coords_are_cartesian=False)
                corrections_made += 1
        
        if corrections_made > 0:
            print(f"Layer crossing prevention (builder): Corrected {corrections_made} atom(s)")
            # Rebuild combined structure with corrected positions
            combined = Structure(lattice, [], [])
            for site in bottom:
                combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
            for site in top:
                combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
    
    combined.sort(key=lambda site: site.frac_coords[2])

    return bottom, top, combined

def auto_run(a_file, b_file, miller_a, miller_b, slab_thickness_a, slab_thickness_b, vacuum, sep, tol, max_area, strain_target, use_builder_interface=False, max_atoms=400):
    """Main orchestrator."""
    print("Loading structures...")
    A = load_structure(a_file)
    B = load_structure(b_file)
    print("Optimizing / getting primitive cells...")
    Aprim = get_primitive(A)
    Bprim = get_primitive(B)
    print("Primitive cell A lattice:", Aprim.lattice.abc)
    print("Primitive cell B lattice:", Bprim.lattice.abc)

    a_label = _safe_structure_label(a_file)
    b_label = _safe_structure_label(b_file)
    hetero_base_name = f"{a_label}@{b_label}"
    hetero_output_dir = Path("structures") / "heterojunctions"

    # Search for matches using bulk structures and Miller indices
    print("Searching for low-strain matches (this may take a moment)...")
    matches = search_matches(
        Aprim,
        Bprim,
        miller_a,
        miller_b,
        tol=tol,
        max_area=max_area,
        allow_bidirectional=True,
    )
    
    if len(matches) == 0:
        raise RuntimeError("No matches found within tolerance. Increase tol or max_area.")
    
    # Collect unique ZSL generator parameter sets observed in the matches.
    zsl_param_list: List[Dict[str, float | bool]] = []
    param_keys_seen: Set[Tuple[Tuple[str, float | bool], ...]] = set()
    for match in matches:
        params = match.get("zsl_params")
        if not params:
            continue
        key = tuple(sorted(params.items()))
        if key in param_keys_seen:
            continue
        param_keys_seen.add(key)
        zsl_param_list.append(params)
    
    # Build slabs after finding matches (needed for final structure construction)
    print("Building slabs (unstrained) for final structure...")
    slabA = build_slab(Aprim, miller_a, slab_thickness_a, vacuum)
    slabB = build_slab(Bprim, miller_b, slab_thickness_b, vacuum)
    
    # First, estimate atom counts for all matches with reduced thickness if needed
    # Calculate optimal thickness based on best match atom density
    def estimate_optimal_thickness(match, target_atoms):
        """Estimate optimal thickness to achieve target atom count."""
        est_atoms_full = estimate_atoms_for_match(
            Aprim, Bprim, miller_a, miller_b,
            match["supercell_a"], match["supercell_b"],
            slab_thickness_a, slab_thickness_b
        )
        
        if est_atoms_full <= target_atoms:
            return slab_thickness_a, slab_thickness_b
        
        # Calculate reduction factor
        reduction = (target_atoms / est_atoms_full) * 0.75  # 75% safety margin
        optimal_a = max(slab_thickness_a * reduction, 2.5)
        optimal_b = max(slab_thickness_b * reduction, 2.5)
        
        return optimal_a, optimal_b
    
    # Score matches prioritizing atom count, then strain/area metrics
    def match_score(m):
        opt_thick_a, opt_thick_b = estimate_optimal_thickness(m, max_atoms)
        est_atoms = estimate_atoms_for_match(
            Aprim,
            Bprim,
            miller_a,
            miller_b,
            m["supercell_a"],
            m["supercell_b"],
            opt_thick_a,
            opt_thick_b,
        )

        length_metric = float(m.get("length_mismatch", 0.0))
        angle_metric = float(m.get("angle_mismatch", 0.0))
        area_metric = float(m.get("area", 0.0))
        supercell_factor = float(max(m.get("det_a", 0.0), m.get("det_b", 0.0)))

        if est_atoms > max_atoms:
            atom_penalty = (est_atoms - max_atoms) * 4.0 + (est_atoms / max_atoms - 1.0) * 20.0
        else:
            atom_penalty = -((max_atoms - est_atoms) / max_atoms) * 5.0

        strain_penalty = length_metric * 200.0 + angle_metric * 2.0
        area_penalty = (area_metric / max(max_area, 1.0)) * 5.0
        supercell_penalty = supercell_factor

        total_penalty = atom_penalty + strain_penalty + area_penalty + supercell_penalty

        m["_cached_opt_thickness"] = (opt_thick_a, opt_thick_b)
        m["_cached_est_atoms"] = est_atoms
        m["_cached_strain_penalty"] = strain_penalty

        return total_penalty, est_atoms, strain_penalty

    matches_with_scores = [(idx, m, match_score(m)) for idx, m in enumerate(matches)]

    matches_with_scores.sort(
        key=lambda item: (
            0 if item[2][1] <= max_atoms else 1,
            item[2][1],
            item[1].get("priority", (item[0],)),
            item[2][0],
        )
    )

    valid_matches = [item for item in matches_with_scores if item[2][1] <= max_atoms]

    if len(valid_matches) > 0:
        ordered_items = valid_matches
        print(f"Found {len(valid_matches)} matches within atom limit ({max_atoms})")
    else:
        ordered_items = matches_with_scores
        print(f"Warning: No matches within atom limit ({max_atoms}). Selecting closest candidate...")

    best_item = ordered_items[0]
    best = best_item[1]
    (best_score, best_atoms, best_strain_metric) = best_item[2]
    
    # Calculate optimal thickness for selected match
    optimal_thick_a, optimal_thick_b = best.get("_cached_opt_thickness", estimate_optimal_thickness(best, max_atoms))
    
    print(f"Selected match with estimated {best_atoms} atoms (strain metric: {best_strain_metric:.4f})")
    print(f"Optimal thickness for this match: A={optimal_thick_a:.2f} Å, B={optimal_thick_b:.2f} Å")
    
    # Use optimal thickness if significantly different
    if abs(optimal_thick_a - slab_thickness_a) > 0.5 or abs(optimal_thick_b - slab_thickness_b) > 0.5:
        print(f"Adjusting thickness to meet atom limit:")
        print(f"  A: {slab_thickness_a:.2f} -> {optimal_thick_a:.2f} Å")
        print(f"  B: {slab_thickness_b:.2f} -> {optimal_thick_b:.2f} Å")
        slab_thickness_a = optimal_thick_a
        slab_thickness_b = optimal_thick_b

    if use_builder_interface:
        print("Attempting ordered interface construction via CoherentInterfaceBuilder...")
        builder_param_candidates = zsl_param_list or [
            {
                "max_area": float(max_area),
                "max_length_tol": float(max(tol, 0.03)),
                "max_angle_tol": float(min(0.08, max(tol / 2.0, 0.02))),
                "bidirectional": True,
            }
        ]

        bottom, top, combined = build_interface_from_builder(
            Aprim,
            Bprim,
            miller_a,
            miller_b,
            slab_thickness_a,
            slab_thickness_b,
            vacuum,
            sep,
            builder_param_candidates,
            target_match=best,
            max_atoms=max_atoms,
        )

        if combined is not None:
            print("Successfully built ordered interface using CoherentInterfaceBuilder.")
            write_poscars(bottom, top, combined, hetero_output_dir, hetero_base_name)
            return
        else:
            print("Ordered interface generation failed; falling back to manual stacking workflow.")
    
    print("Best match summary:")
    print(" area:", best["area"])
    print(" supercell A matrix:\n", best["supercell_a"])
    print(" supercell B matrix:\n", best["supercell_b"])
    if "det_a" in best and "det_b" in best:
        print(f" determinants det(A)={best['det_a']:.2f}, det(B)={best['det_b']:.2f}")
    if "length_components" in best:
        print(
            " length mismatch (fractional per in-plane vector):",
            np.array2string(np.array(best["length_components"]), precision=5),
        )
    if "angle_mismatch" in best:
        angles = best.get("angles", (None, None))
        if angles[0] is not None and angles[1] is not None:
            print(
                f" angles (film/substrate): {angles[0]:.3f}° / {angles[1]:.3f}° | mismatch {best['angle_mismatch']:.3f}°"
            )
        else:
            print(f" angle mismatch: {best['angle_mismatch']:.3f}°")
    if "zsl_params" in best:
        print(" ZSL parameter set:", best["zsl_params"])
    # Use the InterfaceMatch data to build the supercell structures manually.

    # Apply supercell transform matrices to slabs (they are 3x3 integer matrices)
    supA = np.array(best["supercell_a"])
    supB = np.array(best["supercell_b"])
    # Convert int matrices into 3x3 lists
    transA = [[int(supA[0,0]), int(supA[0,1]), int(supA[0,2])],
              [int(supA[1,0]), int(supA[1,1]), int(supA[1,2])],
              [int(supA[2,0]), int(supA[2,1]), int(supA[2,2])]]
    transB = [[int(supB[0,0]), int(supB[0,1]), int(supB[0,2])],
              [int(supB[1,0]), int(supB[1,1]), int(supB[1,2])],
              [int(supB[2,0]), int(supB[2,1]), int(supB[2,2])]]
    slabA_super = SupercellTransformation(transA).apply_transformation(slabA)
    slabB_super = SupercellTransformation(transB).apply_transformation(slabB)

    # Compute in-plane lattice norms to determine actual strains
    aA_pre = compute_inplane_norms(slabA_super.lattice)[0:2]
    aB_pre = compute_inplane_norms(slabB_super.lattice)[0:2]
    print("In-plane sizes after supercell (A):", aA_pre)
    print("In-plane sizes after supercell (B):", aB_pre)

    # Decide which to strain based on user input: 'A', 'B', or 'both'
    if strain_target.lower() == 'a':
        scale_x = aB_pre[0] / aA_pre[0]
        scale_y = aB_pre[1] / aA_pre[1]
        print(f"Straining A to match B with factors x={scale_x:.6f}, y={scale_y:.6f}")
        slabA_strained = apply_inplane_strain(slabA_super, scale_x, scale_y)
        slabB_strained = slabB_super.copy()
    elif strain_target.lower() == 'b':
        scale_x = aA_pre[0] / aB_pre[0]
        scale_y = aA_pre[1] / aB_pre[1]
        print(f"Straining B to match A with factors x={scale_x:.6f}, y={scale_y:.6f}")
        slabB_strained = apply_inplane_strain(slabB_super, scale_x, scale_y)
        slabA_strained = slabA_super.copy()
    elif strain_target.lower() == 'both':
        # apply sqrt of ratio to both (split strain evenly)
        scale_x = math.sqrt(aB_pre[0] / aA_pre[0])
        scale_y = math.sqrt(aB_pre[1] / aA_pre[1])
        print(f"Splitting strain: scale factors applied to A: {scale_x:.6f},{scale_y:.6f} and to B: {1.0/scale_x:.6f},{1.0/scale_y:.6f}")
        slabA_strained = apply_inplane_strain(slabA_super, scale_x, scale_y)
        slabB_strained = apply_inplane_strain(slabB_super, 1.0/scale_x, 1.0/scale_y)
    else:
        raise ValueError("strain_target must be one of 'A', 'B', 'both'")

    # Ensure in-plane lattice vectors match exactly before stacking
    # This is critical for ordered, high-symmetry interfaces
    bottom_lat = slabB_strained.lattice.matrix
    top_lat = slabA_strained.lattice.matrix
    
    # Verify vectors match (should be true after straining, but double-check)
    if np.linalg.norm(bottom_lat[0] - top_lat[0]) > 0.01 or np.linalg.norm(bottom_lat[1] - top_lat[1]) > 0.01:
        print("Warning: In-plane vectors don't match. Adjusting top slab lattice.")
        # Preserve top's c vector while matching in-plane vectors
        new_top_lat = top_lat.copy()
        new_top_lat[0] = bottom_lat[0]
        new_top_lat[1] = bottom_lat[1]
        # Keep original c vector to preserve slab structure
        slabA_strained = Structure(Lattice(new_top_lat), slabA_strained.species, slabA_strained.frac_coords)
    
    # Iteratively optimize thickness to meet atom limit
    # This is critical for DFT efficiency
    current_thickness_a = slab_thickness_a
    current_thickness_b = slab_thickness_b
    min_thickness = 2.5  # Minimum 2.5 Angstroms (very thin, but physically reasonable)
    max_iterations = 10
    
    for iteration in range(max_iterations):
        n_atoms_current = len(slabA_super) + len(slabB_super)
        
        if n_atoms_current <= max_atoms:
            break
        
        print(f"\nIteration {iteration + 1}: Current atom count ({n_atoms_current}) exceeds limit ({max_atoms})")
        
        # Calculate aggressive reduction factor
        # Use 0.6-0.7 factor to ensure we get well under limit
        reduction_factor = (max_atoms / n_atoms_current) * 0.65  # 65% to be safe
        
        # Reduce thickness proportionally
        new_thickness_a = max(current_thickness_a * reduction_factor, min_thickness)
        new_thickness_b = max(current_thickness_b * reduction_factor, min_thickness)
        
        # Don't reduce if already at minimum
        if new_thickness_a >= current_thickness_a - 0.1 and new_thickness_b >= current_thickness_b - 0.1:
            print("Warning: Cannot reduce thickness further. Atom count may exceed limit.")
            break
        
        print(f"Reducing thickness: A: {current_thickness_a:.2f} -> {new_thickness_a:.2f} Å")
        print(f"                   B: {current_thickness_b:.2f} -> {new_thickness_b:.2f} Å")
        
        current_thickness_a = new_thickness_a
        current_thickness_b = new_thickness_b
        
        # Rebuild slabs with reduced thickness
        slabA = build_slab(Aprim, miller_a, current_thickness_a, vacuum)
        slabB = build_slab(Bprim, miller_b, current_thickness_b, vacuum)
        slabA_super = SupercellTransformation(transA).apply_transformation(slabA)
        slabB_super = SupercellTransformation(transB).apply_transformation(slabB)
        
        # Reapply strain
        aA_pre = compute_inplane_norms(slabA_super.lattice)[0:2]
        aB_pre = compute_inplane_norms(slabB_super.lattice)[0:2]
        
        if strain_target.lower() == 'a':
            scale_x = aB_pre[0] / aA_pre[0]
            scale_y = aB_pre[1] / aA_pre[1]
            slabA_strained = apply_inplane_strain(slabA_super, scale_x, scale_y)
            slabB_strained = slabB_super.copy()
        elif strain_target.lower() == 'b':
            scale_x = aA_pre[0] / aB_pre[0]
            scale_y = aA_pre[1] / aB_pre[1]
            slabB_strained = apply_inplane_strain(slabB_super, scale_x, scale_y)
            slabA_strained = slabA_super.copy()
        elif strain_target.lower() == 'both':
            scale_x = math.sqrt(aB_pre[0] / aA_pre[0])
            scale_y = math.sqrt(aB_pre[1] / aA_pre[1])
            slabA_strained = apply_inplane_strain(slabA_super, scale_x, scale_y)
            slabB_strained = apply_inplane_strain(slabB_super, 1.0/scale_x, 1.0/scale_y)
        
        # Update for next iteration
        slabA_super = slabA_strained if strain_target.lower() != 'b' else slabA_super
        slabB_super = slabB_strained if strain_target.lower() != 'a' else slabB_super
    
    final_atoms_before_stack = len(slabA_strained) + len(slabB_strained)
    print(f"\nFinal atom count before stacking: {final_atoms_before_stack} (target: <={max_atoms})")
    
    # align and stack with ordered arrangement (put B as bottom, A as top)
    bottom, top, combined = align_and_stack_ordered(slabB_strained, slabA_strained, separation=sep, vacuum=vacuum)
    
    # Report final atom count
    final_atoms = len(combined)
    print(f"\nFinal structure contains {final_atoms} atoms (limit: {max_atoms})")
    if final_atoms > max_atoms:
        print(f"⚠ WARNING: Atom count ({final_atoms}) exceeds limit ({max_atoms})!")
        print(f"  Consider reducing max_area or using thinner initial slabs.")
    else:
        print(f"✓ Atom count is within limit ({(max_atoms - final_atoms) / max_atoms * 100:.1f}% under limit)")
    
    # Verify final structure has matching in-plane lattice
    final_bottom_lat = bottom.lattice.matrix
    final_top_lat = top.lattice.matrix
    final_combined_lat = combined.lattice.matrix
    
    print("\nFinal lattice verification:")
    print("Bottom in-plane vectors:", final_bottom_lat[0], final_bottom_lat[1])
    print("Top in-plane vectors:", final_top_lat[0], final_top_lat[1])
    print("Combined in-plane vectors:", final_combined_lat[0], final_combined_lat[1])
    
    write_poscars(bottom, top, combined, hetero_output_dir, hetero_base_name)

    # print final in-plane match results
    final_a = np.linalg.norm(bottom.lattice.matrix[0]), np.linalg.norm(bottom.lattice.matrix[1])
    final_b = np.linalg.norm(top.lattice.matrix[0]), np.linalg.norm(top.lattice.matrix[1])
    print("Final in-plane lattice sizes (bottom):", final_a)
    print("Final in-plane lattice sizes (top):", final_b)
    # estimate percent strains applied to original prims (informational)
    # NOTE: more careful calculation can be added

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Auto build matched heterojunction interface from two bulks.")
    # Build-mode inputs (kept for backwards compatibility)
    p.add_argument("--a", dest="afile", required=False, help="File for material A (CIF/POSCAR)")
    p.add_argument("--b", dest="bfile", required=False, help="File for material B (CIF/POSCAR)")

    # Split-mode inputs
    p.add_argument(
        "--split_layers",
        dest="split_layers",
        action="store_true",
        help="Split an existing heterojunction structure into independent layers (no coordinate shifts).",
    )
    p.add_argument("--input", dest="input_file", required=False, help="Heterojunction structure file to split (CIF/POSCAR).")
    p.add_argument("--out_base_dir", dest="out_base_dir", required=False, default=None, help="Base directory to place <input_stem>/ folder (default: input's parent).")
    p.add_argument("--layer1_elements", dest="layer1_elements", required=False, default=None, help="Comma-separated element symbols for layer 1 (e.g., 'Ti,O').")
    p.add_argument("--layer2_elements", dest="layer2_elements", required=False, default=None, help="Comma-separated element symbols for layer 2 (optional).")

    p.add_argument("--miller_a", dest="miller_a", default="0,0,1", help="Miller for A, e.g. 0,0,1")
    p.add_argument("--miller_b", dest="miller_b", default="1,0,1", help="Miller for B")
    p.add_argument("--slab_thickness_a", dest="stka", type=float, default=20.0, help="Slab thickness for A (Å)")
    p.add_argument("--slab_thickness_b", dest="stkb", type=float, default=12.0, help="Slab thickness for B (Å) or trilayer estimate")
    p.add_argument("--vacuum", dest="vac", type=float, default=20.0, help="Vacuum size (Å)")
    p.add_argument("--sep", dest="sep", type=float, default=3.2, help="Initial separation between surfaces (Å)")
    p.add_argument("--tol", dest="tol", type=float, default=0.03, help="Matching strain tolerance (fraction, e.g. 0.03)")
    p.add_argument("--max_area", dest="max_area", type=float, default=800.0, help="Maximum interface area (Å^2) for searching matches")
    p.add_argument("--strain_target", dest="strain_target", default="A", choices=["A","B","both"], help="Which material to strain: A, B, or both (split)")
    p.add_argument("--use_builder_interface", dest="use_builder", action="store_true", help="Use CoherentInterfaceBuilder.get_interfaces for ordered interface (recommended)")
    p.add_argument("--max_atoms", dest="max_atoms", type=int, default=400, help="Maximum number of atoms in final structure (default: 400)")
    # Adsorbate-mode inputs (H5PbCI3N2 slab + C60)
    p.add_argument(
        "--adsorbate_mode",
        dest="adsorbate_mode",
        action="store_true",
        help="Build a H5PbCI3N2 slab and place C60 above the surface (DFT-ready POSCAR).",
    )
    p.add_argument(
        "--base_cif",
        dest="base_cif",
        default="structures/perovskites/H5PbCI3N2.cif",
        help="Perovskite bulk CIF used to generate the slab (default: structures/perovskites/H5PbCI3N2.cif).",
    )
    p.add_argument(
        "--adsorbate_cif",
        dest="adsorbate_cif",
        default="structures/etl/C60-Ih.cif",
        help="Adsorbate CIF (default: structures/etl/C60-Ih.cif).",
    )
    p.add_argument(
        "--adsorbates",
        dest="adsorbates",
        default=None,
        help="Comma-separated adsorbate CIFs for multilayer stacking (bottom->top), e.g. 'structures/etl/C70-D5h.cif,structures/etl/C60-Ih.cif'.",
    )
    p.add_argument(
        "--layer_gaps",
        dest="layer_gaps",
        default=None,
        help="Comma-separated interlayer gaps in Å. Must match --adsorbates length. Example for 3 layers (slab/C70/C60): '2,2'.",
    )
    p.add_argument(
        "--termination",
        dest="termination",
        default="PbI",
        choices=["PbI", "FAI", "MAI", "AI"],
        help="Perovskite termination for adsorbate mode: PbI / FAI / MAI / AI (AI = mixed A-site iodide).",
    )
    p.add_argument(
        "--miller",
        dest="miller",
        default="0,0,1",
        help="Miller index for the perovskite slab cut, e.g. 0,0,1 (default).",
    )
    p.add_argument(
        "--slab_thickness",
        dest="slab_thickness",
        type=float,
        default=20.0,
        help="Minimum slab thickness in Å (default: 20.0).",
    )
    p.add_argument(
        "--adsorbate_distance",
        dest="adsorbate_distance",
        type=float,
        default=3.5,
        help="Distance (Å) between slab top-most atom and the lowest C atom in C60 along surface normal.",
    )
    p.add_argument(
        "--adsorbate_xy",
        dest="adsorbate_xy",
        default=None,
        help="Optional lateral placement of C60 center as fractional 'fx,fy' in the slab cell (default: 0.5,0.5).",
    )
    p.add_argument(
        "--supercell_xy",
        dest="supercell_xy",
        default=None,
        help="Optional in-plane supercell for the perovskite slab as 'nx,ny' (e.g., 2,2).",
    )
    p.add_argument(
        "--auto_supercell",
        dest="auto_supercell",
        action="store_true",
        help="Automatically choose a small in-plane supercell to reduce C60 periodic-image interactions.",
    )
    p.add_argument(
        "--no_symmetric_slab",
        dest="no_symmetric_slab",
        action="store_true",
        help="Disable symmetric slab requirement (top and bottom termination may differ). Not recommended.",
    )
    p.add_argument(
        "--c60_diameter",
        dest="c60_diameter",
        type=float,
        default=7.1,
        help="Approximate C60 diameter in Å (used only when --auto_supercell is set).",
    )
    p.add_argument(
        "--c60_buffer",
        dest="c60_buffer",
        type=float,
        default=3.0,
        help="Extra spacing added to C60 diameter in Å for --auto_supercell (default: 3.0).",
    )
    args = p.parse_args()

    if args.split_layers:
        if not args.input_file:
            raise SystemExit("Error: --split_layers requires --input <heterojunction_file>.")
        split_layers_to_folder(
            args.input_file,
            out_base_dir=args.out_base_dir,
            layer1_elements=args.layer1_elements,
            layer2_elements=args.layer2_elements,
        )
        raise SystemExit(0)

    if args.adsorbate_mode:
        _require_build_deps()
        miller = tuple(map(int, args.miller.split(",")))
        xy = None
        if args.adsorbate_xy:
            parts = [p.strip() for p in str(args.adsorbate_xy).split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit("Error: --adsorbate_xy must be 'fx,fy' (e.g., 0.5,0.5).")
            xy = (float(parts[0]), float(parts[1]))

        sc_xy = None
        if args.supercell_xy:
            parts = [p.strip() for p in str(args.supercell_xy).split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit("Error: --supercell_xy must be 'nx,ny' (e.g., 2,2).")
            sc_xy = (int(parts[0]), int(parts[1]))

        # Multilayer: if --adsorbates is provided, build a stacked adsorbate model.
        if args.adsorbates:
            ads_list = [p.strip() for p in str(args.adsorbates).split(",") if p.strip()]
            if not ads_list:
                raise SystemExit("Error: --adsorbates is provided but empty.")
            if not args.layer_gaps:
                raise SystemExit("Error: --adsorbates requires --layer_gaps (e.g., '2,2').")
            gap_list = [g.strip() for g in str(args.layer_gaps).split(",") if g.strip()]
            if len(gap_list) != len(ads_list):
                raise SystemExit("Error: --layer_gaps length must match --adsorbates length.")
            gaps = [float(g) for g in gap_list]

            slab, combined = build_perovskite_adsorbate_stack(
                base_cif=args.base_cif,
                adsorbate_cifs=ads_list,
                layer_gaps=gaps,
                miller=miller,
                slab_thickness=float(args.slab_thickness),
                vacuum=float(args.vac),
                termination=str(args.termination),
                adsorbate_xy=xy,
                supercell_xy=sc_xy,
                auto_supercell=bool(args.auto_supercell),
                adsorbate_diameter=float(args.c60_diameter),
                adsorbate_buffer=float(args.c60_buffer),
                max_atoms=int(args.max_atoms),
                symmetric_slab=not bool(args.no_symmetric_slab),
            )
        else:
            slab, combined = build_perovskite_c60_adsorbate(
                base_cif=args.base_cif,
                c60_cif=args.adsorbate_cif,
                miller=miller,
                slab_thickness=float(args.slab_thickness),
                vacuum=float(args.vac),
                termination=str(args.termination),
                adsorbate_distance=float(args.adsorbate_distance),
                adsorbate_xy=xy,
                supercell_xy=sc_xy,
                auto_supercell=bool(args.auto_supercell),
                c60_diameter=float(args.c60_diameter),
                c60_buffer=float(args.c60_buffer),
                max_atoms=int(args.max_atoms),
                symmetric_slab=not bool(args.no_symmetric_slab),
            )

        base_label = _safe_structure_label(args.base_cif)
        if args.adsorbates:
            ads_label = "@".join([_safe_structure_label(p) for p in str(args.adsorbates).split(",") if p.strip()])
        else:
            ads_label = _safe_structure_label(args.adsorbate_cif)
        term_label = str(args.termination)
        out_dir = Path("structures") / "heterojunctions"
        out_dir.mkdir(parents=True, exist_ok=True)
        sc_tag = ""
        if args.auto_supercell:
            sc_tag = "_autoSC"
        elif sc_xy is not None:
            sc_tag = f"_{sc_xy[0]}x{sc_xy[1]}"
        if args.adsorbates and args.layer_gaps:
            gap_tag = "-".join([f"{float(g):.2f}" for g in str(args.layer_gaps).split(",") if g.strip()])
            base_name = f"{base_label}@{ads_label}_{term_label}{sc_tag}_g{gap_tag}"
        else:
            base_name = f"{base_label}@{ads_label}_{term_label}{sc_tag}_d{args.adsorbate_distance:.2f}"

        slab_path = out_dir / f"{base_name}_slab.vasp"
        combined_path = out_dir / f"{base_name}.vasp"

        Poscar(
            _reorder_structure_for_poscar(slab),
            comment=f"{base_label} slab ({term_label} termination); vacuum={args.vac:.2f} Å",
            sort_structure=False,
        ).write_file(slab_path)
        Poscar(
            _reorder_structure_for_poscar(combined),
            comment=(
                f"{base_label} slab + {ads_label}; termination={term_label}; "
                + (
                    f"gaps={gap_tag} Å"
                    if (args.adsorbates and args.layer_gaps)
                    else f"distance={args.adsorbate_distance:.2f} Å"
                )
            ),
            sort_structure=False,
        ).write_file(combined_path)
        print("Wrote adsorbate POSCARs:")
        print(" - Slab   :", slab_path)
        print(" - Combined:", combined_path)
        raise SystemExit(0)

    if not args.afile or not args.bfile:
        raise SystemExit("Error: build mode requires --a and --b (or use --split_layers).")

    miller_a = tuple(map(int, args.miller_a.split(",")))
    miller_b = tuple(map(int, args.miller_b.split(",")))

    auto_run(args.afile, args.bfile, miller_a, miller_b,
             args.stka, args.stkb, args.vac, args.sep,
             args.tol, args.max_area, args.strain_target, args.use_builder, args.max_atoms)
