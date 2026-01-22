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
from typing import Dict, Iterable, List, Set, Tuple

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

    if not args.afile or not args.bfile:
        raise SystemExit("Error: build mode requires --a and --b (or use --split_layers).")

    miller_a = tuple(map(int, args.miller_a.split(",")))
    miller_b = tuple(map(int, args.miller_b.split(",")))

    auto_run(args.afile, args.bfile, miller_a, miller_b,
             args.stka, args.stkb, args.vac, args.sep,
             args.tol, args.max_area, args.strain_target, args.use_builder, args.max_atoms)
