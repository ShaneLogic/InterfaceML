"""
Adsorbate placement and stacking utilities for InterfaceML.

This module provides a lightweight workflow for building interface models
by stacking an adsorbate (e.g., fullerene) on top of a perovskite slab.
It is designed to be deterministic and fast for high-throughput generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator

from interfaceml.utils import geometry
from interfaceml.core import layering


@dataclass(frozen=True)
class SupercellChoice:
    nx: int
    ny: int
    target_xy: Tuple[float, float]
    diameter: float
    termination: str = "auto"
    termination_top: Tuple[str, ...] = ()
    termination_bottom: Tuple[str, ...] = ()


def build_slab(
    structure: Structure,
    miller: Tuple[int, int, int],
    slab_thickness: float,
    vacuum: float,
    *,
    center_slab: bool = True,
    termination: Optional[str] = None,
    layer_tol: float = 1.5,
) -> Structure:
    """
    Build a surface slab from a bulk structure using pymatgen's SlabGenerator.
    """
    slabgen = SlabGenerator(
        structure,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=float(slab_thickness),
        min_vacuum_size=float(vacuum),
        center_slab=center_slab,
    )
    slabs = slabgen.get_slabs()
    if not slabs:
        raise RuntimeError(f"No slabs generated for miller={miller}")

    if termination is None or str(termination).lower() in {"auto", "any"}:
        slab, _ = _select_slab_by_termination(slabs, target=None, layer_tol=layer_tol)
        return slab

    slab, _ = _select_slab_by_termination(slabs, target=str(termination), layer_tol=layer_tol)
    return slab


def auto_supercell_xy(
    slab: Structure,
    adsorbate: Structure,
    *,
    buffer: float = 10.0,
    min_supercell: int = 1,
    max_supercell: int = 8,
) -> SupercellChoice:
    """
    Choose an in-plane supercell (nx, ny) to avoid adsorbate-adsorbate overlap.

    The target in-plane size is diameter + buffer. Each in-plane lattice
    vector is scaled by the minimum integer that meets the target.
    """
    a_len = float(np.linalg.norm(slab.lattice.matrix[0]))
    b_len = float(np.linalg.norm(slab.lattice.matrix[1]))

    diameter = geometry.estimate_molecule_diameter(np.asarray(adsorbate.cart_coords))
    target = max(diameter + buffer, max(a_len, b_len))

    nx = int(np.ceil(target / max(a_len, 1e-6)))
    ny = int(np.ceil(target / max(b_len, 1e-6)))

    nx = min(max(nx, min_supercell), max_supercell)
    ny = min(max(ny, min_supercell), max_supercell)

    return SupercellChoice(nx=nx, ny=ny, target_xy=(target, target), diameter=diameter)


def _center_adsorbate(coords: np.ndarray) -> np.ndarray:
    center = geometry.center_of_mass(coords)
    return coords - center


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _apply_rotation(coords: np.ndarray, rot: Optional[np.ndarray]) -> np.ndarray:
    if rot is None:
        return coords
    return np.dot(coords, rot.T)


def prepare_adsorbate_layer(
    adsorbate: Structure,
    lattice: Lattice,
    *,
    xy_frac: Tuple[float, float] = (0.5, 0.5),
    rotation: Optional[np.ndarray] = None,
) -> Structure:
    """
    Center the adsorbate and place it at a target in-plane fractional position.
    """
    coords = np.asarray(adsorbate.cart_coords, dtype=float)
    coords = _center_adsorbate(coords)
    coords = _apply_rotation(coords, rotation)

    a_vec = np.asarray(lattice.matrix[0], dtype=float)
    b_vec = np.asarray(lattice.matrix[1], dtype=float)
    xy_cart = a_vec * float(xy_frac[0]) + b_vec * float(xy_frac[1])

    coords = coords + xy_cart

    return Structure(
        lattice,
        adsorbate.species,
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
    )


def _layer_elements(
    structure: Structure,
    *,
    which: str,
    layer_tol: float = 1.5,
) -> Tuple[str, ...]:
    n = layering.interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n)
    if len(heights) == 0:
        return tuple()
    max_h = float(np.max(heights))
    min_h = float(np.min(heights))
    if which == "top":
        mask = heights >= (max_h - float(layer_tol))
    else:
        mask = heights <= (min_h + float(layer_tol))
    elems = {str(site.specie) for site, keep in zip(structure, mask) if keep}
    return tuple(sorted(elems))


def _classify_termination(elements: Iterable[str]) -> str:
    elems = set(elements)
    if not elems:
        return "unknown"
    if "Pb" in elems and "I" in elems and not ({"C", "N"} & elems):
        return "PbI"
    if {"C", "N"} & elems and "I" in elems:
        return "AI"
    if "Pb" in elems and "I" in elems:
        return "PbI"
    if {"C", "N"} & elems:
        return "AI"
    return "unknown"


def _select_slab_by_termination(
    slabs: List[Structure],
    *,
    target: Optional[str],
    layer_tol: float = 1.5,
) -> Tuple[Structure, SupercellChoice]:
    best = None
    best_score = None
    target_norm = target.strip() if target else None

    for slab in slabs:
        top_elems = _layer_elements(slab, which="top", layer_tol=layer_tol)
        bottom_elems = _layer_elements(slab, which="bottom", layer_tol=layer_tol)
        top_label = _classify_termination(top_elems)
        bottom_label = _classify_termination(bottom_elems)

        # Jaccard distance between top and bottom element sets (lower is more symmetric)
        top_set = set(top_elems)
        bottom_set = set(bottom_elems)
        union = top_set | bottom_set
        inter = top_set & bottom_set
        jaccard = 1.0 if not union else 1.0 - (len(inter) / len(union))

        label_penalty = 0.0
        if target_norm and top_label != target_norm:
            label_penalty = 1.0

        score = jaccard + label_penalty

        if best is None or score < best_score:
            best = (slab, top_label, top_elems, bottom_elems)
            best_score = score

        if target_norm and top_label == target_norm and score <= 0.01:
            best = (slab, top_label, top_elems, bottom_elems)
            best_score = score
            break

    if best is None:
        raise RuntimeError("Could not select a termination slab.")

    slab, top_label, top_elems, bottom_elems = best
    info = SupercellChoice(
        nx=1,
        ny=1,
        target_xy=(0.0, 0.0),
        diameter=0.0,
        termination=top_label,
        termination_top=tuple(top_elems),
        termination_bottom=tuple(bottom_elems),
    )
    return slab, info


def stack_structures(
    bottom: Structure,
    top: Structure,
    *,
    separation: float = 3.2,
    vacuum: float = 20.0,
) -> Tuple[Structure, Structure, Structure]:
    """
    Stack two slabs/structures along the interface normal.

    Returns (bottom_aligned, top_aligned, combined).
    """
    bottom_aligned = bottom.copy()
    top_aligned = top.copy()

    n = layering.interface_normal_unit(bottom_aligned)
    a_vec = np.asarray(bottom_aligned.lattice.matrix[0], dtype=float)
    b_vec = np.asarray(bottom_aligned.lattice.matrix[1], dtype=float)

    if len(bottom_aligned) > 0:
        bottom_proj = np.dot(np.asarray(bottom_aligned.cart_coords, dtype=float), n)
        min_bottom = float(bottom_proj.min())
        bottom_aligned.translate_sites(
            list(range(len(bottom_aligned))),
            -n * min_bottom,
        )
        bottom_proj = np.dot(np.asarray(bottom_aligned.cart_coords, dtype=float), n)
        max_bottom = float(bottom_proj.max())
    else:
        max_bottom = 0.0

    if len(top_aligned) > 0:
        top_proj = np.dot(np.asarray(top_aligned.cart_coords, dtype=float), n)
        min_top = float(top_proj.min())
        top_aligned.translate_sites(
            list(range(len(top_aligned))),
            -n * min_top,
        )
        top_aligned.translate_sites(
            list(range(len(top_aligned))),
            n * (max_bottom + float(separation)),
        )
        top_proj = np.dot(np.asarray(top_aligned.cart_coords, dtype=float), n)
        max_top = float(top_proj.max())
    else:
        max_top = max_bottom + float(separation)

    total_z = max_top + float(vacuum)
    c_vec = n * total_z

    combined_lat = Lattice(np.vstack([a_vec, b_vec, c_vec]))

    interface_z_cart = max_bottom + float(separation) / 2.0
    interface_z_frac = interface_z_cart / total_z if total_z > 1e-8 else 0.5

    def _wrap_layer(coords: np.ndarray, keep_above: Optional[bool]) -> List[np.ndarray]:
        fracs: List[np.ndarray] = []
        for coord in coords:
            fc = combined_lat.get_fractional_coords(coord)
            fc_x = fc[0] % 1.0
            fc_y = fc[1] % 1.0
            fc_z = fc[2] % 1.0
            if keep_above is True and fc_z < interface_z_frac - 0.05:
                fc_z = fc_z + 1.0
            if keep_above is False and fc_z > interface_z_frac + 0.05:
                fc_z = fc_z - 1.0
            fracs.append(np.array([fc_x, fc_y, fc_z], dtype=float))
        return fracs

    bottom_fracs = _wrap_layer(np.asarray(bottom_aligned.cart_coords, dtype=float), keep_above=False)
    top_fracs = _wrap_layer(np.asarray(top_aligned.cart_coords, dtype=float), keep_above=True)

    bottom_out = Structure(combined_lat, bottom_aligned.species, bottom_fracs, coords_are_cartesian=False)
    top_out = Structure(combined_lat, top_aligned.species, top_fracs, coords_are_cartesian=False)

    combined = Structure(combined_lat, [], [])
    for site in bottom_out:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
    for site in top_out:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)

    return bottom_out, top_out, combined


def build_adsorbate_interface(
    base_structure: Structure,
    adsorbate_structure: Structure,
    *,
    miller: Tuple[int, int, int] = (0, 0, 1),
    slab_thickness: float = 18.0,
    vacuum: float = 20.0,
    separation: float = 3.2,
    supercell_xy: Optional[Tuple[int, int]] = None,
    buffer: float = 10.0,
    xy_frac: Tuple[float, float] = (0.5, 0.5),
    termination: Optional[str] = None,
    layer_tol: float = 1.5,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[Structure, Structure, Structure, SupercellChoice]:
    """
    Build a perovskite/adsorbate interface from bulk inputs.

    Returns bottom slab, top adsorbate layer, combined interface, and supercell choice.
    """
    slab, term_info = _select_slab_by_termination(
        SlabGenerator(
            base_structure,
            miller_index=tuple(int(x) for x in miller),
            min_slab_size=float(slab_thickness),
            min_vacuum_size=0.0,
            center_slab=True,
        ).get_slabs(),
        target=termination,
        layer_tol=layer_tol,
    )

    if supercell_xy is None:
        choice = auto_supercell_xy(slab, adsorbate_structure, buffer=buffer)
        nx, ny = choice.nx, choice.ny
    else:
        nx, ny = int(supercell_xy[0]), int(supercell_xy[1])
        choice = SupercellChoice(nx=nx, ny=ny, target_xy=(0.0, 0.0), diameter=0.0)

    slab_super = slab.copy()
    slab_super.make_supercell([nx, ny, 1])

    top_lattice = Lattice(np.array(slab_super.lattice.matrix))
    adsorbate_layer = prepare_adsorbate_layer(
        adsorbate_structure,
        top_lattice,
        xy_frac=xy_frac,
        rotation=rotation,
    )

    bottom_out, top_out, combined = stack_structures(
        slab_super,
        adsorbate_layer,
        separation=separation,
        vacuum=vacuum,
    )

    choice = SupercellChoice(
        nx=choice.nx,
        ny=choice.ny,
        target_xy=choice.target_xy,
        diameter=choice.diameter,
        termination=term_info.termination,
        termination_top=term_info.termination_top,
        termination_bottom=term_info.termination_bottom,
    )

    return bottom_out, top_out, combined, choice
