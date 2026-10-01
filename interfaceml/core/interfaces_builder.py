"""Coherent slab-slab interface construction via pymatgen ZSL + CIB.

Phase 1 scope: lattice-matched supercell search (Zur-McGill), build top-K
candidates ranked by von Mises strain, return interfaces with metrics.

Termination handling, lateral registry, and N-slab stacking come in later
phases.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.core import Structure


def apply_registry_twist(
    interface: Structure,
    *,
    xy_shift: tuple[float, float] = (0.0, 0.0),
    xy_units: str = "angstrom",
    twist_deg: float = 0.0,
) -> Structure:
    """Translate + rotate the film slab over the substrate.

    Requires `interface.film_indices`. Twist rotates the film about its own
    in-plane center-of-mass around the c-axis. Shift moves the film laterally:
    `xy_units='angstrom'` → (dx, dy) in Å; `xy_units='fractional'` → fractions
    of the in-plane lattice vectors a, b.
    """
    if not hasattr(interface, "film_indices"):
        return interface
    film_idx = list(interface.film_indices)
    if not film_idx:
        return interface

    coords = np.array(interface.cart_coords, dtype=float)

    if abs(twist_deg) > 1e-9:
        com = coords[film_idx].mean(axis=0)
        rad = math.radians(float(twist_deg))
        ca, sa = math.cos(rad), math.sin(rad)
        R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        coords[film_idx] = (coords[film_idx] - com) @ R.T + com

    dx, dy = float(xy_shift[0]), float(xy_shift[1])
    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
        if xy_units == "fractional":
            a_vec = np.asarray(interface.lattice.matrix[0], dtype=float)
            b_vec = np.asarray(interface.lattice.matrix[1], dtype=float)
            delta = dx * a_vec + dy * b_vec
        else:
            delta = np.array([dx, dy, 0.0], dtype=float)
        coords[film_idx] += delta

    new = interface.copy()
    for index in film_idx:
        new.translate_sites(
            [index],
            coords[index] - interface.cart_coords[index],
            frac_coords=False,
            to_unit_cell=False,
        )
    return new


def list_interface_terminations(
    substrate: Structure,
    film: Structure,
    *,
    substrate_miller: tuple[int, int, int] = (0, 0, 1),
    film_miller: tuple[int, int, int] = (0, 0, 1),
    max_area: float = 400.0,
    max_length_tol: float = 0.03,
    max_angle_tol: float = 0.01,
    bidirectional: bool = True,
) -> dict[str, Any]:
    """Enumerate substrate/film termination labels available for this pair.

    Returns dict with `substrate_terminations` and `film_terminations`, each a
    list of {label, formula, spacegroup, instance} entries derived from
    pymatgen's CIB `<formula>_<spacegroup>_<id>` convention. Also exposes
    `n_zsl_matches` and `n_pairs` for UI sanity checks.
    """
    zsl = ZSLGenerator(
        max_area=float(max_area),
        max_length_tol=float(max_length_tol),
        max_angle_tol=float(max_angle_tol),
        bidirectional=bool(bidirectional),
    )
    cib = CoherentInterfaceBuilder(
        substrate_structure=substrate,
        film_structure=film,
        film_miller=tuple(int(x) for x in film_miller),
        substrate_miller=tuple(int(x) for x in substrate_miller),
        zslgen=zsl,
    )

    def _split(label: str) -> dict[str, str]:
        parts = label.rsplit("_", 2)
        if len(parts) == 3:
            return {
                "label": label,
                "formula": parts[0],
                "spacegroup": parts[1],
                "instance": parts[2],
            }
        return {"label": label, "formula": label, "spacegroup": "", "instance": ""}

    # pymatgen stores termination pairs in (film, substrate) order.
    sub_set = sorted({t[1] for t in cib.terminations})
    film_set = sorted({t[0] for t in cib.terminations})

    return {
        "substrate_terminations": [_split(t) for t in sub_set],
        "film_terminations": [_split(t) for t in film_set],
        "n_pairs": len(cib.terminations),
        "n_zsl_matches": len(cib.zsl_matches) if cib.zsl_matches else 0,
    }


def build_coherent_interfaces(
    substrate: Structure,
    film: Structure,
    *,
    substrate_miller: tuple[int, int, int] = (0, 0, 1),
    film_miller: tuple[int, int, int] = (0, 0, 1),
    max_area: float = 400.0,
    max_length_tol: float = 0.03,
    max_angle_tol: float = 0.01,
    bidirectional: bool = True,
    gap: float = 2.5,
    vacuum: float = 20.0,
    film_thickness: int = 1,
    substrate_thickness: int = 1,
    in_layers: bool = True,
    top_k: int = 3,
    max_termination_pairs: int = 3,
    termination: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Generate top-K coherent interface candidates ranked by strain × n_atoms.

    Returns a list of dicts:
      - interface: pymatgen Interface (Structure subclass)
      - termination: (substrate_term_label, film_term_label)
      - von_mises_strain: scalar strain magnitude
      - strain_tensor: 3x3 list
      - area: in-plane supercell area (Å²)
      - n_atoms: total atom count
      - film_transformation: 2x2 supercell matrix (film)
      - substrate_transformation: 2x2 supercell matrix (substrate)
    """
    zsl = ZSLGenerator(
        max_area=float(max_area),
        max_length_tol=float(max_length_tol),
        max_angle_tol=float(max_angle_tol),
        bidirectional=bool(bidirectional),
    )
    cib = CoherentInterfaceBuilder(
        substrate_structure=substrate,
        film_structure=film,
        film_miller=tuple(int(x) for x in film_miller),
        substrate_miller=tuple(int(x) for x in substrate_miller),
        zslgen=zsl,
    )

    if termination is not None:
        pair = (termination[1], termination[0])
        if pair not in cib.terminations:
            raise ValueError("The selected termination pair is unavailable for these materials.")
        term_pairs = [pair]
    else:
        term_pairs = list(cib.terminations)[: max(1, int(max_termination_pairs))]

    if not term_pairs:
        return []

    candidates: list[dict[str, Any]] = []
    for tp in term_pairs:
        try:
            ifcs = list(
                cib.get_interfaces(
                    termination=tp,
                    gap=float(gap),
                    vacuum_over_film=float(vacuum),
                    film_thickness=int(film_thickness),
                    substrate_thickness=int(substrate_thickness),
                    in_layers=bool(in_layers),
                )
            )
        except Exception:
            continue
        for ifc in ifcs:
            ip = getattr(ifc, "interface_properties", {}) or {}
            try:
                strain_tensor = np.asarray(ip.get("strain"), dtype=float)
            except Exception:
                strain_tensor = np.zeros((3, 3))
            try:
                vm = float(ip.get("von_mises_strain", 0.0))
            except Exception:
                vm = float("nan")
            try:
                sl = np.asarray(ip.get("film_sl_vectors"), dtype=float)
                area = float(np.linalg.norm(np.cross(sl[0], sl[1])))
            except Exception:
                area = float(ifc.lattice.a * ifc.lattice.b)

            candidates.append(
                {
                    "interface": ifc,
                    "termination": (tp[1], tp[0]),
                    "von_mises_strain": vm,
                    "strain_tensor": strain_tensor.tolist(),
                    "area": area,
                    "n_atoms": len(ifc),
                    "film_transformation": np.asarray(
                        ip.get("film_transformation"), dtype=float
                    ).tolist()
                    if ip.get("film_transformation") is not None
                    else None,
                    "substrate_transformation": np.asarray(
                        ip.get("substrate_transformation"), dtype=float
                    ).tolist()
                    if ip.get("substrate_transformation") is not None
                    else None,
                    "film_thickness": int(ip.get("film_thickness", film_thickness)),
                    "substrate_thickness": int(ip.get("substrate_thickness", substrate_thickness)),
                }
            )

    # Rank: smallest strain first, then smallest n_atoms (smaller cell preferred at equal strain)
    def _key(c: dict[str, Any]) -> tuple[float, int]:
        vm = c["von_mises_strain"]
        if vm != vm:  # NaN
            vm = float("inf")
        return (vm, c["n_atoms"])

    candidates.sort(key=_key)
    return candidates[: max(1, int(top_k))]
