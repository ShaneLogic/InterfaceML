"""
_utils_layering.py

Shared geometry/layering utilities for heterostructure processing.

All functions are pure helpers used by CLI scripts and should not perform file IO.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from pymatgen.core import Structure

try:
    # When executed as a module: `python -m build_heterojunctions.<script>`
    from ._utils_structures import element_symbols
except ImportError:
    # When executed as a script: `python build_heterojunctions/<script>.py`
    from _utils_structures import element_symbols


def interface_normal_unit(structure: Structure) -> np.ndarray:
    """
    Compute an interface normal from the in-plane lattice vectors (a x b).

    The normal is oriented to have a positive dot product with the c vector.
    """
    a = np.asarray(structure.lattice.matrix[0], dtype=float)
    b = np.asarray(structure.lattice.matrix[1], dtype=float)
    c = np.asarray(structure.lattice.matrix[2], dtype=float)
    n = np.cross(a, b)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / nn
    if float(np.dot(n, c)) < 0.0:
        n *= -1.0
    return n


def unwrap_periodic_1d(values_mod: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap a periodic 1D coordinate by cutting at the largest gap.

    Returns
    -------
    order_unwrapped
        Indices of atoms in unwrapped order.
    coord_unwrapped
        Unwrapped coordinate in [0, period) with a cut at the largest gap.
    """
    order = np.argsort(values_mod)
    v = values_mod[order]
    diffs = np.diff(v)
    wrap_gap = float(v[0] + period - v[-1]) if len(v) > 1 else float(period)
    diffs_circ = np.concatenate([diffs, np.array([wrap_gap], dtype=float)])
    k_max = int(np.argmax(diffs_circ))
    start = (k_max + 1) % len(order) if len(order) > 0 else 0
    order_unwrapped = np.concatenate([order[start:], order[:start]])
    v0 = float(values_mod[order_unwrapped[0]]) if len(order_unwrapped) else 0.0
    coord_unwrapped = np.mod(values_mod[order_unwrapped] - v0, float(period))
    return order_unwrapped, coord_unwrapped


def split_stack_layers(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
) -> list[list[int]]:
    """
    Split a stacked heterostructure into (n_interfaces + 1) layers by largest gaps.

    Returns a list of 0-based index lists, ordered from bottom to top.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")
    n_layers = n_interfaces + 1

    n = interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n)
    c_vec = np.asarray(structure.lattice.matrix[2], dtype=float)
    period = float(abs(np.dot(c_vec, n)))
    if period <= 1e-8:
        period = float(np.max(heights) - np.min(heights) + 1.0)

    h_mod = np.mod(heights, period)
    order_unwrapped, h_unwrapped = unwrap_periodic_1d(h_mod, period)
    if len(order_unwrapped) == 0:
        return []

    diffs = np.diff(h_unwrapped)
    if len(diffs) == 0:
        return [order_unwrapped.tolist()]

    cut_needed = n_layers - 1
    gap_order = np.argsort(diffs)[::-1]
    cuts: list[int] = []
    for idx in gap_order:
        if float(diffs[idx]) < float(min_gap):
            continue
        cuts.append(int(idx) + 1)
        if len(cuts) >= cut_needed:
            break
    cuts = sorted(set(cuts))

    starts = [0] + cuts
    ends = cuts + [len(order_unwrapped)]
    groups: list[list[int]] = []
    for s, e in zip(starts, ends):
        groups.append([int(i) for i in order_unwrapped[s:e].tolist()])

    groups.sort(key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0)
    return groups


def layer_indices_to_string(indices_0based: list[int], *, one_based: bool = True) -> str:
    """Format indices for printing."""
    if one_based:
        return " ".join(str(i + 1) for i in indices_0based)
    return " ".join(str(i) for i in indices_0based)


def auto_layer_tol(diffs: np.ndarray) -> float:
    """
    Auto-estimate a layer-separation tolerance (Å) from successive height differences.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs > 1e-8]
    if diffs.size == 0:
        return 0.35
    med = float(np.median(diffs))
    thr = 6.0 * med
    return float(max(0.20, min(1.00, thr)))


def split_layers_by_z(
    structure: Structure,
    *,
    tol: float | None = None,
    gap_cut: bool = True,
) -> tuple[list[list[int]], float]:
    """
    Split ALL atoms into z-layers by clustering projected heights along the interface normal.

    Returns layers (0-based indices) ordered bottom->top and the tolerance used.
    """
    n = interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n)

    c_vec = np.asarray(structure.lattice.matrix[2], dtype=float)
    period = float(abs(np.dot(c_vec, n)))
    if period <= 1e-8:
        period = float(np.max(heights) - np.min(heights) + 1.0)

    h_mod = np.mod(heights, period)
    if gap_cut:
        order, h_unwrapped = unwrap_periodic_1d(h_mod, period)
    else:
        order = np.argsort(h_mod)
        h_unwrapped = h_mod[order]

    if len(order) == 0:
        return [], float(tol or 0.35)

    diffs = np.diff(h_unwrapped)
    tol_used = float(tol) if tol is not None else auto_layer_tol(diffs)

    layers: list[list[int]] = []
    current: list[int] = [int(order[0])]
    for i in range(1, len(order)):
        if float(h_unwrapped[i] - h_unwrapped[i - 1]) > tol_used:
            layers.append(current)
            current = [int(order[i])]
        else:
            current.append(int(order[i]))
    layers.append(current)

    layers.sort(key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0)
    return layers, tol_used


def connected_components_by_distance(
    structure: Structure,
    indices: np.ndarray,
    *,
    cutoffs: Dict[Tuple[str, str], float],
) -> list[list[int]]:
    """
    Find connected components for a subset of atoms using distance cutoffs.
    """
    if indices.size == 0:
        return []
    idx_list = [int(i) for i in indices.tolist()]
    coords = np.asarray(structure.cart_coords, dtype=float)
    syms = element_symbols(structure)
    n = len(idx_list)

    adj: list[list[int]] = [[] for _ in range(n)]
    for a in range(n):
        ia = idx_list[a]
        sa = syms[ia]
        pa = coords[ia]
        for b in range(a + 1, n):
            ib = idx_list[b]
            sb = syms[ib]
            dmax = cutoffs.get((sa, sb)) or cutoffs.get((sb, sa))
            if dmax is None:
                continue
            if float(np.linalg.norm(pa - coords[ib])) <= float(dmax):
                adj[a].append(b)
                adj[b].append(a)

    seen = [False] * n
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp_local: list[int] = []
        while stack:
            u = stack.pop()
            comp_local.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append([idx_list[i] for i in comp_local])
    return comps


def include_whole_molecules(
    structure: Structure,
    fixed_indices_0based: list[int],
    *,
    molecule_elements: Set[str] | None = None,
) -> list[int]:
    """
    Expand fixed indices so that partially selected organic molecules are fully included.
    """
    if molecule_elements is None:
        molecule_elements = {"C", "N", "H"}
    syms = element_symbols(structure)
    mol_idx = np.array([i for i, s in enumerate(syms) if s in molecule_elements], dtype=int)

    cutoffs = {
        ("C", "N"): 1.75,
        ("C", "H"): 1.25,
        ("N", "H"): 1.25,
        ("N", "C"): 1.75,
        ("H", "C"): 1.25,
        ("H", "N"): 1.25,
    }
    comps = connected_components_by_distance(structure, mol_idx, cutoffs=cutoffs)
    fixed_set = set(int(i) for i in fixed_indices_0based)
    for comp in comps:
        if any(i in fixed_set for i in comp):
            fixed_set.update(comp)
    return sorted(fixed_set)


# Prefer canonical core implementations when available
try:
    from interfaceml.core import layering as _core_layering
except ImportError:  # pragma: no cover - fallback for standalone usage
    _core_layering = None  # type: ignore[assignment]

if _core_layering is not None:
    interface_normal_unit = _core_layering.interface_normal_unit  # type: ignore[assignment]
    unwrap_periodic_1d = _core_layering.unwrap_periodic_1d  # type: ignore[assignment]
    split_stack_layers = _core_layering.split_stack_layers  # type: ignore[assignment]
    connected_components_by_distance = _core_layering.connected_components_by_distance  # type: ignore[assignment]

    def auto_layer_tol(diffs: np.ndarray) -> float:  # type: ignore[override]
        return _core_layering.auto_layer_tolerance(diffs)

    def split_layers_by_z(  # type: ignore[override]
        structure: Structure,
        *,
        tol: float | None = None,
        gap_cut: bool = True,
    ) -> tuple[list[list[int]], float]:
        return _core_layering.split_layers_by_z(structure, tolerance=tol, gap_cut=gap_cut)

    def include_whole_molecules(  # type: ignore[override]
        structure: Structure,
        fixed_indices_0based: list[int],
        *,
        molecule_elements: Set[str] | None = None,
    ) -> list[int]:
        return _core_layering.include_whole_molecules(
            structure,
            fixed_indices_0based,
            molecule_elements=molecule_elements,
        )

    def layer_indices_to_string(  # type: ignore[override]
        indices_0based: list[int],
        *,
        one_based: bool = True,
    ) -> str:
        return _core_layering.format_layer_indices(indices_0based, one_based=one_based)
