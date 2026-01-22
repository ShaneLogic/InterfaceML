"""
Layer detection and manipulation module for InterfaceML.

This module provides algorithms for:
- Computing interface normals from lattice vectors
- Unwrapping periodic coordinates
- Splitting stacked structures into layers by gap detection
- Clustering atoms into z-layers for selective dynamics
- Identifying and including whole organic molecules in fixed regions

These tools are essential for setting up relaxation calculations with
proper constraints on bulk and interface regions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pymatgen.core import Structure

from interfaceml.core.io import get_element_symbols


def interface_normal_unit(structure: Structure) -> np.ndarray:
    """
    Compute a unit normal vector to the interface plane.

    The normal is derived from the cross product of in-plane lattice vectors
    (a × b) and oriented to have a positive projection along the c vector.

    Parameters
    ----------
    structure
        A pymatgen Structure object.

    Returns
    -------
    normal
        Unit normal vector, shape (3,).
    """
    a = np.asarray(structure.lattice.matrix[0], dtype=float)
    b = np.asarray(structure.lattice.matrix[1], dtype=float)
    c = np.asarray(structure.lattice.matrix[2], dtype=float)

    n = np.cross(a, b)
    norm = float(np.linalg.norm(n))

    if norm < 1e-12:
        # Fallback for degenerate cases (2D or collinear a, b)
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / norm

    # Ensure positive projection along c
    if float(np.dot(n, c)) < 0.0:
        n = -n

    return n


def unwrap_periodic_1d(
    values_mod: np.ndarray,
    period: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unwrap a periodic 1D coordinate by cutting at the largest gap.

    This is used to handle periodic boundary conditions when identifying
    layers in a slab with vacuum. The cut is placed at the largest gap
    to avoid splitting physical layers across the periodic boundary.

    Parameters
    ----------
    values_mod
        1D array of coordinate values mapped to [0, period).
    period
        The periodicity length.

    Returns
    -------
    order_unwrapped
        Indices of atoms in unwrapped order (sorted by unwrapped coordinate).
    coord_unwrapped
        Unwrapped coordinates in [0, period), with the cut at the largest gap.
        
    Notes
    -----
    The algorithm works by:
    1. Sorting coordinates along the periodic direction
    2. Finding the largest gap (typically vacuum in slab calculations)
    3. Cutting at that gap to unwrap the periodic boundary
    4. Reordering atoms so the cut appears at the start
    """
    # Early return for empty arrays
    if len(values_mod) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Single atom case
    if len(values_mod) == 1:
        return np.array([0], dtype=int), np.array([0.0], dtype=float)

    order = np.argsort(values_mod)
    v = values_mod[order]

    # Compute gaps between consecutive sorted values
    diffs = np.diff(v)

    # Also consider the wraparound gap
    wrap_gap = float(v[0] + period - v[-1])
    diffs_circ = np.concatenate([diffs, [wrap_gap]])

    # Find the largest gap (this is where we cut the periodic boundary)
    k_max = int(np.argmax(diffs_circ))
    start = (k_max + 1) % len(order)

    # Reorder so that the cut is at the beginning
    order_unwrapped = np.concatenate([order[start:], order[:start]])

    # Compute unwrapped coordinates relative to the first atom
    v0 = float(values_mod[order_unwrapped[0]])
    coord_unwrapped = np.mod(values_mod[order_unwrapped] - v0, float(period))

    return order_unwrapped, coord_unwrapped


def split_stack_layers(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
) -> List[List[int]]:
    """
    Split a stacked heterostructure into layers by detecting large gaps.

    This function is designed for multi-layer stacks (e.g., perovskite/C70/C60)
    where interfaces are separated by vacuum or low-density regions. It identifies
    the (n_interfaces) largest gaps and uses them to define (n_interfaces + 1) layers.

    Parameters
    ----------
    structure
        The stacked structure to split.
    n_interfaces
        Number of interfaces (gaps) to identify. The structure will be split
        into (n_interfaces + 1) layers.
    min_gap
        Minimum gap size (in Angstroms) to consider. Gaps smaller than this
        are ignored.

    Returns
    -------
    layers
        List of layer index lists (0-based atom indices), ordered from
        bottom to top along the interface normal.

    Raises
    ------
    ValueError
        If n_interfaces < 1.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")

    n_layers = n_interfaces + 1
    n = interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n)

    # Compute period from c-vector projection
    c_vec = np.asarray(structure.lattice.matrix[2], dtype=float)
    period = float(abs(np.dot(c_vec, n)))
    if period <= 1e-8:
        # Fallback for non-periodic or poorly defined cells
        period = float(np.max(heights) - np.min(heights) + 1.0)

    # Unwrap periodic coordinates by cutting at the largest gap (typically vacuum)
    h_mod = np.mod(heights, period)
    order_unwrapped, h_unwrapped = unwrap_periodic_1d(h_mod, period)

    if len(order_unwrapped) == 0:
        return []

    diffs = np.diff(h_unwrapped)
    if len(diffs) == 0:
        return [order_unwrapped.tolist()]

    # Identify the (n_layers - 1) largest gaps as layer boundaries
    cut_needed = n_layers - 1
    gap_order = np.argsort(diffs)[::-1]
    cuts: List[int] = []
    for idx in gap_order:
        if float(diffs[idx]) < float(min_gap):
            continue
        cuts.append(int(idx) + 1)  # Cut between idx and idx+1
        if len(cuts) >= cut_needed:
            break

    cuts = sorted(set(cuts))

    # Split the unwrapped indices into groups
    starts = [0] + cuts
    ends = cuts + [len(order_unwrapped)]
    groups: List[List[int]] = []
    for s, e in zip(starts, ends):
        groups.append([int(i) for i in order_unwrapped[s:e].tolist()])

    # Sort groups by mean absolute height for stable bottom-to-top ordering
    groups.sort(
        key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0
    )

    return groups


def auto_layer_tolerance(height_diffs: np.ndarray) -> float:
    """
    Automatically estimate a layer-separation tolerance from height differences.

    For well-relaxed slabs, atoms within a layer have small height separations
    (< 0.3 Å), while adjacent layers are separated by larger gaps (> 1.5 Å).
    This function estimates a reasonable threshold based on the median gap.

    Parameters
    ----------
    height_diffs
        Successive height differences (sorted order), shape (N-1,).

    Returns
    -------
    tolerance
        Estimated layer separation threshold in Angstroms, clamped to [0.2, 1.0].
        
    Notes
    -----
    Uses a heuristic where inter-layer gaps are typically ~6x larger than
    intra-layer atomic separations. The result is clamped to reasonable
    physical values for atomic structures.
    """
    # Filter to keep only meaningful differences (avoid numerical noise)
    diffs = np.asarray(height_diffs, dtype=float)
    diffs = diffs[diffs > 1e-8]

    if diffs.size == 0:
        return 0.35  # Default fallback for edge cases

    # Use median for robustness against outliers
    median = float(np.median(diffs))
    threshold = 6.0 * median  # Heuristic: inter-layer gaps are ~6x intra-layer
    
    # Clamp to physically reasonable range [0.2, 1.0] Angstroms
    return float(np.clip(threshold, 0.20, 1.00))


def split_layers_by_z(
    structure: Structure,
    *,
    tolerance: Optional[float] = None,
    gap_cut: bool = True,
) -> Tuple[List[List[int]], float]:
    """
    Cluster all atoms into z-layers based on projected heights.

    This function is the core algorithm for fixing bottom layers in selective
    dynamics calculations. It projects all atoms onto the interface normal,
    optionally unwraps the periodic coordinate, and clusters them into layers
    using a height-based tolerance.

    Parameters
    ----------
    structure
        The structure to analyze.
    tolerance
        Layer separation threshold in Angstroms. If None, it is automatically
        estimated from the height distribution.
    gap_cut
        If True, unwrap periodic coordinates by cutting at the largest gap
        (typically vacuum). Set to False for fully periodic structures.

    Returns
    -------
    layers
        List of 0-based atom index lists, ordered from bottom to top.
    tolerance_used
        The tolerance actually used (either provided or auto-estimated).
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
        return [], float(tolerance or 0.35)

    diffs = np.diff(h_unwrapped)
    tol_used = float(tolerance) if tolerance is not None else auto_layer_tolerance(diffs)

    # Cluster atoms into layers using the tolerance
    layers: List[List[int]] = []
    current: List[int] = [int(order[0])]
    for i in range(1, len(order)):
        if float(h_unwrapped[i] - h_unwrapped[i - 1]) > tol_used:
            layers.append(current)
            current = [int(order[i])]
        else:
            current.append(int(order[i]))
    layers.append(current)

    # Sort layers by mean absolute height for stable ordering
    layers.sort(
        key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0
    )

    return layers, tol_used


def connected_components_by_distance(
    structure: Structure,
    indices: np.ndarray,
    *,
    cutoffs: Dict[Tuple[str, str], float],
) -> List[List[int]]:
    """
    Find connected components (molecules) using distance-based bonding criteria.

    This is used to identify intact organic molecules (e.g., MA, FA cations)
    in perovskite slabs, ensuring that when fixing atoms for selective dynamics,
    entire molecules are included rather than being split.

    Parameters
    ----------
    structure
        The structure containing the atoms.
    indices
        Subset of atom indices to analyze (0-based).
    cutoffs
        Dictionary of bonding distance cutoffs for element pairs, e.g.,
        {("C", "N"): 1.75, ("C", "H"): 1.25}.

    Returns
    -------
    components
        List of connected components, each a list of global atom indices.
        
    Notes
    -----
    Uses depth-first search (DFS) for component detection. Time complexity
    is O(n^2) for pairwise distance checks plus O(n + e) for DFS traversal,
    where n is the number of atoms and e is the number of bonds.
    """
    if indices.size == 0:
        return []

    idx_list = [int(i) for i in indices.tolist()]
    coords = np.asarray(structure.cart_coords, dtype=float)
    syms = get_element_symbols(structure)
    n = len(idx_list)

    # Build adjacency list with optimized distance checks
    adj: List[List[int]] = [[] for _ in range(n)]
    
    # Pre-extract coordinates and symbols for faster access
    local_coords = coords[idx_list]
    local_syms = [syms[ia] for ia in idx_list]
    
    for a in range(n):
        sa = local_syms[a]
        pa = local_coords[a]
        for b in range(a + 1, n):
            sb = local_syms[b]
            # Try both orderings for cutoff lookup
            dmax = cutoffs.get((sa, sb)) or cutoffs.get((sb, sa))
            if dmax is None:
                continue
            # Compute squared distance to avoid sqrt when possible
            dist_sq = np.sum((pa - local_coords[b]) ** 2)
            if dist_sq <= dmax * dmax:
                adj[a].append(b)
                adj[b].append(a)

    # DFS to find connected components
    seen = [False] * n
    comps: List[List[int]] = []
    
    for start in range(n):
        if seen[start]:
            continue
        # Use iterative DFS to avoid recursion limit issues
        stack = [start]
        seen[start] = True
        comp_local: List[int] = []
        
        while stack:
            u = stack.pop()
            comp_local.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        
        # Convert local indices back to global
        comps.append([idx_list[i] for i in comp_local])

    return comps


def include_whole_molecules(
    structure: Structure,
    fixed_indices: List[int],
    *,
    molecule_elements: Optional[Set[str]] = None,
) -> List[int]:
    """
    Expand a fixed-atom list to include complete organic molecules.

    When setting up selective dynamics for interface calculations, it's important
    to avoid splitting organic cations (e.g., MA, FA) across the fixed/relaxed
    boundary, as this can lead to unphysical geometries. This function identifies
    molecules and ensures that if any atom of a molecule is fixed, the entire
    molecule is fixed.

    Parameters
    ----------
    structure
        The structure containing the atoms.
    fixed_indices
        Initial list of 0-based atom indices to fix.
    molecule_elements
        Set of element symbols considered to be part of organic molecules.
        Defaults to {"C", "N", "H"}.

    Returns
    -------
    expanded_indices
        Sorted list of 0-based atom indices with whole molecules included.
    """
    if molecule_elements is None:
        molecule_elements = {"C", "N", "H"}

    syms = get_element_symbols(structure)
    mol_idx = np.array(
        [i for i, s in enumerate(syms) if s in molecule_elements], dtype=int
    )

    # Conservative bonding cutoffs for organic molecules (Angstroms)
    cutoffs = {
        ("C", "N"): 1.75,
        ("C", "H"): 1.25,
        ("N", "H"): 1.25,
        ("N", "C"): 1.75,
        ("H", "C"): 1.25,
        ("H", "N"): 1.25,
    }

    comps = connected_components_by_distance(structure, mol_idx, cutoffs=cutoffs)

    fixed_set = set(int(i) for i in fixed_indices)
    for comp in comps:
        if any(i in fixed_set for i in comp):
            fixed_set.update(comp)

    return sorted(fixed_set)


def format_layer_indices(
    indices: List[int],
    *,
    one_based: bool = True,
) -> str:
    """
    Format a list of atom indices as a space-separated string.

    Parameters
    ----------
    indices
        List of 0-based atom indices.
    one_based
        If True, convert to 1-based indexing (common in DFT codes like CP2K).

    Returns
    -------
    formatted
        Space-separated string of indices.
    """
    if one_based:
        return " ".join(str(i + 1) for i in indices)
    return " ".join(str(i) for i in indices)
