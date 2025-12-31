#!/usr/bin/env python3
"""
fix_interface_layers.py

This script adds Selective Dynamics constraints to a heterojunction POSCAR file,
relaxing the two closest layers at the interface (one from each material) and fixing all other layers.

The script:
1. Reads a combined heterojunction POSCAR file
2. Identifies the interface position (boundary between two materials)
3. Finds the two closest atomic layers to the interface (one from each side)
4. Adds Selective Dynamics flags: T T T (free/relax) for interface layers, F F F (fixed) for others
5. Writes a new POSCAR file with Selective Dynamics

Author: Generated for heterojunction interface relaxation
Requirements: pymatgen, numpy
"""

import argparse
import re
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

# Shared utilities (engineering refactor):
# - Keep the CLI behavior unchanged
# - Delegate reusable logic to helper modules
try:
    # When executed as a module: `python -m build_heterojunctions.fix_interface_layers ...`
    from ._utils_structures import load_structure_any as _load_structure_any
    from ._utils_layering import (
        interface_normal_unit as _interface_normal_unit,
        unwrap_periodic_1d as _unwrap_periodic_1d_shared,
        split_stack_layers as _split_stack_layers,
        layer_indices_to_string as _layer_indices_to_string,
        auto_layer_tol as _auto_layer_tol_shared,
        split_layers_by_z as _split_layers_by_z_shared,
        include_whole_molecules as _include_whole_molecules_shared,
    )
except ImportError:
    # When executed as a script: `python build_heterojunctions/fix_interface_layers.py ...`
    from _utils_structures import load_structure_any as _load_structure_any
    from _utils_layering import (
        interface_normal_unit as _interface_normal_unit,
        unwrap_periodic_1d as _unwrap_periodic_1d_shared,
        split_stack_layers as _split_stack_layers,
        layer_indices_to_string as _layer_indices_to_string,
        auto_layer_tol as _auto_layer_tol_shared,
        split_layers_by_z as _split_layers_by_z_shared,
        include_whole_molecules as _include_whole_molecules_shared,
    )


_TV_RE = re.compile(
    r"Tv_1:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+Tv_2:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+Tv_3:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


def _read_cp2k_xyz_last_frame(path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """
    Read the last frame of a CP2K-style XYZ file.

    Expected format:
      <natoms>
      Tv_1: ax ay az Tv_2: bx by bz Tv_3: cx cy cz   (optional but recommended)
      Elem x y z
      ...
    """
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file too short: {path}")

    i = 0
    last_block: tuple[int, int] | None = None
    last_cell: np.ndarray | None = None
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            nat = int(lines[i].strip().split()[0])
        except Exception:
            break
        if i + 1 + nat >= len(lines):
            break
        comment = lines[i + 1].strip()
        m = _TV_RE.search(comment)
        if m:
            vals = [float(x) for x in m.groups()]
            last_cell = np.array(vals, dtype=float).reshape(3, 3)
        last_block = (i, i + 2 + nat)
        i = i + 2 + nat

    if last_block is None:
        raise ValueError(f"Could not parse any XYZ frame from: {path}")

    start, _end = last_block
    nat = int(lines[start].strip().split()[0])
    symbols: list[str] = []
    coords: list[list[float]] = []
    for ln in lines[start + 2 : start + 2 + nat]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != nat:
        raise ValueError(f"Expected {nat} atoms but parsed {len(symbols)} atoms from: {path}")
    return symbols, np.array(coords, dtype=float), last_cell


def load_structure_any(path_str: str) -> Structure:
    """
    Load a structure from POSCAR/CIF/XYZ.

    For XYZ, we expect CP2K-style cell vectors (Tv_1/Tv_2/Tv_3) on the comment line.
    If not found, the code falls back to a large orthorhombic box.
    """
    # Delegate to the shared implementation for consistent behavior across tools.
    return _load_structure_any(path_str)


def _element_symbols(structure: Structure) -> list[str]:
    """Return plain element symbols for each site."""
    return [str(sp) for sp in structure.species]


def _connected_components_by_distance(
    structure: Structure,
    indices: np.ndarray,
    *,
    cutoffs: dict[tuple[str, str], float],
) -> list[list[int]]:
    """
    Find connected components for a subset of atoms using simple distance cutoffs.

    This is used to keep organic molecules intact when selecting fixed atoms:
    if any atom of a molecule is selected, include the whole molecule.
    """
    if indices.size == 0:
        return []

    idx_list = [int(i) for i in indices.tolist()]
    coords = np.asarray(structure.cart_coords, dtype=float)
    syms = _element_symbols(structure)
    n = len(idx_list)

    # Adjacency on the local index space.
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


def _include_whole_molecules(
    structure: Structure,
    fixed_indices_0based: list[int],
    *,
    molecule_elements: set[str],
) -> list[int]:
    """
    Expand a fixed-index list so that partially selected organic molecules are fully included.

    If any atom of an organic molecule is fixed, all atoms of that molecule are added.
    """
    return _include_whole_molecules_shared(structure, fixed_indices_0based, molecule_elements=set(molecule_elements))


def interface_normal_unit(structure: Structure) -> np.ndarray:
    """
    Compute an interface normal from the in-plane lattice vectors (a x b).

    The normal is oriented to have a positive dot product with the c vector.
    """
    return _interface_normal_unit(structure)


def _unwrap_periodic_1d(values_mod: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap a periodic 1D coordinate by cutting at the largest gap.

    Parameters
    ----------
    values_mod
        Values mapped to [0, period).
    period
        Period length.

    Returns
    -------
    order_unwrapped
        Indices of atoms in unwrapped order.
    coord_unwrapped
        Unwrapped coordinate in [0, period) with a cut at the largest gap.
    """
    return _unwrap_periodic_1d_shared(values_mod, period)


def split_stack_layers(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
) -> list[list[int]]:
    """
    Split a stacked heterostructure into (n_interfaces + 1) layers by largest gaps.

    This is intended for slab/molecule stacks and multi-layer heterostructures where
    interfaces are separated by low-density gaps along the interface normal.

    Returns a list of index lists (0-based indices into structure.sites), ordered
    from bottom to top.
    """
    return _split_stack_layers(structure, n_interfaces=n_interfaces, min_gap=min_gap)


def layer_indices_to_string(indices_0based: list[int], *, one_based: bool = True) -> str:
    """
    Format indices for printing / CP2K style lists.
    """
    return _layer_indices_to_string(indices_0based, one_based=one_based)


def _auto_layer_tol(diffs: np.ndarray) -> float:
    """
    Auto-estimate a layer-separation tolerance (Å) from successive height differences.

    For relaxed slabs, atoms inside a layer have small height separations, while adjacent
    layers produce noticeably larger gaps. A robust heuristic is proportional to the median.
    """
    return _auto_layer_tol_shared(diffs)


def split_layers_by_z(
    structure: Structure,
    *,
    tol: float | None = None,
    gap_cut: bool = True,
) -> tuple[list[list[int]], float]:
    """
    Split ALL atoms into z-layers by clustering projected heights along the interface normal.

    This is a direct replacement for the earlier fixed_atoms_from_xyz workflow:
    - compute interface normal from a x b
    - project cart coords onto the normal
    - unwrap the periodic coordinate by cutting at the largest gap (typically vacuum) if gap_cut=True
    - cluster into layers using tol (Å)

    Returns
    -------
    layers : list[list[int]]
        List of 0-based atom indices per layer, ordered from bottom to top.
    tol_used : float
        The tolerance actually used (either provided tol or auto-estimated).
    """
    return _split_layers_by_z_shared(structure, tol=tol, gap_cut=gap_cut)


def print_fixed_atoms_by_z_layers(
    input_file: str,
    *,
    n_fix_layers: int,
    tol: float | None,
    one_based_output: bool,
    gap_cut: bool = True,
    debug: bool = False,
    include_molecules: bool = True,
    molecule_elements: set[str] | None = None,
) -> list[int]:
    """
    Print FIXED_ATOMS LIST by fixing the bottom N z-layers.

    Returns the fixed atom indices (0-based).
    """
    structure = load_structure_any(input_file)
    layers, tol_used = split_layers_by_z(structure, tol=tol, gap_cut=gap_cut)
    if not layers:
        raise ValueError("Could not detect any layers.")
    if n_fix_layers < 1:
        raise ValueError("n_fix_layers must be >= 1")
    if n_fix_layers > len(layers):
        n_fix_layers = len(layers)

    fixed = sorted({i for layer in layers[:n_fix_layers] for i in layer})
    if include_molecules:
        if molecule_elements is None:
            molecule_elements = {"C", "N", "H"}
        fixed = _include_whole_molecules(structure, fixed, molecule_elements=set(molecule_elements))
    if debug:
        print(f"Detected {len(layers)} layers | tol_used={tol_used:.3f} Å")
        for k, layer in enumerate(layers[: min(10, len(layers))], start=1):
            zs = structure.cart_coords[np.array(layer, dtype=int), 2]
            print(f" - Layer {k}: {len(layer)} atoms | z_range=[{zs.min():.3f}, {zs.max():.3f}]")
        if len(layers) > 10:
            print(" - ... (more layers omitted)")

    print("FIXED_ATOMS LIST:", layer_indices_to_string(fixed, one_based=one_based_output))
    return fixed

def identify_interface_z(structure, method='density_gap'):
    """
    Identify the z-coordinate of the interface between two materials.
    
    Args:
        structure: pymatgen Structure object
        method: Method to identify interface ('density_gap' or 'median')
    
    Returns:
        float: z-coordinate of the interface
    """
    z_coords = structure.cart_coords[:, 2]
    
    if method == 'density_gap':
        # Find z-coordinate with minimum atomic density (gap between materials)
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min
        n_bins = max(50, len(structure) // 10)  # Adaptive binning
        bins = np.linspace(z_min, z_max, n_bins)
        hist, bin_edges = np.histogram(z_coords, bins=bins)
        
        # Find bin with minimum density (gap)
        min_density_idx = np.argmin(hist)
        interface_z = (bin_edges[min_density_idx] + bin_edges[min_density_idx + 1]) / 2.0
        
    elif method == 'median':
        # Use median as interface (simple but may not be accurate)
        interface_z = np.median(z_coords)
    
    else:
        # Default: find the largest gap in z-coordinates
        sorted_z = np.sort(z_coords)
        gaps = np.diff(sorted_z)
        max_gap_idx = np.argmax(gaps)
        interface_z = (sorted_z[max_gap_idx] + sorted_z[max_gap_idx + 1]) / 2.0
    
    return interface_z


def find_interface_layers(structure, interface_z, n_layers_per_side=1, layer_thickness=2.0):
    """
    Find atomic layers closest to the interface on both sides.
    Selects n_layers_per_side distinct atomic layers from each side (TiO2 and perovskite).
    
    Args:
        structure: pymatgen Structure object
        interface_z: z-coordinate of the interface
        n_layers_per_side: Number of layers to relax on each side (default: 1)
        layer_thickness: Thickness threshold for grouping atoms into layers (Angstroms)
    
    Returns:
        list: List of site indices to relax (at interface layers)
    """
    z_coords = structure.cart_coords[:, 2]
    
    # Separate atoms into bottom (substrate, e.g., TiO2) and top (film, e.g., perovskite)
    bottom_indices = [i for i, z in enumerate(z_coords) if z < interface_z]
    top_indices = [i for i, z in enumerate(z_coords) if z >= interface_z]
    
    if len(bottom_indices) == 0 or len(top_indices) == 0:
        raise ValueError("Cannot identify interface: all atoms on one side")
    
    # Find z-coordinates for bottom and top materials
    bottom_z = [z_coords[i] for i in bottom_indices]
    top_z = [z_coords[i] for i in top_indices]
    
    # Round z-coordinates to identify distinct atomic layers
    # Use reasonable precision (0.01 Å) to group atoms that are essentially at the same z-level
    z_precision = 0.01  # 0.01 Å precision for layer identification
    
    bottom_z_rounded = [round(z, 2) for z in bottom_z]  # Round to 0.01 Å
    top_z_rounded = [round(z, 2) for z in top_z]
    
    # Get unique z-levels for each side (these represent distinct atomic layers)
    bottom_unique_z = sorted(set(bottom_z_rounded), reverse=True)  # Highest z first (closest to interface)
    top_unique_z = sorted(set(top_z_rounded))  # Lowest z first (closest to interface)
    
    # Select n_layers_per_side unique z-levels closest to interface
    # Each z-level represents a distinct atomic layer
    selected_bottom_z_levels = bottom_unique_z[:n_layers_per_side] if len(bottom_unique_z) >= n_layers_per_side else bottom_unique_z
    selected_top_z_levels = top_unique_z[:n_layers_per_side] if len(top_unique_z) >= n_layers_per_side else top_unique_z
    
    # Collect atoms in selected layers (these will be relaxed)
    # Use z_precision (0.01 Å) to match atoms to exact z-levels, not layer_thickness
    # layer_thickness is only used for documentation/clustering purposes, not for selection
    relaxed_indices = []
    
    # Bottom side (TiO2): select atoms in the closest n_layers_per_side layers
    for selected_z in selected_bottom_z_levels:
        for i in bottom_indices:
            z = z_coords[i]
            # Check if atom's z (rounded) matches the selected z-level exactly
            if abs(round(z, 2) - selected_z) < z_precision:
                relaxed_indices.append(i)
    
    # Top side (perovskite): select atoms in the closest n_layers_per_side layers
    for selected_z in selected_top_z_levels:
        for i in top_indices:
            z = z_coords[i]
            # Check if atom's z (rounded) matches the selected z-level exactly
            if abs(round(z, 2) - selected_z) < z_precision:
                relaxed_indices.append(i)
    
    # Remove duplicates
    relaxed_indices = list(set(relaxed_indices))
    
    return relaxed_indices


def add_selective_dynamics(poscar_file, output_file, n_layers_per_side=1, 
                          layer_thickness=2.0, interface_method='density_gap'):
    """
    Add Selective Dynamics to a POSCAR file, relaxing interface layers and fixing others.
    
    Args:
        poscar_file: Input POSCAR file path
        output_file: Output POSCAR file path
        n_layers_per_side: Number of layers to relax on each side of interface
        layer_thickness: Thickness threshold for defining a layer (Angstroms)
        interface_method: Method to identify interface ('density_gap', 'median', or 'max_gap')
    """
    print(f"Reading POSCAR file: {poscar_file}")
    structure = load_structure_any(poscar_file)
    
    print(f"Structure contains {len(structure)} atoms")
    print(f"Lattice parameters: {structure.lattice.abc}")
    
    # Identify interface position
    interface_z = identify_interface_z(structure, method=interface_method)
    print(f"Identified interface at z = {interface_z:.4f} Å")
    
    # Find z-coordinate range
    z_coords = structure.cart_coords[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    print(f"Z-coordinate range: {z_min:.4f} - {z_max:.4f} Å")
    
    # Find interface layers to relax
    relaxed_indices = find_interface_layers(structure, interface_z, 
                                           n_layers_per_side=n_layers_per_side,
                                           layer_thickness=layer_thickness)
    
    print(f"Found {len(relaxed_indices)} atoms in interface layers to relax")
    
    # Create Selective Dynamics flags
    # T T T = free (relax), F F F = fixed
    selective_dynamics = []
    for i in range(len(structure)):
        if i in relaxed_indices:
            selective_dynamics.append([True, True, True])  # Free (relax interface layers)
        else:
            selective_dynamics.append([False, False, False])  # Fixed (all other layers)
    
    # Add selective dynamics to structure
    structure.add_site_property("selective_dynamics", selective_dynamics)
    
    # Write POSCAR with Selective Dynamics
    poscar = Poscar(structure)
    poscar.write_file(output_file)
    
    print(f"✓ Written POSCAR with Selective Dynamics to: {output_file}")
    print(f"  Relaxed atoms: {len(relaxed_indices)} (interface layers)")
    print(f"  Fixed atoms: {len(structure) - len(relaxed_indices)} (all other layers)")
    
    # Print some statistics
    relaxed_z_coords = [z_coords[i] for i in relaxed_indices]
    if len(relaxed_z_coords) > 0:
        print(f"  Relaxed layer z-range: {min(relaxed_z_coords):.4f} - {max(relaxed_z_coords):.4f} Å")


def add_selective_dynamics_by_layers(
    poscar_file: str,
    output_file: str,
    *,
    n_interfaces: int,
    fixed_layers: list[int],
    min_gap: float = 0.0,
    one_based_output: bool = True,
    print_only: bool = False,
) -> None:
    """
    Add Selective Dynamics by splitting the structure into multiple layers.

    This mode is designed for workflows where you want to fix (or select) entire layers
    in a stacked heterostructure (e.g., slab/C70/C60) and output the corresponding
    atom indices.

    Parameters
    ----------
    fixed_layers
        Layer numbers to fix. Uses 1-based layer numbering in the CLI (Layer 1 = bottom layer).
    """
    print(f"Reading POSCAR file: {poscar_file}")
    structure = load_structure_any(poscar_file)
    print(f"Structure contains {len(structure)} atoms")

    groups = split_stack_layers(structure, n_interfaces=n_interfaces, min_gap=min_gap)
    if len(groups) < 2:
        raise ValueError("Failed to split into multiple layers. Try lowering --min_gap or check the structure.")

    print(f"Detected {len(groups)} layers for n_interfaces={n_interfaces} (expected {n_interfaces + 1})")
    for k, idxs in enumerate(groups, start=1):
        print(f" - Layer {k}: {len(idxs)} atoms")

    # Convert fixed layer numbers to a set of indices.
    fixed_layer_set = set(int(x) for x in fixed_layers)
    if any(x < 1 or x > len(groups) for x in fixed_layer_set):
        raise ValueError(f"fixed_layers must be between 1 and {len(groups)}")

    fixed_indices = sorted({i for k, idxs in enumerate(groups, start=1) if k in fixed_layer_set for i in idxs})
    print(f"Fixed atom count: {len(fixed_indices)}")
    print("FIXED_ATOMS LIST:", layer_indices_to_string(fixed_indices, one_based=one_based_output))

    if print_only:
        return

    # Build Selective Dynamics flags: fixed layers -> FFF, others -> TTT.
    sel = []
    fixed_set0 = set(fixed_indices)
    for i in range(len(structure)):
        if i in fixed_set0:
            sel.append([False, False, False])
        else:
            sel.append([True, True, True])
    structure.add_site_property("selective_dynamics", sel)
    Poscar(structure).write_file(output_file)
    print(f"✓ Written POSCAR with Selective Dynamics to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add Selective Dynamics to heterojunction POSCAR, relaxing interface layers and fixing others"
    )
    parser.add_argument(
        "input_file",
        help="Input POSCAR file (combined heterojunction structure)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        default=None,
        help="Output POSCAR file (default: input_file with '_relaxed' suffix)"
    )
    parser.add_argument(
        "-n", "--n_layers",
        dest="n_layers",
        type=int,
        default=1,
        help="Number of layers to relax on each side of interface (default: 1)"
    )
    parser.add_argument(
        "-t", "--thickness",
        dest="layer_thickness",
        type=float,
        default=2.0,
        help="Layer thickness threshold in Angstroms (default: 2.0)"
    )
    parser.add_argument(
        "-m", "--method",
        dest="interface_method",
        choices=['density_gap', 'median', 'max_gap'],
        default='density_gap',
        help="Method to identify interface position (default: density_gap)"
    )
    parser.add_argument(
        "--by_layers",
        action="store_true",
        help="Use multi-layer splitting mode: split by n_interfaces and fix selected layer(s).",
    )
    parser.add_argument(
        "--by_z_layers",
        action="store_true",
        help="Fix the bottom N z-layers (height clustering). Works for POSCAR/CIF/CP2K .xyz.",
    )
    parser.add_argument(
        "--n_interfaces",
        type=int,
        default=1,
        help="Number of interfaces in the stacked model (layers = n_interfaces + 1). Used with --by_layers.",
    )
    parser.add_argument(
        "--fixed_layers",
        type=str,
        default="1",
        help="Comma-separated layer numbers to fix (1-based; Layer 1 is bottom). Used with --by_layers.",
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.0,
        help="Minimum gap (Å) to accept as a layer boundary in --by_layers mode (default: 0.0).",
    )
    parser.add_argument(
        "--index_base",
        choices=["0", "1"],
        default="1",
        help="Index base for printed/written atom lists: 1 (CP2K style) or 0. Default: 1.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print layer sizes and FIXED_ATOMS list (do not write a Selective Dynamics POSCAR).",
    )
    parser.add_argument(
        "--n_fix_layers",
        type=int,
        default=3,
        help="Number of bottom z-layers to fix in --by_z_layers mode (default: 3).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Layer clustering tolerance (Å) for --by_z_layers. If omitted, auto-estimated.",
    )
    parser.add_argument(
        "--no_gap_cut",
        action="store_true",
        help="Disable largest-gap cut when unwrapping the periodic coordinate in --by_z_layers mode (not recommended).",
    )
    parser.add_argument(
        "--debug_layers",
        action="store_true",
        help="Print layer statistics in --by_z_layers mode.",
    )
    parser.add_argument(
        "--no_include_molecules",
        action="store_true",
        help="Disable whole-molecule inclusion when fixing bottom z-layers.",
    )
    parser.add_argument(
        "--molecule_elements",
        type=str,
        default="C,N,H",
        help="Comma-separated element symbols treated as organic molecules for whole-molecule inclusion (default: C,N,H).",
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        if args.input_file.endswith('.vasp') or args.input_file.endswith('.POSCAR'):
            args.output_file = args.input_file.replace('.vasp', '_relaxed.vasp').replace('.POSCAR', '_relaxed.POSCAR')
        else:
            args.output_file = args.input_file + '_relaxed'
    
    try:
        if args.by_z_layers:
            mol_elems = {e.strip() for e in str(args.molecule_elements).split(",") if e.strip()}
            print_fixed_atoms_by_z_layers(
                args.input_file,
                n_fix_layers=int(args.n_fix_layers),
                tol=args.tol,
                one_based_output=(str(args.index_base) == "1"),
                gap_cut=not bool(args.no_gap_cut),
                debug=bool(args.debug_layers),
                include_molecules=not bool(args.no_include_molecules),
                molecule_elements=mol_elems,
            )
            # In this mode, the user requested terminal printing only.
            return 0

        if args.by_layers:
            fixed_layers = [int(x.strip()) for x in str(args.fixed_layers).split(",") if x.strip()]
            add_selective_dynamics_by_layers(
                args.input_file,
                args.output_file,
                n_interfaces=int(args.n_interfaces),
                fixed_layers=fixed_layers,
                min_gap=float(args.min_gap),
                one_based_output=(str(args.index_base) == "1"),
                print_only=bool(args.print_only),
            )
        else:
            add_selective_dynamics(
                args.input_file,
                args.output_file,
                n_layers_per_side=args.n_layers,
                layer_thickness=args.layer_thickness,
                interface_method=args.interface_method
            )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

