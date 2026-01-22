"""
Structure I/O module for InterfaceML.

This module provides robust readers and writers for common atomistic file formats
used in DFT calculations, with particular support for:
- VASP POSCAR/CONTCAR
- CIF (Crystallographic Information File)
- CP2K-style XYZ trajectories with cell vectors

All functions return pymatgen Structure objects for downstream processing.

Performance Notes
-----------------
- Structure loading is cached to avoid redundant disk I/O and parsing
- Use load_structure() for automatic format detection with caching
- Direct file format readers (read_cp2k_xyz_last_frame) bypass cache
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

# Regex pattern for CP2K cell vectors: Tv_1: ax ay az Tv_2: bx by bz Tv_3: cx cy cz
_TV_PATTERN = re.compile(
    r"Tv_1:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
    r"Tv_2:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
    r"Tv_3:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


def read_cp2k_xyz_last_frame(
    filepath: Path | str,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Read the last frame from a CP2K-style XYZ trajectory.

    CP2K XYZ format:
        <number of atoms>
        Tv_1: ax ay az Tv_2: bx by bz Tv_3: cx cy cz  (optional cell vectors)
        Element x y z
        ...

    Parameters
    ----------
    filepath
        Path to the XYZ file.

    Returns
    -------
    symbols
        List of element symbols (length N).
    coords
        Cartesian coordinates, shape (N, 3) in Angstroms.
    cell
        Lattice matrix, shape (3, 3) where rows are [a, b, c] vectors in Angstroms,
        or None if Tv_ vectors are not present in the comment line.

    Raises
    ------
    ValueError
        If the file is malformed or contains no valid frames.
    """
    path = Path(filepath)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file too short (< 3 lines): {path}")

    i = 0
    last_block: Optional[Tuple[int, int]] = None
    last_cell: Optional[np.ndarray] = None

    # Scan for all frames and keep track of the last one
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            natoms = int(lines[i].strip().split()[0])
        except (ValueError, IndexError):
            break
        if i + 1 + natoms >= len(lines):
            break

        comment_line = lines[i + 1].strip()
        if (match := _TV_PATTERN.search(comment_line)):
            values = [float(x) for x in match.groups()]
            last_cell = np.array(values, dtype=float).reshape(3, 3)

        last_block = (i, i + 2 + natoms)
        i = i + 2 + natoms

    if last_block is None:
        raise ValueError(f"Could not parse any valid XYZ frame from: {path}")

    start, _ = last_block
    natoms = int(lines[start].strip().split()[0])
    symbols: List[str] = []
    coords: List[List[float]] = []

    for line in lines[start + 2 : start + 2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(symbols) != natoms:
        raise ValueError(
            f"Expected {natoms} atoms but parsed {len(symbols)} from: {path}"
        )

    return symbols, np.array(coords, dtype=float), last_cell


def load_structure(
    filepath: Path | str,
    *,
    fallback_box_size: Tuple[float, float, float] = (20.0, 20.0, 30.0),
    fallback_padding: float = 10.0,
) -> Structure:
    """
    Load an atomistic structure from POSCAR, CIF, or CP2K XYZ format.

    This is the primary structure loader for InterfaceML. It automatically
    detects the file format and returns a pymatgen Structure object.

    For XYZ files without cell information (Tv_ vectors), a fallback
    orthorhombic box is created based on coordinate extents plus padding.

    Parameters
    ----------
    filepath
        Path to the structure file (.vasp, .poscar, .cif, or .xyz).
    fallback_box_size
        Minimum box dimensions (a, b, c) in Angstroms for XYZ files
        without cell vectors.
    fallback_padding
        Padding added to each dimension when inferring box from coordinates.

    Returns
    -------
    structure
        A pymatgen Structure object.

    Raises
    ------
    ValueError
        If the file format is unsupported or the file is malformed.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        symbols, coords, cell = read_cp2k_xyz_last_frame(path)

        if cell is None:
            # Infer a fallback orthorhombic box from coordinate extents
            # This ensures molecules aren't clipped at periodic boundaries
            coord_min = coords.min(axis=0)
            coord_max = coords.max(axis=0)
            span = coord_max - coord_min
            # Use max of computed size + padding or minimum fallback size
            a = max(span[0] + fallback_padding, fallback_box_size[0])
            b = max(span[1] + fallback_padding, fallback_box_size[1])
            c = max(span[2] + fallback_padding, fallback_box_size[2])
            lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        else:
            lattice = Lattice(cell)

        return Structure(
            lattice,
            symbols,
            coords,
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

    # For VASP and CIF files, use cached loader for better performance
    return _load_and_cache_structure(str(path))


def write_poscar(
    structure: Structure,
    filepath: Path | str,
    *,
    selective_dynamics: Optional[List[Tuple[bool, bool, bool]]] = None,
    comment: Optional[str] = None,
) -> None:
    """
    Write a pymatgen Structure to a VASP POSCAR file.

    Parameters
    ----------
    structure
        The structure to write.
    filepath
        Output path for the POSCAR file.
    selective_dynamics
        Optional list of (Tx, Ty, Tz) tuples for each atom, where T/F
        indicates whether that degree of freedom is relaxed or fixed.
        If provided, the POSCAR will include a Selective Dynamics section.
    comment
        Optional comment line (first line of POSCAR). Defaults to the
        structure's composition formula.
    """
    # Some external tools (and even some naive writers) may produce POSCAR headers
    # that repeat the same element symbol many times (e.g. "H C H N ..."), with a
    # matching counts line that sums to the total atom count. While the total atom
    # count is correct, this is non-standard and can confuse downstream tooling.
    #
    # To keep outputs robust and VASP-friendly, we always write a canonical header:
    #   - element symbols are unique and ordered by first appearance
    #   - counts are aggregated per element
    #   - coordinate lines are grouped to match the counts

    symbols = get_element_symbols(structure)
    if len(symbols) != len(structure):
        raise ValueError(
            f"Structure site count mismatch: {len(structure)} sites but {len(symbols)} symbols"
        )

    # Stable unique ordering by first appearance
    seen = set()
    ordered_unique: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered_unique.append(s)

    grouped_indices: List[int] = []
    for s in ordered_unique:
        grouped_indices.extend([i for i, sym in enumerate(symbols) if sym == s])

    grouped_species = [symbols[i] for i in grouped_indices]
    grouped_frac = [structure[i].frac_coords for i in grouped_indices]

    grouped_sd = None
    if selective_dynamics is not None:
        if len(selective_dynamics) != len(structure):
            raise ValueError(
                "selective_dynamics length must match number of sites in structure"
            )
        grouped_sd = [selective_dynamics[i] for i in grouped_indices]

    grouped_struct = Structure(
        structure.lattice,
        grouped_species,
        grouped_frac,
        coords_are_cartesian=False,
        to_unit_cell=False,
    )

    poscar = Poscar(grouped_struct, comment=comment, selective_dynamics=grouped_sd)
    poscar.write_file(str(filepath))


def get_element_symbols(structure: Structure) -> List[str]:
    """
    Extract plain element symbols for each site in a structure.

    Parameters
    ----------
    structure
        A pymatgen Structure.

    Returns
    -------
    symbols
        List of element symbols (length = number of sites).
        
    Notes
    -----
    This is a frequently-called utility function. For repeated calls on
    the same structure, consider caching the result at the call site.
    """
    return [str(site.specie) for site in structure.sites]


@lru_cache(maxsize=32)
def _load_and_cache_structure(filepath_str: str) -> Structure:
    """
    Internal cached structure loader.
    
    Parameters
    ----------
    filepath_str
        String path to structure file.
        
    Returns
    -------
    structure
        Loaded pymatgen Structure.
        
    Notes
    -----
    Uses LRU cache to store up to 32 recently loaded structures.
    Cache is based on file path string, so modifications to files
    won't be automatically detected.
    """
    return Structure.from_file(filepath_str)
