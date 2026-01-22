"""
_utils_structures.py

Shared structure IO utilities for CLI scripts in this folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pymatgen.core import Lattice, Structure

try:
    # When executed as a module: `python -m build_heterojunctions.<script>`
    from ._utils_xyz import read_cp2k_xyz_last_frame
except ImportError:
    # When executed as a script: `python build_heterojunctions/<script>.py`
    from _utils_xyz import read_cp2k_xyz_last_frame


def load_structure_any(path_str: str) -> Structure:
    """
    Load a structure from POSCAR/CIF/XYZ.

    For XYZ, this function expects a CP2K-style comment line containing Tv_1/Tv_2/Tv_3.
    If not found, it falls back to a large orthorhombic box based on coordinate extents.
    """
    path = Path(path_str)
    if path.suffix.lower() == ".xyz":
        symbols, coords, cell = read_cp2k_xyz_last_frame(path)
        if cell is None:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            span = maxs - mins
            a = float(max(span[0] + 10.0, 20.0))
            b = float(max(span[1] + 10.0, 20.0))
            c = float(max(span[2] + 10.0, 30.0))
            lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
        else:
            lattice = Lattice(cell)
        return Structure(lattice, symbols, coords, coords_are_cartesian=True, to_unit_cell=False)
    return Structure.from_file(path_str)


def element_symbols(structure: Structure) -> list[str]:
    """Return plain element symbols for each site."""
    return [str(sp) for sp in structure.species]

