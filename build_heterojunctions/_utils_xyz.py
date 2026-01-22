"""
_utils_xyz.py

Shared XYZ parsing utilities used by the CLI tools in this folder.

Notes
-----
This module is intentionally dependency-light (standard library + numpy).
It focuses on CP2K-style XYZ trajectories where the comment line may contain
cell vectors as:

  Tv_1: ax ay az Tv_2: bx by bz Tv_3: cx cy cz
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

_TV_RE = re.compile(
    r"Tv_1:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
    r"Tv_2:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
    r"Tv_3:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
)


def read_cp2k_xyz_last_frame(path: Path) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Read the last frame of a CP2K-style XYZ file.

    Returns
    -------
    symbols
        Element symbols length N.
    coords
        Cartesian coordinates array shape (N, 3) in Å.
    cell
        Lattice matrix shape (3, 3) where rows correspond to a, b, c vectors (Å),
        or None if Tv_ vectors are not present.
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
    symbols: List[str] = []
    coords: List[List[float]] = []
    for ln in lines[start + 2 : start + 2 + nat]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(symbols) != nat:
        raise ValueError(f"Expected {nat} atoms but parsed {len(symbols)} atoms from: {path}")
    return symbols, np.array(coords, dtype=float), last_cell

