"""
DOS/PDOS parsing and processing helpers for the InterfaceML web app.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np


def read_cp2k_dos_like(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, float | None, str, str]:
    """
    Read CP2K .dos/.pdos files and extract Fermi energy when available.

    Returns
    -------
    energy
        Energy axis in eV.
    data
        Numeric columns after energy.
    ef
        Fermi energy in eV if present in header, otherwise None.
    unit
        Original energy unit detected in the header ("a.u." or "eV").
    header_text
        Full header text for label extraction.
    """
    ef = None
    header_lines: list[str] = []

    with open(filepath, "r", errors="ignore") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                header_lines.append(line.strip())
            else:
                break

    header_text = "\n".join(header_lines)
    patterns = [
        r"Fermi\s*energy.*?([-\d\.Ee+]+)",
        r"E\s*\(\s*Fermi\s*\)\s*=?\s*([-\d\.Ee+]+)",
        r"Efermi\s*=?\s*([-\d\.Ee+]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, header_text, flags=re.IGNORECASE)
        if match:
            try:
                ef = float(match.group(1))
                break
            except ValueError:
                continue

    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    unit = "eV"
    if "a.u." in header_text or "au" in header_text.lower():
        unit = "a.u."

    # CP2K PDOS format often includes an index column followed by energy and occupation.
    has_index_column = False
    if data.shape[1] >= 3:
        idx_col = data[:, 0]
        if np.allclose(idx_col, np.round(idx_col), atol=1e-6):
            has_index_column = True

    if has_index_column:
        energy = data[:, 1]
        rest = data[:, 3:] if data.shape[1] > 3 else np.zeros((len(energy), 0))
    else:
        energy = data[:, 0]
        rest = data[:, 1:] if data.shape[1] > 1 else np.zeros((len(energy), 0))

    if unit == "a.u.":
        hartree_to_ev = 27.211386245988
        energy = energy * hartree_to_ev
        if ef is not None:
            ef = ef * hartree_to_ev

    return energy, rest, ef, unit, header_text


def extract_pdos_meta(header_text: str, fallback: str) -> tuple[str, int | None]:
    """Extract PDOS label and atom count from CP2K header text."""
    match = re.search(
        r"list\s+(\d+)\s+of\s+(\d+)\s+atoms",
        header_text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"List {match.group(1)} ({match.group(2)} atoms)", int(match.group(2))
    return fallback, None


def gaussian_broaden(
    energies: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply Gaussian broadening to discrete DOS data."""
    if sigma <= 0:
        return np.interp(grid, energies, weights, left=0.0, right=0.0)

    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    broadened = np.zeros_like(grid, dtype=float)
    for energy, weight in zip(energies, weights):
        broadened += weight * np.exp(-0.5 * ((grid - energy) / sigma) ** 2)
    return coeff * broadened


def merge_overlapping_layers(
    layers: list[list[int]],
    heights: np.ndarray,
) -> list[list[int]]:
    """Merge overlapping layers and return sorted layers by mean height."""
    pending = [set(layer) for layer in layers if layer]
    merged: list[set[int]] = []

    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            for other in list(pending):
                if current & other:
                    current.update(other)
                    pending.remove(other)
                    changed = True
        merged.append(current)

    merged_lists = [sorted(layer) for layer in merged]
    merged_lists.sort(
        key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0
    )
    return merged_lists
