"""
your_module_name.py

Brief description of what this module does (one line).

This module provides functionality for [specific task/domain], including:
- Main feature 1
- Main feature 2
- Main feature 3

Key algorithms:
- Algorithm A: Does X
- Algorithm B: Does Y

Author: Your Name
Date: YYYY-MM-DD
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Set, Union
import numpy as np
from pymatgen.core import Structure, Lattice

# Import from other InterfaceML modules
from interfaceml.core.io import load_structure, get_element_symbols
from interfaceml.core.layering import interface_normal_unit
from interfaceml.utils.geometry import angle_between, compute_distance


def main_function(
    structure: Structure,
    *,
    parameter1: float,
    parameter2: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Brief one-line description of what this function does.

    More detailed description here. Explain the algorithm, inputs,
    outputs, and any important assumptions or limitations.

    Parameters
    ----------
    structure
        Input pymatgen Structure object to process.
    parameter1
        Description of parameter1. Units if applicable (e.g., Angstroms, degrees).
    parameter2
        Optional parameter. If None, uses automatic detection.
        Valid options: "auto", "manual", "custom".
    verbose
        If True, print detailed progress information.

    Returns
    -------
    result_indices
        List of 0-based atom indices that satisfy the condition.
        Sorted from lowest to highest.
    metadata
        Dictionary containing analysis results:
        - 'metric1': float, description
        - 'metric2': float, description
        - 'count': int, number of results

    Raises
    ------
    ValueError
        If structure is empty or invalid.
        If parameter1 is out of valid range [min, max].
    RuntimeError
        If algorithm fails to converge.

    Examples
    --------
    >>> from interfaceml.core import io, your_module
    >>> struct = io.load_structure("structure.cif")
    >>> indices, meta = your_module.main_function(struct, parameter1=2.5)
    >>> print(f"Found {len(indices)} results")
    >>> print(f"Average metric: {meta['metric1']:.3f}")

    Notes
    -----
    - This function assumes periodic boundary conditions
    - Performance is O(N log N) where N is number of atoms
    - For very large structures (>10000 atoms), consider using chunking

    See Also
    --------
    helper_function : Related utility function
    other_module.related_function : Similar functionality in another module
    """
    # ─────────────────────────────────────────────────────────────
    # 1. Input Validation
    # ─────────────────────────────────────────────────────────────
    if len(structure) == 0:
        raise ValueError("Structure cannot be empty")
    
    if parameter1 < 0 or parameter1 > 100:
        raise ValueError(f"parameter1 must be in [0, 100], got {parameter1}")
    
    if verbose:
        print(f"Processing structure: {structure.composition.formula}")
        print(f"Number of atoms: {len(structure)}")
    
    # ─────────────────────────────────────────────────────────────
    # 2. Initialization
    # ─────────────────────────────────────────────────────────────
    result_indices: List[int] = []
    metadata: Dict[str, float] = {}
    
    # Extract structure information
    coords = np.array(structure.cart_coords, dtype=float)
    symbols = get_element_symbols(structure)
    
    # ─────────────────────────────────────────────────────────────
    # 3. Main Algorithm
    # ─────────────────────────────────────────────────────────────
    
    # Example: Find atoms within distance threshold
    for i in range(len(structure)):
        # Your algorithm logic here
        # ...
        
        if _meets_criteria(coords[i], parameter1):
            result_indices.append(i)
    
    if verbose:
        print(f"Found {len(result_indices)} results")
    
    # ─────────────────────────────────────────────────────────────
    # 4. Compute Metadata
    # ─────────────────────────────────────────────────────────────
    if len(result_indices) > 0:
        metadata['count'] = len(result_indices)
        metadata['metric1'] = float(np.mean([coords[i, 2] for i in result_indices]))
        metadata['metric2'] = float(np.std([coords[i, 2] for i in result_indices]))
    else:
        metadata['count'] = 0
        metadata['metric1'] = 0.0
        metadata['metric2'] = 0.0
    
    # ─────────────────────────────────────────────────────────────
    # 5. Return Results
    # ─────────────────────────────────────────────────────────────
    return sorted(result_indices), metadata


def helper_function(
    data: np.ndarray,
    threshold: float = 1.0,
) -> bool:
    """
    Helper function for internal use.

    Parameters
    ----------
    data
        Input data array
    threshold
        Threshold value

    Returns
    -------
    bool
        True if condition is met
    """
    return float(np.mean(data)) > threshold


def _meets_criteria(position: np.ndarray, parameter: float) -> bool:
    """
    Private helper function (underscore prefix).

    This function is not exported in __all__ and is for internal use only.

    Parameters
    ----------
    position
        Cartesian position array [x, y, z]
    parameter
        Threshold parameter

    Returns
    -------
    bool
        True if position meets criteria
    """
    return float(position[2]) < parameter


def analyze_structure(
    structure: Structure,
    *,
    mode: str = "auto",
) -> Dict[str, Union[float, int, List[int]]]:
    """
    High-level analysis function.

    This provides a convenient interface for common analysis tasks.

    Parameters
    ----------
    structure
        Structure to analyze
    mode
        Analysis mode: "auto", "detailed", or "fast"

    Returns
    -------
    results
        Dictionary with comprehensive analysis results
    """
    results: Dict[str, Union[float, int, List[int]]] = {}
    
    # Call main function
    indices, metadata = main_function(structure, parameter1=2.5)
    
    results['n_results'] = len(indices)
    results['indices'] = indices
    results['metrics'] = metadata
    
    return results


# ═════════════════════════════════════════════════════════════════
# Public API - explicitly declare what can be imported
# ═════════════════════════════════════════════════════════════════
__all__ = [
    "main_function",
    "helper_function",
    "analyze_structure",
]


# ═════════════════════════════════════════════════════════════════
# Module-level constants (if needed)
# ═════════════════════════════════════════════════════════════════
DEFAULT_THRESHOLD = 2.5
VALID_MODES = ["auto", "manual", "custom"]


# ═════════════════════════════════════════════════════════════════
# Command-line interface (optional, for quick testing)
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Quick test
    if len(sys.argv) > 1:
        structure_file = sys.argv[1]
        if Path(structure_file).exists():
            struct = load_structure(structure_file)
            indices, meta = main_function(struct, parameter1=2.5, verbose=True)
            print(f"\nResults: {len(indices)} indices")
            print(f"Metadata: {meta}")
        else:
            print(f"Error: File not found: {structure_file}")
            sys.exit(1)
    else:
        print("Usage: python your_module_name.py <structure_file>")
        print("Example: python your_module_name.py structures/perovskites/fapbi3-1.cif")
