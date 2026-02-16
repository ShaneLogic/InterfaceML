"""
Performance utilities for InterfaceML.

This module provides helper functions for profiling, benchmarking, and
monitoring performance of structure operations.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def timer(operation_name: str = "Operation", verbose: bool = True):
    """
    Context manager for timing code blocks.
    
    Parameters
    ----------
    operation_name
        Descriptive name for the operation being timed.
    verbose
        If True, print timing information.
        
    Yields
    ------
    elapsed
        Dict with 'elapsed' key containing execution time in seconds.
        
    Examples
    --------
    >>> from interfaceml.utils.performance import timer
    >>> with timer("Structure loading"):
    ...     structure = load_structure("big_file.vasp")
    Structure loading: 2.34 seconds
    """
    result = {'elapsed': 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result['elapsed'] = elapsed
        if verbose:
            logger.info("%s: %.2f seconds", operation_name, elapsed)


def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark function execution time.
    
    Parameters
    ----------
    func
        Function to benchmark.
        
    Returns
    -------
    wrapper
        Wrapped function that prints execution time.
        
    Examples
    --------
    >>> from interfaceml.utils.performance import benchmark
    >>> @benchmark
    ... def slow_function():
    ...     time.sleep(1)
    >>> slow_function()
    slow_function: 1.00 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s: %.2f seconds", func.__name__, elapsed)
        return result
    return wrapper


def estimate_memory_usage(structure: Any) -> Dict[str, float]:
    """
    Estimate memory usage of a pymatgen Structure.
    
    Parameters
    ----------
    structure
        A pymatgen Structure object.
        
    Returns
    -------
    memory_info
        Dictionary with memory usage estimates in MB:
        - 'coordinates': Memory for atomic coordinates
        - 'species': Memory for species information
        - 'lattice': Memory for lattice parameters
        - 'total': Total estimated memory
        
    Notes
    -----
    This is an approximation based on typical numpy array sizes.
    Actual memory usage may vary due to Python object overhead.
    """
    n_atoms = len(structure)
    
    # Coordinates: 3 * n_atoms * 8 bytes (float64)
    coords_mb = (3 * n_atoms * 8) / (1024 ** 2)
    
    # Species: approximately 50 bytes per site (string overhead)
    species_mb = (n_atoms * 50) / (1024 ** 2)
    
    # Lattice: ~200 bytes (3x3 matrix + metadata)
    lattice_mb = 200 / (1024 ** 2)
    
    total_mb = coords_mb + species_mb + lattice_mb
    
    return {
        'coordinates': coords_mb,
        'species': species_mb,
        'lattice': lattice_mb,
        'total': total_mb,
    }


def suggest_batch_size(
    n_structures: int,
    avg_atoms_per_structure: int,
    available_memory_mb: float = 1000.0,
) -> int:
    """
    Suggest optimal batch size for processing multiple structures.
    
    Parameters
    ----------
    n_structures
        Total number of structures to process.
    avg_atoms_per_structure
        Average number of atoms per structure.
    available_memory_mb
        Available memory in megabytes (default: 1000 MB).
        
    Returns
    -------
    batch_size
        Recommended number of structures to process at once.
        
    Examples
    --------
    >>> from interfaceml.utils.performance import suggest_batch_size
    >>> batch = suggest_batch_size(n_structures=100, avg_atoms_per_structure=500)
    >>> print(f"Process in batches of {batch}")
    """
    # Estimate memory per structure (rough approximation)
    mb_per_structure = (avg_atoms_per_structure * 3 * 8) / (1024 ** 2) + 0.1
    
    # Calculate batch size with 20% safety margin
    batch_size = int((available_memory_mb * 0.8) / mb_per_structure)
    
    # Clamp to reasonable range [1, n_structures]
    return max(1, min(batch_size, n_structures))


def analyze_structure_complexity(structure: Any) -> Dict[str, Any]:
    """
    Analyze structure complexity for performance estimation.
    
    Parameters
    ----------
    structure
        A pymatgen Structure object.
        
    Returns
    -------
    complexity_info
        Dictionary with complexity metrics:
        - 'n_atoms': Number of atoms
        - 'n_species': Number of unique species
        - 'cell_volume': Unit cell volume (Ų)
        - 'density': Atomic density (atoms/Ų)
        - 'complexity_score': Overall complexity (1-10)
        - 'recommended_approach': Suggested algorithm choice
        
    Examples
    --------
    >>> from interfaceml.core import io
    >>> from interfaceml.utils.performance import analyze_structure_complexity
    >>> structure = io.load_structure("structure.vasp")
    >>> info = analyze_structure_complexity(structure)
    >>> print(f"Complexity score: {info['complexity_score']}/10")
    """
    n_atoms = len(structure)
    n_species = len({site.species_string for site in structure})
    volume = structure.lattice.volume
    density = n_atoms / volume if volume > 0 else 0
    
    # Calculate complexity score (1-10)
    # Based on: number of atoms, species diversity, and density
    atom_score = min(10, n_atoms / 100)  # Max at 1000 atoms
    species_score = min(3, n_species / 2)  # Max at 6 species
    density_score = min(2, density / 0.05)  # Max at 0.1 atoms/Ų
    
    complexity_score = atom_score + species_score + density_score
    
    # Recommend approach based on complexity
    if complexity_score < 3:
        approach = "simple"
    elif complexity_score < 7:
        approach = "standard"
    else:
        approach = "optimized"
    
    return {
        'n_atoms': n_atoms,
        'n_species': n_species,
        'cell_volume': volume,
        'density': density,
        'complexity_score': round(complexity_score, 1),
        'recommended_approach': approach,
    }
