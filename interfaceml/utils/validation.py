"""
Input validation utilities for InterfaceML.

This module provides robust validation functions to ensure data integrity
and provide helpful error messages for invalid inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_structure(structure: Any) -> None:
    """
    Validate that a structure object is valid.
    
    Parameters
    ----------
    structure
        A pymatgen Structure object to validate.
        
    Raises
    ------
    ValidationError
        If the structure is invalid.
        
    Examples
    --------
    >>> from interfaceml.utils.validation import validate_structure
    >>> validate_structure(my_structure)  # Raises if invalid
    """
    if structure is None:
        raise ValidationError("Structure is None")
    
    if not hasattr(structure, 'lattice'):
        raise ValidationError("Structure missing 'lattice' attribute")
    
    if not hasattr(structure, 'sites'):
        raise ValidationError("Structure missing 'sites' attribute")
    
    if len(structure) == 0:
        raise ValidationError("Structure contains no atoms")
    
    # Check for reasonable lattice parameters
    lattice = structure.lattice
    if lattice.volume < 1e-6:
        raise ValidationError(f"Lattice volume too small: {lattice.volume:.2e} Ų")
    
    if lattice.volume > 1e8:
        raise ValidationError(f"Lattice volume too large: {lattice.volume:.2e} Ų")


def validate_filepath(
    filepath: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """
    Validate and normalize a file path.
    
    Parameters
    ----------
    filepath
        Path to validate.
    must_exist
        If True, require that the file exists.
    allowed_extensions
        List of allowed file extensions (e.g., ['.cif', '.vasp']).
        If None, any extension is allowed.
        
    Returns
    -------
    path
        Validated Path object.
        
    Raises
    ------
    ValidationError
        If the path is invalid.
        
    Examples
    --------
    >>> from interfaceml.utils.validation import validate_filepath
    >>> path = validate_filepath("structure.cif", allowed_extensions=['.cif', '.vasp'])
    """
    if not filepath:
        raise ValidationError("File path is empty")
    
    path = Path(filepath)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")
    
    if allowed_extensions is not None:
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            allowed = ', '.join(allowed_extensions)
            raise ValidationError(
                f"Invalid file extension '{ext}'. Allowed: {allowed}"
            )
    
    return path


def validate_positive_number(
    value: float,
    name: str = "value",
    allow_zero: bool = False,
) -> float:
    """
    Validate that a number is positive.
    
    Parameters
    ----------
    value
        Number to validate.
    name
        Name of the parameter (for error messages).
    allow_zero
        If True, zero is considered valid.
        
    Returns
    -------
    value
        The validated value.
        
    Raises
    ------
    ValidationError
        If the value is invalid.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    
    return value


def validate_integer_range(
    value: int,
    name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """
    Validate that an integer is within a specified range.
    
    Parameters
    ----------
    value
        Integer to validate.
    name
        Name of the parameter (for error messages).
    min_value
        Minimum allowed value (inclusive).
    max_value
        Maximum allowed value (inclusive).
        
    Returns
    -------
    value
        The validated integer.
        
    Raises
    ------
    ValidationError
        If the value is invalid.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be an integer, got {type(value)}")
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_miller_indices(indices: List[int], name: str = "Miller indices") -> List[int]:
    """
    Validate Miller indices.
    
    Parameters
    ----------
    indices
        List of three integers representing Miller indices.
    name
        Name of the parameter (for error messages).
        
    Returns
    -------
    indices
        The validated Miller indices.
        
    Raises
    ------
    ValidationError
        If the indices are invalid.
        
    Examples
    --------
    >>> from interfaceml.utils.validation import validate_miller_indices
    >>> miller = validate_miller_indices([0, 0, 1])
    """
    if not isinstance(indices, (list, tuple)):
        raise ValidationError(f"{name} must be a list or tuple")
    
    if len(indices) != 3:
        raise ValidationError(f"{name} must have exactly 3 elements, got {len(indices)}")
    
    try:
        indices = [int(i) for i in indices]
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must contain only integers")
    
    if all(i == 0 for i in indices):
        raise ValidationError(f"{name} cannot be [0, 0, 0]")
    
    return indices


def validate_coordinates(
    coords: np.ndarray,
    expected_shape: Optional[tuple] = None,
) -> np.ndarray:
    """
    Validate coordinate array.
    
    Parameters
    ----------
    coords
        Coordinate array to validate.
    expected_shape
        Expected shape of the array (e.g., (N, 3) for 3D coordinates).
        If None, only basic checks are performed.
        
    Returns
    -------
    coords
        Validated coordinate array.
        
    Raises
    ------
    ValidationError
        If the coordinates are invalid.
    """
    coords = np.asarray(coords, dtype=float)
    
    if coords.size == 0:
        raise ValidationError("Coordinate array is empty")
    
    if not np.all(np.isfinite(coords)):
        raise ValidationError("Coordinate array contains non-finite values")
    
    if expected_shape is not None:
        if coords.shape != expected_shape:
            raise ValidationError(
                f"Expected shape {expected_shape}, got {coords.shape}"
            )
    
    return coords


def validate_distance(
    distance: float,
    min_distance: float = 0.5,
    max_distance: float = 50.0,
    name: str = "distance",
) -> float:
    """
    Validate a distance value is physically reasonable.
    
    Parameters
    ----------
    distance
        Distance in Angstroms.
    min_distance
        Minimum reasonable distance (default: 0.5 Å).
    max_distance
        Maximum reasonable distance (default: 50 Å).
    name
        Name of the parameter (for error messages).
        
    Returns
    -------
    distance
        Validated distance.
        
    Raises
    ------
    ValidationError
        If the distance is invalid.
    """
    distance = validate_positive_number(distance, name=name)
    
    if distance < min_distance:
        raise ValidationError(
            f"{name} too small: {distance:.3f} Å (min: {min_distance} Å)"
        )
    
    if distance > max_distance:
        raise ValidationError(
            f"{name} too large: {distance:.3f} Å (max: {max_distance} Å)"
        )
    
    return distance


def validate_termination(
    termination: str,
    allowed_terminations: Optional[List[str]] = None,
) -> str:
    """
    Validate surface termination string.
    
    Parameters
    ----------
    termination
        Termination string (e.g., 'PbI', 'FAI', 'MAI').
    allowed_terminations
        List of allowed termination strings. If None, only basic
        validation is performed.
        
    Returns
    -------
    termination
        Validated termination string.
        
    Raises
    ------
    ValidationError
        If the termination is invalid.
    """
    if not termination or not isinstance(termination, str):
        raise ValidationError("Termination must be a non-empty string")
    
    termination = termination.strip()
    
    if not termination:
        raise ValidationError("Termination string is empty after stripping")
    
    if allowed_terminations is not None:
        if termination not in allowed_terminations:
            allowed = ', '.join(allowed_terminations)
            raise ValidationError(
                f"Invalid termination '{termination}'. Allowed: {allowed}"
            )
    
    return termination


def validate_composition_weight(weight: float) -> float:
    """
    Validate composition weight parameter (must be in [0, 1]).
    
    Parameters
    ----------
    weight
        Weight value to validate.
        
    Returns
    -------
    weight
        Validated weight.
        
    Raises
    ------
    ValidationError
        If the weight is not in [0, 1].
    """
    weight = float(weight)
    
    if not 0.0 <= weight <= 1.0:
        raise ValidationError(
            f"Composition weight must be in [0, 1], got {weight}"
        )
    
    return weight
