"""
Utility functions for InterfaceML.

This module provides helper functions for:
- Geometry calculations and vector operations
- Performance monitoring and profiling
- Input validation and error checking
"""

from interfaceml.utils.geometry import (
    angle_between,
    normalize_vector,
    compute_distance,
)

# Optional utilities (may not be available in all environments)
try:
    from interfaceml.utils import performance
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

try:
    from interfaceml.utils import validation
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

__all__ = [
    "angle_between",
    "normalize_vector",
    "compute_distance",
]

if PERFORMANCE_AVAILABLE:
    __all__.extend(["performance"])

if VALIDATION_AVAILABLE:
    __all__.extend(["validation"])

