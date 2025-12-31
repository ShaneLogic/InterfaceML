"""
InterfaceML: A Professional Toolkit for Heterojunction Modeling

This package provides comprehensive tools for building, analyzing, and optimizing
heterostructure interfaces for DFT calculations, with a focus on perovskite-based
solar cells and related materials.

Main modules:
- core: Core functionality for structure manipulation and interface building
- cli: Command-line interface tools
- web: Web application for interactive modeling
- utils: Utility functions and helpers

Author: Interface Modeling Lab
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Interface Modeling Lab"

# Import available core modules
try:
    from interfaceml.core import io, layering
    __all__ = ["io", "layering"]
except ImportError:
    __all__ = []
