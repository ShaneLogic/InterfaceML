"""
Core functionality modules for InterfaceML.

This subpackage contains the fundamental building blocks for heterojunction modeling:
- io: Structure file I/O (VASP, CIF, XYZ formats)
- layering: Layer detection and manipulation
- termination: Surface termination analysis and selection (TODO)
- adsorbate: Adsorbate placement and stacking (TODO)
- interface: Interface matching and construction (TODO)
"""

# Only import available modules to avoid circular imports
__all__ = ["io", "layering"]
