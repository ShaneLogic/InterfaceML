#!/usr/bin/env python3
"""
CLI wrapper for delta_density_cube functionality.

This script maintains backward compatibility with the original build_heterojunctions/delta_density_cube.py.
"""

import sys
from pathlib import Path

# Add build_heterojunctions to path for backward compatibility
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build_heterojunctions"))

# Import and run the original script
from delta_density_cube import main as original_main


def main():
    """Entry point that delegates to the original implementation."""
    original_main()


if __name__ == "__main__":
    main()
