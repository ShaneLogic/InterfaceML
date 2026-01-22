#!/usr/bin/env python3
"""
your_tool_name.py

Brief description of what this tool does.

This command-line tool provides functionality for [specific task],
allowing users to [what users can achieve].

Features:
- Feature 1
- Feature 2
- Feature 3

Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import core functionality
try:
    from interfaceml.core import io, your_module
    from interfaceml.utils.geometry import compute_distance
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False
    print("Warning: InterfaceML core modules not found.")
    print("Make sure you're in the project directory or have installed the package.")


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure command-line argument parser.

    Returns
    -------
    parser
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[1],  # Use second paragraph of docstring
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --input structure.cif --output result.vasp
  
  # With custom parameters
  %(prog)s --input file.vasp --param1 2.5 --param2 auto --verbose
  
  # Multiple inputs
  %(prog)s --input file1.cif file2.cif --output combined.vasp
  
  # Debug mode
  %(prog)s --input structure.xyz --param1 3.0 --debug --verbose

For more information, see the documentation at:
https://github.com/yourusername/InterfaceML
        """
    )
    
    # ─────────────────────────────────────────────────────────────
    # Required Arguments
    # ─────────────────────────────────────────────────────────────
    required = parser.add_argument_group('required arguments')
    
    required.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input structure file (CIF, VASP, or XYZ format)"
    )
    
    required.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (will be created if doesn't exist)"
    )
    
    # ─────────────────────────────────────────────────────────────
    # Core Parameters
    # ─────────────────────────────────────────────────────────────
    params = parser.add_argument_group('parameters')
    
    params.add_argument(
        "--param1",
        type=float,
        default=2.5,
        metavar="VALUE",
        help="Description of parameter 1 in Angstroms (default: 2.5)"
    )
    
    params.add_argument(
        "--param2",
        type=str,
        choices=["auto", "manual", "custom"],
        default="auto",
        help="Description of parameter 2 (default: auto)"
    )
    
    params.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        metavar="VALUE",
        help="Threshold value for filtering (default: 1.0)"
    )
    
    # ─────────────────────────────────────────────────────────────
    # Optional Flags
    # ─────────────────────────────────────────────────────────────
    optional = parser.add_argument_group('optional arguments')
    
    optional.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )
    
    optional.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (show full error traceback)"
    )
    
    optional.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually processing"
    )
    
    optional.add_argument(
        "--format",
        type=str,
        choices=["vasp", "cif", "xyz"],
        help="Force output format (default: infer from extension)"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.

    Parameters
    ----------
    args
        Parsed arguments from argparse

    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return False
    
    # Check input file extension
    valid_extensions = {'.cif', '.vasp', '.poscar', '.xyz'}
    if input_path.suffix.lower() not in valid_extensions:
        print(f"Warning: Unexpected file extension: {input_path.suffix}")
        print(f"Supported: {', '.join(valid_extensions)}")
    
    # Check output directory exists or can be created
    output_path = Path(args.output)
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.verbose:
                print(f"Created output directory: {output_path.parent}")
        except Exception as e:
            print(f"Error: Cannot create output directory: {e}")
            return False
    
    # Validate parameters
    if args.param1 < 0:
        print(f"Error: param1 must be non-negative, got {args.param1}")
        return False
    
    return True


def print_banner(args: argparse.Namespace) -> None:
    """Print informative banner with parameters."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              Your Tool Name                              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Parameters:")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Param1:     {args.param1}")
    print(f"  Param2:     {args.param2}")
    print(f"  Threshold:  {args.threshold}")
    print()


def main() -> int:
    """
    Main entry point for the command-line tool.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Check core availability
    if not _CORE_AVAILABLE:
        print("\nError: InterfaceML core modules could not be imported.")
        print("Please ensure you're running from the project directory")
        print("or have installed the package with: pip install -e .")
        return 1
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    # Print banner if verbose
    if args.verbose:
        print_banner(args)
    
    # Dry run mode
    if args.dry_run:
        print("Dry run mode - showing what would be done:")
        print(f"  1. Load structure from: {args.input}")
        print(f"  2. Process with param1={args.param1}, param2={args.param2}")
        print(f"  3. Write output to: {args.output}")
        print("\nNo files were modified.")
        return 0
    
    # ═════════════════════════════════════════════════════════════
    # Main Processing
    # ═════════════════════════════════════════════════════════════
    try:
        # Step 1: Load structure
        if args.verbose:
            print("Step 1: Loading structure...")
        
        structure = io.load_structure(args.input)
        
        if args.verbose:
            print(f"  ✓ Loaded {len(structure)} atoms")
            print(f"  Composition: {structure.composition.formula}")
            print(f"  Lattice: a={structure.lattice.a:.3f} Å, "
                  f"b={structure.lattice.b:.3f} Å, c={structure.lattice.c:.3f} Å")
        
        # Step 2: Process with your module
        if args.verbose:
            print("\nStep 2: Processing structure...")
        
        result_indices, metadata = your_module.main_function(
            structure,
            parameter1=args.param1,
            parameter2=args.param2 if args.param2 != "auto" else None,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"  ✓ Found {len(result_indices)} results")
            print(f"  Metrics: {metadata}")
        
        # Step 3: Create output
        # (Modify structure, create selective dynamics, etc.)
        
        # Step 4: Write output
        if args.verbose:
            print(f"\nStep 3: Writing output to {args.output}...")
        
        io.write_poscar(structure, args.output)
        
        if args.verbose:
            print(f"  ✓ Output written successfully")
        
        # Print summary
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                    Success!                              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Results:        {len(result_indices)} items")
        print(f"  Output file:    {args.output}")
        print(f"  Processing time: [calculate if needed]")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 130
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        
        if args.debug:
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
        else:
            print("(Use --debug flag for full traceback)")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
