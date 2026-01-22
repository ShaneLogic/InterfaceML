#!/bin/bash
#
# Example Workflow: Building a FAPbI3/C60 Interface
# This script demonstrates a complete workflow from structure files to DFT-ready model
#

set -e  # Exit on error

echo "========================================="
echo "InterfaceML Example Workflow"
echo "Building FAPbI3/C60 Solar Cell Interface"
echo "========================================="
echo ""

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRUCTURES_DIR="$BASE_DIR/structures"
OUTPUT_DIR="$BASE_DIR/examples/output"
SCRIPTS_DIR="$BASE_DIR/build_heterojunctions"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Step 1: Building Interface Structure"
echo "-------------------------------------"
python "$SCRIPTS_DIR/interface_builder.py" \
    --adsorbate_mode \
    --base "$STRUCTURES_DIR/perovskites/fapbi3-1.cif" \
    --adsorbate "$STRUCTURES_DIR/etl/C60-Ih.cif" \
    --termination FAI \
    --distance 2.5 \
    --miller 0 0 1 \
    --supercell auto \
    --output fapbi3_c60_interface.vasp

if [ -f "fapbi3_c60_interface.vasp" ]; then
    echo "✓ Interface structure created successfully"
else
    echo "✗ Failed to create interface structure"
    exit 1
fi
echo ""

echo "Step 2: Adding Selective Dynamics (Fix Bottom Layers)"
echo "-------------------------------------------------------"
python "$SCRIPTS_DIR/fix_interface_layers.py" \
    --by_z_layers \
    --input fapbi3_c60_interface.vasp \
    --n_fix_layers 3 \
    --output fapbi3_c60_interface_fixed.vasp

if [ -f "fapbi3_c60_interface_fixed.vasp" ]; then
    echo "✓ Selective dynamics added successfully"
else
    echo "✗ Failed to add selective dynamics"
    exit 1
fi
echo ""

echo "Step 3: Structure Information"
echo "-----------------------------"
echo "Output files created in: $OUTPUT_DIR"
echo ""
echo "Files:"
echo "  - fapbi3_c60_interface.vasp        : Base interface structure"
echo "  - fapbi3_c60_interface_fixed.vasp  : With selective dynamics (DFT-ready)"
echo ""
echo "Next Steps:"
echo "  1. Visualize the structure: ase gui fapbi3_c60_interface_fixed.vasp"
echo "  2. Copy to your DFT calculation directory"
echo "  3. Prepare INCAR, KPOINTS, and POTCAR files"
echo "  4. Run your DFT calculation"
echo "  5. Use delta_density_cube.py to analyze charge transfer"
echo ""
echo "========================================="
echo "Workflow completed successfully! 🎉"
echo "========================================="
