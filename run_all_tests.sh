#!/bin/bash
#
# InterfaceML Automated Test Suite
# Run all tests to verify package functionality
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        InterfaceML Automated Test Suite                   ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create test output directory
TEST_OUTPUT_DIR="test_output"
mkdir -p "$TEST_OUTPUT_DIR"

# Test counter
PASSED=0
FAILED=0
TOTAL=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$TOTAL] $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Command: $test_command"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Test 1: Core module imports
run_test "Core Module Imports" \
    "python tests/test_basic.py"

# Test 2: I/O module with CIF file
run_test "I/O Module - CIF Loading" \
    "python -c \"from interfaceml.core import io; s = io.load_structure('structures/perovskites/fapbi3-1.cif'); print(f'Loaded {len(s)} atoms')\""

# Test 3: Layering module
run_test "Layering Module - Layer Detection" \
    "python -c \"from interfaceml.core import io, layering; s = io.load_structure('structures/perovskites/fapbi3-1.cif'); layers, tol = layering.split_layers_by_z(s); print(f'{len(layers)} layers')\""

# Test 4: Geometry utilities
run_test "Geometry Utils - Vector Operations" \
    "python -c \"from interfaceml.utils.geometry import angle_between, normalize_vector; import numpy as np; v1 = np.array([1,0,0]); v2 = np.array([0,1,0]); angle = angle_between(v1,v2); assert abs(angle - 90) < 0.1\""

# Test 5: Interface builder - adsorbate mode (if files exist)
if [ -f "structures/perovskites/fapbi3-1.cif" ] && [ -f "structures/etl/C60-Ih.cif" ]; then
    run_test "CLI - Interface Builder (Adsorbate Mode)" \
        "python build_heterojunctions/interface_builder.py --adsorbate_mode --base structures/perovskites/fapbi3-1.cif --adsorbate structures/etl/C60-Ih.cif --termination PbI --distance 2.5 --output $TEST_OUTPUT_DIR/test_interface.vasp"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: Interface Builder (missing input files)"
    TOTAL=$((TOTAL + 1))
fi

# Test 6: Fix layers tool (requires output from test 5)
if [ -f "$TEST_OUTPUT_DIR/test_interface.vasp" ]; then
    run_test "CLI - Fix Layers Tool" \
        "python build_heterojunctions/fix_interface_layers.py --by_z_layers --input $TEST_OUTPUT_DIR/test_interface.vasp --n_fix_layers 3 --output $TEST_OUTPUT_DIR/test_interface_fixed.vasp"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: Fix Layers Tool (missing input from previous test)"
    TOTAL=$((TOTAL + 1))
fi

# Test 7: Verify output has selective dynamics
if [ -f "$TEST_OUTPUT_DIR/test_interface_fixed.vasp" ]; then
    run_test "Output Verification - Selective Dynamics" \
        "grep -q 'Selective' $TEST_OUTPUT_DIR/test_interface_fixed.vasp"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC}: Output Verification (missing output file)"
    TOTAL=$((TOTAL + 1))
fi

# Test 8: Python API usage
run_test "Python API - End-to-End Workflow" \
    "python -c \"
from interfaceml.core import io, layering
s = io.load_structure('structures/perovskites/fapbi3-1.cif')
layers, tol = layering.split_layers_by_z(s)
fixed = []
for layer in layers[:2]:
    fixed.extend(layer)
fixed = layering.include_whole_molecules(s, fixed)
print(f'Fixed {len(fixed)} atoms')
\""

# Test 9: Web app module import
run_test "Web App Module Import" \
    "python -c \"from interfaceml.web import app; print('Web app loaded')\""

# Test 10: CLI wrapper scripts
run_test "CLI Wrapper - Import Check" \
    "python -c \"from interfaceml.cli import fix_layers, build_interface; print('CLI wrappers loaded')\""

# Print summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Test Summary                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
printf "║  Total Tests:  %-40s  ║\n" "$TOTAL"
printf "║  ${GREEN}Passed:       %-40s${NC}  ║\n" "$PASSED"
if [ $FAILED -gt 0 ]; then
    printf "║  ${RED}Failed:       %-40s${NC}  ║\n" "$FAILED"
else
    printf "║  Failed:       %-40s  ║\n" "$FAILED"
fi
echo "╚════════════════════════════════════════════════════════════╝"

# Output directory info
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo ""
    echo "Test outputs saved to: $TEST_OUTPUT_DIR/"
    ls -lh "$TEST_OUTPUT_DIR/" 2>/dev/null | tail -n +2 | head -5
fi

# Exit with appropriate code
if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed successfully!${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
    echo ""
    exit 1
fi
