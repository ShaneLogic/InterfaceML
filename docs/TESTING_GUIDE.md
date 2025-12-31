# InterfaceML Testing Guide

This guide provides step-by-step instructions for testing all components of the InterfaceML package.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Testing Core Modules](#testing-core-modules)
4. [Testing CLI Tools](#testing-cli-tools)
5. [Testing Web Interface](#testing-web-interface)
6. [Testing Python API](#testing-python-api)
7. [Integration Testing](#integration-testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- Python 3.8 or higher
- pip package manager
- Web browser (Chrome, Firefox, Safari, or Edge)

### Check Your Environment

```bash
# 1. Check Python version
python --version
# Should output: Python 3.8.x or higher

# 2. Check pip
pip --version

# 3. Navigate to project root
cd /path/to/InterfaceML

# 4. Verify directory structure
ls -la
# Should see: interfaceml/, build_heterojunctions/, structures/, etc.
```

---

## Environment Setup

### Step 1: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify pymatgen installation
python -c "import pymatgen; print(f'✓ Pymatgen {pymatgen.__version__} installed')"

# Verify Flask installation (for web interface)
python -c "import flask; print(f'✓ Flask {flask.__version__} installed')"

# Verify numpy
python -c "import numpy; print(f'✓ Numpy {numpy.__version__} installed')"
```

### Step 2: Install Package in Development Mode

```bash
# Install package in editable mode
pip install -e .

# Verify package installation
python -c "import interfaceml; print(f'✓ InterfaceML {interfaceml.__version__} installed')"
```

---

## Testing Core Modules

### Test 1: Basic Module Imports

```bash
# Run the basic test suite
python tests/test_basic.py
```

**Expected Output:**
```
============================================================
InterfaceML Basic Functionality Tests
============================================================
Testing imports...
✓ All core modules imported successfully

Testing geometry functions...
✓ Geometry functions working correctly

Testing I/O module...
✓ I/O module functions available

Testing layering module...
✓ Layering module functions available

============================================================
Test Results: 4/4 passed
============================================================
```

### Test 2: I/O Module (Structure Loading)

Create a test script `test_io_manual.py`:

```python
from interfaceml.core import io
from pathlib import Path

print("Testing I/O Module...")
print("-" * 50)

# Test 1: Load CIF file
try:
    cif_file = "structures/perovskites/fapbi3-1.cif"
    if Path(cif_file).exists():
        struct = io.load_structure(cif_file)
        print(f"✓ Loaded CIF: {struct.composition.formula}")
        print(f"  - Atoms: {len(struct)}")
        print(f"  - Lattice: {struct.lattice.abc}")
    else:
        print(f"⚠ File not found: {cif_file}")
except Exception as e:
    print(f"✗ CIF loading failed: {e}")

# Test 2: Load VASP file
try:
    vasp_file = "structures/heterojunctions/H6PbCI3N@TiO2.vasp"
    if Path(vasp_file).exists():
        struct = io.load_structure(vasp_file)
        print(f"✓ Loaded VASP: {struct.composition.formula}")
        print(f"  - Atoms: {len(struct)}")
    else:
        print(f"⚠ File not found: {vasp_file}")
except Exception as e:
    print(f"✗ VASP loading failed: {e}")

print("\n✓ I/O module tests complete")
```

Run it:
```bash
python test_io_manual.py
```

### Test 3: Layering Module (Layer Detection)

Create `test_layering_manual.py`:

```python
from interfaceml.core import io, layering
from pathlib import Path

print("Testing Layering Module...")
print("-" * 50)

# Load a structure
struct_file = "structures/perovskites/fapbi3-1.cif"
if not Path(struct_file).exists():
    print(f"⚠ File not found: {struct_file}")
    print("Please provide a valid structure file")
    exit(1)

struct = io.load_structure(struct_file)
print(f"Loaded structure: {struct.composition.formula}")
print(f"Total atoms: {len(struct)}")

# Test layer detection
layers, tol = layering.split_layers_by_z(struct, gap_cut=False)
print(f"\n✓ Detected {len(layers)} layers (tolerance: {tol:.3f} Å)")

for i, layer in enumerate(layers[:5]):  # Show first 5 layers
    print(f"  Layer {i+1}: {len(layer)} atoms")

# Test whole molecule inclusion
if len(layers) > 0:
    fixed_indices = layers[0]  # Fix first layer
    expanded = layering.include_whole_molecules(struct, fixed_indices)
    print(f"\n✓ Whole molecule inclusion:")
    print(f"  Original: {len(fixed_indices)} atoms")
    print(f"  Expanded: {len(expanded)} atoms")

print("\n✓ Layering module tests complete")
```

Run it:
```bash
python test_layering_manual.py
```

---

## Testing CLI Tools

### Test 4: Interface Builder (Adsorbate Mode)

```bash
# Step 1: Verify script exists
ls -la build_heterojunctions/interface_builder.py

# Step 2: Check help message
python build_heterojunctions/interface_builder.py --help | head -20

# Step 3: Build a simple interface
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --miller 0 0 1 \
    --supercell auto \
    --output test_output/test_interface.vasp \
    --verbose
```

**Expected Output:**
- Console messages about slab generation
- Supercell information
- Output file: `test_output/test_interface.vasp`

**Verify Output:**
```bash
# Check output file exists
ls -lh test_output/test_interface.vasp

# Count atoms (should be more than input)
grep -A 1 "Direct" test_output/test_interface.vasp | tail -5

# Visualize (if you have ASE)
# ase gui test_output/test_interface.vasp
```

### Test 5: Fix Layers Tool

```bash
# Use the interface we just built
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input test_output/test_interface.vasp \
    --n_fix_layers 3 \
    --output test_output/test_interface_fixed.vasp \
    --debug_layers
```

**Expected Output:**
- Layer detection statistics
- Fixed atom indices
- Output file: `test_output/test_interface_fixed.vasp`

**Verify Output:**
```bash
# Check for Selective Dynamics section
grep "Selective" test_output/test_interface_fixed.vasp

# Count fixed atoms (F F F lines)
grep "F  F  F" test_output/test_interface_fixed.vasp | wc -l

# Count relaxed atoms (T T T lines)
grep "T  T  T" test_output/test_interface_fixed.vasp | wc -l
```

### Test 6: Multi-Layer Stack

```bash
# Build a 3-layer stack (Perovskite/C70/C60)
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C70-D5h.cif structures/etl/C60-Ih.cif \
    --termination AI \
    --layer_gaps 2.0 2.0 \
    --supercell auto \
    --output test_output/trilayer.vasp \
    --verbose

# Split it into layers
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input test_output/trilayer.vasp \
    --n_interfaces 2 \
    --fixed_layers 1 \
    --output test_output/trilayer_fixed.vasp \
    --print_only
```

---

## Testing Web Interface

### Test 7: Start Web Server

```bash
# Terminal 1: Start the server
python -m interfaceml.web.app
```

**Expected Output:**
```
Starting InterfaceML Web Server...
Upload folder: /tmp/interfaceml_XXXXXX
Core modules available: True

Open your browser and navigate to: http://localhost:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

### Test 8: Web Interface Health Check

**In a new terminal:**

```bash
# Test health endpoint
curl http://localhost:5000/api/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "core_available": true,
  "version": "1.0.0"
}
```

### Test 9: Web Interface File Upload

**Open browser to http://localhost:5000**

**Manual Test Steps:**

1. **Tab Navigation**
   - [ ] Click each tab (Adsorbate, Interface, Layers, Density)
   - [ ] Verify tab switching works smoothly
   - [ ] Check animations and highlighting

2. **File Upload (Adsorbate Tab)**
   - [ ] Click "Click or drag file here" for base structure
   - [ ] Select `structures/perovskites/fapbi3-1.cif`
   - [ ] Verify file info appears (composition, atoms, lattice)
   - [ ] Upload C60: `structures/etl/C60-Ih.cif`
   - [ ] Verify both files show green checkmarks

3. **Parameter Configuration**
   - [ ] Change termination to "FAI-terminated"
   - [ ] Set distance to 2.5 Å
   - [ ] Select "Auto (optimal)" supercell
   - [ ] Verify all inputs respond correctly

4. **Build Structure** (Note: Full functionality requires implementation)
   - [ ] Click "Build Structure" button
   - [ ] Check for loading indicator
   - [ ] Verify error message or result display

5. **Layer Fixing Tab**
   - [ ] Switch to "Layer Fixing" tab
   - [ ] Upload `test_output/test_interface.vasp`
   - [ ] Set "Number of Layers to Fix" to 3
   - [ ] Check "Include whole molecules"
   - [ ] Click "Fix Layers" button
   - [ ] Verify result appears with download link

### Test 10: Web API Testing with curl

```bash
# Upload a file via API
curl -X POST http://localhost:5000/api/upload \
  -F "file=@structures/perovskites/fapbi3-1.cif" \
  -F "file_type=base"

# Fix layers via API
curl -X POST http://localhost:5000/api/fix-layers \
  -H "Content-Type: application/json" \
  -d '{
    "structure_file": "/path/to/uploaded/file.vasp",
    "mode": "by_z_layers",
    "n_fix_layers": 3,
    "include_molecules": true
  }'
```

---

## Testing Python API

### Test 11: Interactive Python Session

```bash
# Start Python interactive session
python
```

```python
from interfaceml.core import io, layering
from pathlib import Path

# Load a structure
struct = io.load_structure("structures/perovskites/fapbi3-1.cif")
print(f"Loaded: {struct.composition.formula}")
print(f"Atoms: {len(struct)}")

# Detect layers
layers, tol = layering.split_layers_by_z(struct, gap_cut=False)
print(f"Detected {len(layers)} layers")

# Fix bottom 3 layers
fixed_indices = []
for layer in layers[:3]:
    fixed_indices.extend(layer)
print(f"Fixing {len(fixed_indices)} atoms")

# Include whole molecules
fixed_expanded = layering.include_whole_molecules(struct, fixed_indices)
print(f"After molecule inclusion: {len(fixed_expanded)} atoms")

# Create selective dynamics
selective_dynamics = [
    (False, False, False) if i in fixed_expanded else (True, True, True)
    for i in range(len(struct))
]

# Write output
io.write_poscar(struct, "test_output/api_test.vasp", 
                selective_dynamics=selective_dynamics)
print("✓ Output written to test_output/api_test.vasp")
```

### Test 12: Jupyter Notebook (Optional)

Create `test_notebook.ipynb`:

```python
# Cell 1: Imports
from interfaceml.core import io, layering
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Load structure
struct = io.load_structure("structures/perovskites/fapbi3-1.cif")
print(f"Structure: {struct.composition.formula}")

# Cell 3: Analyze layer heights
n = layering.interface_normal_unit(struct)
heights = np.dot(np.array(struct.cart_coords), n)

plt.figure(figsize=(10, 6))
plt.hist(heights, bins=50, edgecolor='black')
plt.xlabel('Height along interface normal (Å)')
plt.ylabel('Number of atoms')
plt.title('Atomic Height Distribution')
plt.grid(True, alpha=0.3)
plt.show()

# Cell 4: Layer statistics
layers, tol = layering.split_layers_by_z(struct)
layer_sizes = [len(layer) for layer in layers]

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(layers)+1), layer_sizes)
plt.xlabel('Layer number')
plt.ylabel('Atoms per layer')
plt.title(f'Layer Distribution (tolerance: {tol:.3f} Å)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Integration Testing

### Test 13: Complete Workflow

Run the example workflow script:

```bash
# Make sure it's executable
chmod +x examples/example_workflow.sh

# Run the workflow
bash examples/example_workflow.sh
```

**Expected Steps:**
1. Build FAPbI3/C60 interface
2. Add selective dynamics
3. Create output files in `examples/output/`

**Verify Results:**
```bash
# Check output files
ls -lh examples/output/

# Verify POSCAR structure
head -20 examples/output/fapbi3_c60_interface.vasp

# Verify selective dynamics
grep "Selective" examples/output/fapbi3_c60_interface_fixed.vasp
```

### Test 14: Stress Test (Large Structure)

```bash
# Build a large supercell
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --supercell_xy 3 3 \
    --output test_output/large_interface.vasp \
    --verbose

# Check atom count
grep -A 1 "Direct" test_output/large_interface.vasp | tail -1 | awk '{print $1}'
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem:**
```
ImportError: No module named 'pymatgen'
```

**Solution:**
```bash
pip install pymatgen
# or
conda install -c conda-forge pymatgen
```

#### Issue 2: Web Server Won't Start

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill the process or use a different port
python -m interfaceml.web.app --port 8000
```

#### Issue 3: Permission Errors

**Problem:**
```
PermissionError: [Errno 13] Permission denied: 'test_output'
```

**Solution:**
```bash
# Create output directory with proper permissions
mkdir -p test_output
chmod 755 test_output
```

#### Issue 4: Structure Loading Fails

**Problem:**
```
ValueError: Could not parse structure from file
```

**Solution:**
```bash
# Verify file format
file structures/perovskites/fapbi3-1.cif

# Check file contents
head -20 structures/perovskites/fapbi3-1.cif

# Try converting with pymatgen
python -c "from pymatgen.core import Structure; s = Structure.from_file('file.cif'); print(s)"
```

---

## Test Results Checklist

After completing all tests, verify:

- [ ] All core module tests pass
- [ ] CLI tools run without errors
- [ ] Web interface loads correctly
- [ ] File upload works in browser
- [ ] API endpoints respond correctly
- [ ] Python API imports successfully
- [ ] Example workflow completes
- [ ] Output files are valid POSCAR format
- [ ] Selective dynamics flags are correct
- [ ] Documentation is accurate

---

## Performance Benchmarks

Record performance for future comparison:

```bash
# Time interface building
time python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --supercell auto \
    --output test_output/benchmark.vasp

# Time layer detection
time python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input test_output/benchmark.vasp \
    --n_fix_layers 3 \
    --output test_output/benchmark_fixed.vasp
```

**Expected Times:**
- Interface building: 5-30 seconds
- Layer fixing: < 5 seconds
- Web page load: < 1 second

---

## Automated Testing Script

Create `run_all_tests.sh`:

```bash
#!/bin/bash
# Automated test suite for InterfaceML

set -e  # Exit on error

echo "╔════════════════════════════════════════════════╗"
echo "║   InterfaceML Automated Test Suite            ║"
echo "╚════════════════════════════════════════════════╝"

# Setup
mkdir -p test_output
cd "$(dirname "$0")"

# Test 1: Core modules
echo -e "\n[1/6] Testing core modules..."
python tests/test_basic.py || exit 1

# Test 2: CLI - Interface builder
echo -e "\n[2/6] Testing interface builder..."
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --output test_output/auto_test_interface.vasp \
    > /dev/null 2>&1 || exit 1

# Test 3: CLI - Layer fixing
echo -e "\n[3/6] Testing layer fixing..."
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input test_output/auto_test_interface.vasp \
    --n_fix_layers 3 \
    --output test_output/auto_test_fixed.vasp \
    > /dev/null 2>&1 || exit 1

# Test 4: Verify output files
echo -e "\n[4/6] Verifying output files..."
[ -f test_output/auto_test_interface.vasp ] || exit 1
[ -f test_output/auto_test_fixed.vasp ] || exit 1
grep -q "Selective" test_output/auto_test_fixed.vasp || exit 1

# Test 5: API test
echo -e "\n[5/6] Testing Python API..."
python -c "
from interfaceml.core import io, layering
s = io.load_structure('test_output/auto_test_interface.vasp')
layers, tol = layering.split_layers_by_z(s)
print(f'API test: {len(layers)} layers detected')
" || exit 1

# Test 6: Web API health check
echo -e "\n[6/6] Testing web API..."
python -c "
from interfaceml.web import app as web_app
print('Web app module loaded successfully')
" || exit 1

echo -e "\n╔════════════════════════════════════════════════╗"
echo "║   ✓ All tests passed successfully!            ║"
echo "╚════════════════════════════════════════════════╝"
```

Make it executable and run:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

---

## Next Steps

After successful testing:

1. **Document your results** - Note any issues or observations
2. **Commit tested changes** - If using git
3. **Review extension guide** - See [EXTENSIBILITY_GUIDE.md](EXTENSIBILITY_GUIDE.md)
4. **Add your own tests** - For new features
5. **Share feedback** - Report bugs or suggest improvements

---

**Testing Complete! 🎉**

You now have a fully validated InterfaceML installation ready for production use.
