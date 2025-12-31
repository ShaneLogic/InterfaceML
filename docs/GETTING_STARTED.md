# Getting Started with InterfaceML

This guide will help you get started with InterfaceML for heterojunction modeling.

## Table of Contents
- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Your First Interface](#your-first-interface)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Check Prerequisites

```bash
# Check Python version (must be 3.8+)
python --version

# Verify pip is installed
pip --version
```

### Step 2: Install InterfaceML

```bash
# Clone the repository
git clone https://github.com/yourusername/InterfaceML.git
cd InterfaceML

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test CLI tools
python build_heterojunctions/interface_builder.py --help

# Test Python import
python -c "from interfaceml.core import io, layering; print('✓ InterfaceML imported successfully')"

# (Optional) Start web interface
python -m interfaceml.web.app
```

---

## Basic Concepts

### 1. Structure Files

InterfaceML supports multiple file formats:

- **CIF** (.cif) - Crystallographic Information File
- **POSCAR/VASP** (.vasp, .poscar) - VASP format
- **XYZ** (.xyz) - CP2K-style with cell vectors

```python
from interfaceml.core import io

# Load any supported format
structure = io.load_structure("path/to/file.cif")
print(f"Loaded {len(structure)} atoms")
print(f"Composition: {structure.composition.formula}")
```

### 2. Surface Terminations

For perovskite structures, InterfaceML supports four termination types:

- **PbI** - Lead-iodide terminated (common for stability)
- **FAI** - Formamidinium-iodide terminated
- **MAI** - Methylammonium-iodide terminated  
- **AI** - Generic A-site cation terminated

The algorithm automatically:
- Selects slabs with desired termination
- Ensures complete organic molecules at surfaces
- Creates symmetric slabs (same top/bottom termination)

### 3. Supercells

For adsorbate modeling, supercells are needed to minimize periodic interactions:

- **Auto mode**: Automatically sizes based on adsorbate diameter (recommended)
- **Manual mode**: Specify exact nx × ny dimensions

Rule of thumb: Adsorbate-adsorbate distance should be > 10 Å

### 4. Layer Detection

InterfaceML can automatically detect atomic layers for:
- Fixing bulk regions in relaxation calculations
- Splitting multi-layer stacks
- Analyzing interface structure

---

## Your First Interface

Let's build a FAPbI3/C60 interface step-by-step:

### Step 1: Prepare Structure Files

```bash
# Check if example structures exist
ls structures/perovskites/fapbi3-1.cif
ls structures/etl/C60-Ih.cif
```

### Step 2: Build the Interface

```bash
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination FAI \
    --distance 2.5 \
    --miller 0 0 1 \
    --supercell auto \
    --output fapbi3_c60_interface.vasp
```

**Output:**
- `fapbi3_c60_interface.vasp` - Ready for DFT calculation
- Console will show:
  - Number of atoms
  - Supercell dimensions
  - Termination details
  - Vacuum thickness

### Step 3: Add Selective Dynamics

```bash
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input fapbi3_c60_interface.vasp \
    --n_fix_layers 3 \
    --output fapbi3_c60_interface_fixed.vasp
```

**Output:**
- `fapbi3_c60_interface_fixed.vasp` - With selective dynamics flags
- Bottom 3 layers are fixed (F F F)
- Upper layers and adsorbate are relaxed (T T T)
- Organic molecules kept intact

### Step 4: Visualize (Optional)

Use VESTA, Ovito, or ASE to view the structure:

```bash
# With ASE (if installed)
ase gui fapbi3_c60_interface_fixed.vasp
```

---

## Common Workflows

### Workflow 1: Perovskite Solar Cell

```bash
# 1. Build perovskite/ETL interface
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/H5PbCI3N2.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --supercell auto \
    --output perovskite_etl.vasp

# 2. Fix bottom layers
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input perovskite_etl.vasp \
    --n_fix_layers 3 \
    --output perovskite_etl_fixed.vasp

# 3. Run DFT (VASP example)
# Edit INCAR, KPOINTS, POTCAR as needed
# mpirun -np 16 vasp_std

# 4. Analyze charge transfer
python build_heterojunctions/delta_density_cube.py \
    --interface CHGCAR_interface.cube \
    --layer1 CHGCAR_perovskite.cube \
    --layer2 CHGCAR_c60.cube \
    --output delta_charge.cube
```

### Workflow 2: Multi-Layer Stack

```bash
# Build Perovskite/C70/C60 stack
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C70-D5h.cif structures/etl/C60-Ih.cif \
    --termination AI \
    --layer_gaps 2.0 2.0 \
    --supercell auto \
    --output trilayer.vasp

# Split into layers and fix bottom
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input trilayer.vasp \
    --n_interfaces 2 \
    --fixed_layers 1 \
    --output trilayer_fixed.vasp
```

### Workflow 3: Using Pre-Relaxed Structures

```bash
# If you already have a relaxed slab
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --slab_input my_relaxed_slab_supercell.vasp \
    --adsorbate structures/etl/C60-Ih.cif \
    --distance 2.5 \
    --output relaxed_with_adsorbate.vasp
```

---

## Troubleshooting

### Problem: "Could not find slab with desired termination"

**Solution:**
- Try different `--miller` indices (e.g., `0 0 1`, `1 1 1`)
- Increase `--min_slab_size` and `--min_vacuum_size`
- For perovskites, ensure the input structure has complete formula units
- Check if your termination type matches the structure (FAI for FAPbI3, MAI for MAPbI3)

### Problem: "Too many atoms in supercell"

**Solution:**
- Reduce supercell size: `--supercell_xy 1 1` (instead of auto)
- Use a smaller adsorbate or fewer adsorbate layers
- Increase `--max_atoms` if your hardware can handle it

### Problem: "Layer detection gives wrong layers"

**Solution:**
- Visualize with `--debug_layers` flag
- Adjust `--tol` parameter (try 0.3-0.7 Å)
- Use `--layer_elements` to specify which elements define layers
- For XYZ files, ensure cell vectors (Tv_1/Tv_2/Tv_3) are present

### Problem: "Organic molecules are split across fixed/relaxed boundary"

**Solution:**
- The `--no_include_molecules` flag should NOT be used (molecules are included by default)
- Check that you're using the latest version of the script
- Verify with `--debug_layers` that molecules are being detected

### Problem: "Web interface not starting"

**Solution:**
```bash
# Check Flask is installed
pip install flask flask-cors

# Try specifying port
python -m interfaceml.web.app --port 8000

# Check firewall settings
# Make sure port 5000 (or chosen port) is not blocked
```

### Problem: "Import errors with pymatgen"

**Solution:**
```bash
# Update pymatgen
pip install --upgrade pymatgen

# Or use conda
conda install -c conda-forge pymatgen

# Verify installation
python -c "import pymatgen; print(pymatgen.__version__)"
```

---

## Next Steps

- Read the [detailed CLI documentation](../build_heterojunctions/README.md)
- Explore example structures in `structures/`
- Try the [web interface](../interfaceml/web/) for interactive modeling
- Join our community discussions on GitHub

---

## Getting Help

- 📖 Check the [main README](../README.md)
- 🐛 Report bugs on [GitHub Issues](https://github.com/yourusername/InterfaceML/issues)
- 💬 Ask questions in [Discussions](https://github.com/yourusername/InterfaceML/discussions)
- 📧 Email: interface@example.com

---

**Happy modeling! 🚀**
