# InterfaceML

<div align="center">

**Professional Heterojunction Modeling Platform**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Pymatgen](https://img.shields.io/badge/Powered%20by-Pymatgen-orange)](https://pymatgen.org/)

*A comprehensive toolkit for building, analyzing, and optimizing heterostructure interfaces for DFT calculations*

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Documentation](#documentation) | [Web Interface](#web-interface)

</div>

---

## Features

InterfaceML provides a complete suite of tools for heterojunction modeling with a focus on:

### Adsorbate Modeling
- Build perovskite/fullerene interfaces for solar cell applications
- Automatic surface termination selection (PbI, FAI, MAI, AI)
- Multi-layer stacking (e.g., Perovskite/C70/C60)
- Intelligent supercell generation with minimal adsorbate interactions
- Support for symmetric slabs with consistent terminations

### Interface Builder
- Coherent interface matching with automatic strain optimization
- ZSL algorithm for finding commensurate supercells
- Bidirectional or unidirectional strain application
- Customizable Miller indices for both materials
- Comprehensive strain analysis and reporting

### Layer Management
- **Selective Dynamics**: Fix bottom N layers for relaxation calculations
- **Layer Splitting**: Separate multi-layer stacks into individual files
- Automatic layer detection for interface structures
- Intelligent whole-molecule inclusion (prevents splitting organic cations)
- Support for multi-interface structures
- Preserves original lattice parameters when splitting
- Compatible with VASP and CP2K

### Density Analysis
- Compute charge density differences: Delta rho = rho(interface) - rho(A) - rho(B)
- Streaming cube file processing (memory-efficient)
- Direct VESTA visualization support

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager

### Install from source

```bash
git clone https://github.com/yourusername/InterfaceML.git
cd InterfaceML
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

### Core Dependencies
- `numpy >= 1.22.0`
- `pymatgen >= 2022.0.0`
- `flask >= 2.0.0` (for web interface)
- `flask-cors >= 3.0.0` (for web interface)

---

## Quick Start

### Command-Line Interface

#### 1. Build an Adsorbate Model

```bash
# Basic usage: Perovskite + C60
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/H5PbCI3N2.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --supercell auto \
    --output my_interface.vasp

# Multi-layer stack: Perovskite + C70 + C60
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C70-D5h.cif structures/etl/C60-Ih.cif \
    --termination FAI \
    --layer_gaps 2.0 2.0 \
    --supercell auto \
    --output perovskite_c70_c60.vasp
```

#### 2. Fix Layers for Relaxation

```bash
# Fix bottom 3 layers
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input my_interface.vasp \
    --n_fix_layers 3 \
    --output my_interface_fixed.vasp

# Multi-interface structure: split into layers
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input perovskite_c70_c60.vasp \
    --n_interfaces 2 \
    --fixed_layers 1 \
    --output perovskite_c70_c60_fixed.vasp
```

#### 2b. Split Multi-Layer Structure

```bash
# Split a 3-layer stack (2 interfaces) into separate files
# Each layer keeps the original lattice parameters
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input trilayer.vasp \
    --n_interfaces 2 \
    --print_only

# Or use Python API
python -c "
from interfaceml.core import io, splitting
structure = io.load_structure('trilayer.vasp')
layers = splitting.split_structure_into_layers(structure, n_interfaces=2)
for i, layer in enumerate(layers, 1):
    io.write_poscar(layer, f'layer{i}.vasp')
    print(f'Layer {i}: {len(layer)} atoms')
"
```

#### 3. Compute Charge Density Difference

```bash
python build_heterojunctions/delta_density_cube.py \
    --interface interface_density.cube \
    --layer1 perovskite_density.cube \
    --layer2 c60_density.cube \
    --output delta_density.cube
```

### Python API

```python
from interfaceml.core import io, layering

# Load structure
structure = io.load_structure("my_structure.vasp")

# Detect layers
layers, tolerance = layering.split_layers_by_z(structure, gap_cut=True)

# Fix bottom 3 layers
fixed_indices = []
for layer in layers[:3]:
    fixed_indices.extend(layer)

# Include whole molecules
fixed_indices = layering.include_whole_molecules(structure, fixed_indices)

# Generate selective dynamics
selective_dynamics = [(False, False, False) if i in fixed_indices else (True, True, True) 
                      for i in range(len(structure))]

# Write POSCAR
io.write_poscar(structure, "output_fixed.vasp", selective_dynamics=selective_dynamics)
```

---

## Web Interface

InterfaceML includes a web interface for interactive modeling:

```bash
# Start the web server
python -m interfaceml.web.app

# Or use the CLI command
interfaceml-web
```

### AI Interface Generation (EGNN + Local GNN)

If the fullerene diffusion checkpoint is available, you can generate a
perovskite/fullerene interface directly through the web API:

#### AI Setup (Required for the AI tab)

The AI module depends on PyTorch + PyTorch Geometric and is not installed
with the default `requirements.txt`. Recommended Python: 3.10 or 3.11.

```bash
# Option A: install optional AI dependencies
pip install -e ".[ai]"

# Option B: install fullerene module requirements directly
pip install -r requirements.txt
pip install -r fullerene_e3gen/requirements.txt
```

If PyTorch/PyG installation fails for your platform, follow their official
installation guidance for your CUDA/CPU stack.

You can override the AI asset locations with:
- `INTERFACEML_FULLERENE_PATH`
- `INTERFACEML_FULLERENE_CHECKPOINT`

To start the web app with a specific Python environment:

```bash
INTERFACEML_PYTHON=/path/to/python ./start_web.sh
```

```json
POST /api/ai/generate-interface
{
  "base_filename": "my_base_slab.cif",
  "num_atoms": 60,
  "miller": [0, 0, 1],
  "slab_thickness": 18.0,
  "vacuum": 20.0,
  "separation": 3.2,
  "xy_frac": [0.5, 0.5],
  "local_refine": true,
  "refine_scope": "adsorbate"
}
```

The response includes a downloadable `POSCAR` (`.vasp`) for the combined interface.

Then open your browser:
- Documentation homepage: http://localhost:5000/
- Web app interface: http://localhost:5000/app

### Web Features
- **Drag-and-drop file upload** for structure files (.cif, .vasp, .xyz)
- **Interactive parameter configuration** with real-time validation
- **Visual feedback** with structure information display
- **Direct download** of generated structure files
- **Multi-tab interface** for different modeling workflows
- **Layer splitting** - Separate multi-layer structures (preserves lattice)
- **Selective dynamics** - Fix layers for relaxation

---

## Documentation

### Directory Structure

```
InterfaceML/
|-- interfaceml/              # Main Python package
|   |-- core/                 # Core functionality modules
|   |   |-- io.py            # Structure I/O (VASP, CIF, XYZ)
|   |   |-- layering.py      # Layer detection and manipulation
|   |   `-- splitting.py     # Smart layer splitting
|   |-- cli/                  # Command-line interface tools
|   |   |-- build_interface.py
|   |   |-- fix_layers.py
|   |   `-- delta_density.py
|   |-- web/                  # Flask web application
|   |   |-- app.py           # App factory + blueprint registration
|   |   |-- routes/          # API blueprints (common, interface, dos, ai)
|   |   |-- ai.py            # AI module discovery/loading
|   |   |-- core.py          # Core module availability
|   |   |-- dos.py           # DOS/PDOS helpers
|   |   |-- utils.py         # Shared web utilities
|   |   |-- templates/       # HTML templates
|   |   `-- static/          # CSS and JavaScript
|   `-- utils/                # Utility functions
|       `-- geometry.py      # Geometric calculations
|-- build_heterojunctions/    # Original scripts (maintained for compatibility)
|   |-- interface_builder.py
|   |-- fix_interface_layers.py
|   |-- delta_density_cube.py
|   `-- README.md            # Detailed technical documentation
|-- structures/               # Example structure files
|   |-- perovskites/
|   |-- etl/
|   `-- heterojunctions/
|-- examples/                 # Tutorial notebooks and scripts
|-- docs/                     # Additional documentation
|-- setup.py                  # Package installation script
|-- requirements.txt          # Python dependencies
`-- README.md                 # This file
```

Note: `interfaceml/core` is the canonical API surface. The legacy scripts in `build_heterojunctions/`
delegate to core utilities when available to keep behavior consistent.

### Detailed Documentation
- **[build_heterojunctions/README.md](build_heterojunctions/README.md)** - Comprehensive CLI usage guide
- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Tutorial and workflows
- **[docs/LAYER_SPLITTING.md](docs/LAYER_SPLITTING.md)** - Layer splitting guide
- **[docs/SMART_SPLITTING_ALGORITHM.md](docs/SMART_SPLITTING_ALGORITHM.md)** - Smart splitting algorithm details
- **[docs/MATHEMATICAL_OVERVIEW.md](docs/MATHEMATICAL_OVERVIEW.md)** - Mathematical and algorithmic overview
- **[Tutorials](examples/)** - Example workflows

---

## Use Cases

### Solar Cell Modeling
- Perovskite/ETL (C60, TiO2) interfaces
- Charge transfer analysis
- Interface optimization

### Heterostructure Research
- 2D material interfaces
- Metal-semiconductor junctions
- Coherent interface engineering

### DFT Workflow Integration
- Pre-processing for VASP, CP2K, Quantum ESPRESSO
- Automatic structure preparation
- Post-processing charge density analysis

---

## Advanced Usage

### Building from Pre-relaxed Structures

```bash
# Use a pre-relaxed slab directly
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --slab_input my_relaxed_slab.vasp \
    --adsorbate structures/etl/C60-Ih.cif \
    --distance 2.5 \
    --output relaxed_with_c60.vasp
```

### Custom Supercells

```bash
# Specify exact supercell dimensions
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/H5PbCI3N2.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --supercell_xy 2 2 \
    --output interface_2x2.vasp
```

### Layer Detection Tuning

```bash
# Adjust tolerance for layer clustering
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input structure.xyz \
    --n_fix_layers 3 \
    --tol 0.5 \
    --debug_layers \
    --print_only
```

---

## Examples

### Example 1: FAPbI3/C60 Solar Cell Interface

```bash
# Step 1: Build interface
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination FAI \
    --distance 2.5 \
    --supercell auto \
    --output fapbi3_c60.vasp

# Step 2: Add selective dynamics (fix bottom 3 layers)
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input fapbi3_c60.vasp \
    --n_fix_layers 3 \
    --output fapbi3_c60_fixed.vasp

# Step 3: Run DFT calculation (your favorite code)
# ...

# Step 4: Analyze charge transfer
python build_heterojunctions/delta_density_cube.py \
    --interface fapbi3_c60_density.cube \
    --layer1 fapbi3_density.cube \
    --layer2 c60_density.cube \
    --output delta_charge.cube
```

---

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/InterfaceML.git
cd InterfaceML

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Quick import check
python -c "from interfaceml.core import io, layering, splitting; print('OK')"
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions, bug reports, or feature requests:
- Open an issue on [GitHub](https://github.com/yourusername/InterfaceML/issues)
- Email: interface@example.com

---

## Acknowledgments

- Built with [Pymatgen](https://pymatgen.org/) - Materials analysis library
- Interface matching based on ZSL algorithm
- Inspired by real-world DFT workflow challenges in heterojunction research

---

## Citation

If you use InterfaceML in your research, please cite:

```bibtex
@software{interfaceml2024,
  title = {InterfaceML: A Professional Toolkit for Heterojunction Modeling},
  author = {Interface Modeling Lab},
  year = {2024},
  url = {https://github.com/yourusername/InterfaceML}
}
```

---

<div align="center">

If InterfaceML supports your research, citations are appreciated.

</div>
