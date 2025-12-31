# InterfaceML Project Structure

This document describes the professional package structure of InterfaceML after refactoring.

## Overview

InterfaceML has been transformed from a collection of standalone scripts into a professional, modular Python package with both command-line and web interfaces.

---

## Directory Layout

```
InterfaceML/
│
├── interfaceml/                    # Main Python package
│   ├── __init__.py                # Package initialization
│   │
│   ├── core/                       # Core functionality modules
│   │   ├── __init__.py
│   │   ├── io.py                  # Structure I/O (VASP, CIF, XYZ)
│   │   ├── layering.py            # Layer detection and manipulation
│   │   ├── termination.py         # Surface termination (placeholder)
│   │   ├── adsorbate.py           # Adsorbate placement (placeholder)
│   │   └── interface.py           # Interface matching (placeholder)
│   │
│   ├── cli/                        # Command-line interface wrappers
│   │   ├── __init__.py
│   │   ├── build_interface.py     # Wrapper for interface_builder.py
│   │   ├── fix_layers.py          # Wrapper for fix_interface_layers.py
│   │   └── delta_density.py       # Wrapper for delta_density_cube.py
│   │
│   ├── web/                        # Flask web application
│   │   ├── __init__.py
│   │   ├── app.py                 # Main Flask application
│   │   ├── templates/             # HTML templates
│   │   │   └── index.html         # Main web interface
│   │   └── static/                # Static assets
│   │       ├── css/
│   │       │   └── style.css      # Modern, responsive styles
│   │       └── js/
│   │           └── app.js         # Client-side JavaScript
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── geometry.py             # Geometric calculations
│
├── build_heterojunctions/          # Original scripts (maintained)
│   ├── interface_builder.py       # Heterojunction & adsorbate builder
│   ├── fix_interface_layers.py    # Selective dynamics tool
│   ├── delta_density_cube.py      # Charge density analysis
│   ├── stack_slabs_relax.py       # Legacy stacking tool
│   ├── expand_cif_cell.py         # CIF manipulation
│   ├── _utils_xyz.py              # XYZ parsing helpers
│   ├── _utils_structures.py       # Structure loading helpers
│   ├── _utils_layering.py         # Layering helpers
│   └── README.md                  # Detailed technical documentation
│
├── structures/                     # Example structure files
│   ├── perovskites/               # Perovskite CIF files
│   │   ├── H5PbCI3N2.cif
│   │   ├── H6PbCI3N.cif
│   │   └── fapbi3-1.cif
│   ├── etl/                       # Electron transport layer molecules
│   │   ├── C60-Ih.cif
│   │   ├── C70-D5h.cif
│   │   └── TiO2.cif
│   └── heterojunctions/           # Example interfaces
│       └── ...
│
├── examples/                       # Example scripts and tutorials
│   ├── example_workflow.sh        # Complete workflow demonstration
│   └── output/                    # Output directory for examples
│
├── docs/                           # Documentation
│   ├── GETTING_STARTED.md         # Quick start guide
│   └── API.md                     # API reference (TODO)
│
├── tests/                          # Unit tests
│   ├── test_basic.py              # Basic functionality tests
│   └── ...
│
├── relax_slab/                     # Example relaxed structures
│   └── ...
│
├── setup.py                        # Package installation script
├── requirements.txt                # Python dependencies
├── README.md                       # Main project documentation
├── PROJECT_STRUCTURE.md            # This file
└── .gitignore                      # Git ignore patterns
```

---

## Module Descriptions

### Core Modules (`interfaceml/core/`)

#### `io.py`
- **Purpose**: Unified structure file I/O
- **Key Functions**:
  - `load_structure()`: Load VASP, CIF, or XYZ files
  - `write_poscar()`: Write VASP POSCAR with optional selective dynamics
  - `read_cp2k_xyz_last_frame()`: Parse CP2K trajectory files
  - `get_element_symbols()`: Extract element symbols from structure

#### `layering.py`
- **Purpose**: Layer detection and atom selection
- **Key Functions**:
  - `interface_normal_unit()`: Compute interface normal vector
  - `split_layers_by_z()`: Cluster atoms into z-layers
  - `split_stack_layers()`: Split multi-layer structures by gaps
  - `include_whole_molecules()`: Ensure molecules aren't split
  - `unwrap_periodic_1d()`: Handle periodic boundary conditions

#### `termination.py` (Placeholder)
- **Purpose**: Surface termination analysis (to be extracted from interface_builder.py)
- **Planned Functions**:
  - `identify_termination()`: Determine surface termination type
  - `score_slab()`: Score slab quality based on termination
  - `select_symmetric_slab()`: Choose slab with desired termination

#### `adsorbate.py` (Placeholder)
- **Purpose**: Adsorbate placement logic
- **Planned Functions**:
  - `place_adsorbate()`: Position molecule above surface
  - `build_multilayer_stack()`: Stack multiple adsorbates
  - `optimize_supercell()`: Determine optimal supercell size

#### `interface.py` (Placeholder)
- **Purpose**: Interface matching algorithms
- **Planned Functions**:
  - `find_matching_supercells()`: ZSL algorithm
  - `apply_strain()`: Apply strain to match interfaces
  - `build_interface()`: Construct heterojunction

### Utility Modules (`interfaceml/utils/`)

#### `geometry.py`
- **Purpose**: Basic geometric operations
- **Key Functions**:
  - `angle_between()`: Compute angle between vectors
  - `normalize_vector()`: Vector normalization
  - `compute_distance()`: Euclidean distance
  - `estimate_molecule_diameter()`: Molecular size estimation
  - `center_of_mass()`: Compute center of mass

### Web Application (`interfaceml/web/`)

#### `app.py`
- **Purpose**: Flask web server with REST API
- **Endpoints**:
  - `GET /`: Main web interface
  - `POST /api/upload`: File upload
  - `POST /api/build-adsorbate`: Build adsorbate models
  - `POST /api/build-interface`: Build heterojunctions
  - `POST /api/fix-layers`: Add selective dynamics
  - `GET /api/download/<filename>`: Download results

#### Frontend (`templates/` and `static/`)
- **Modern, responsive design** with gradient backgrounds
- **Tab-based interface** for different functionalities
- **Drag-and-drop file upload**
- **Real-time feedback** and result display
- **Mobile-friendly** responsive layout

### CLI Wrappers (`interfaceml/cli/`)

These scripts maintain backward compatibility by delegating to the original `build_heterojunctions/` scripts:
- `build_interface.py`: Wrapper for `interface_builder.py`
- `fix_layers.py`: Wrapper for `fix_interface_layers.py`
- `delta_density.py`: Wrapper for `delta_density_cube.py`

---

## Usage Modes

### 1. Original CLI (Unchanged)

```bash
# Still works exactly as before
python build_heterojunctions/interface_builder.py --adsorbate_mode ...
python build_heterojunctions/fix_interface_layers.py --by_z_layers ...
```

### 2. New Package CLI

```bash
# After pip install -e .
interfaceml-build --adsorbate_mode ...
interfaceml-fix --by_z_layers ...
interfaceml-web  # Start web server
```

### 3. Python API

```python
from interfaceml.core import io, layering

structure = io.load_structure("file.vasp")
layers, tol = layering.split_layers_by_z(structure)
# ...
```

### 4. Web Interface

```bash
python -m interfaceml.web.app
# Open http://localhost:5000 in browser
```

---

## Design Principles

### 1. **Backward Compatibility**
- Original scripts in `build_heterojunctions/` remain fully functional
- All existing command-line usage patterns work unchanged
- Legacy helper modules (`_utils_*.py`) preserved

### 2. **Modularity**
- Core functionality separated into focused modules
- Clear separation between logic (`core/`) and interface (`cli/`, `web/`)
- Reusable utility functions in `utils/`

### 3. **Professional Structure**
- Standard Python package layout
- Proper `setup.py` for installation
- Comprehensive documentation in English
- Unit tests for core functionality

### 4. **User-Friendly**
- Multiple access modes (CLI, API, Web)
- Beautiful, modern web interface
- Comprehensive documentation and examples
- Clear error messages and feedback

---

## Development Workflow

### For Users

1. **Install package**: `pip install -e .`
2. **Run examples**: `bash examples/example_workflow.sh`
3. **Use web interface**: `interfaceml-web`
4. **Read docs**: `docs/GETTING_STARTED.md`

### For Developers

1. **Modify core modules**: Edit `interfaceml/core/*.py`
2. **Test changes**: `python tests/test_basic.py`
3. **Update documentation**: Edit relevant `.md` files
4. **Maintain compatibility**: Ensure `build_heterojunctions/` scripts still work

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Package structure
- ✅ Core I/O and layering modules
- ✅ Web interface MVP
- ✅ Basic tests
- ✅ English documentation

### Phase 2 (Planned)
- Extract termination logic into `core/termination.py`
- Extract adsorbate logic into `core/adsorbate.py`
- Extract interface matching into `core/interface.py`
- Refactor original scripts to use core modules
- Add comprehensive unit tests

### Phase 3 (Future)
- Jupyter notebook tutorials
- API documentation with Sphinx
- CI/CD pipeline
- PyPI package release
- Web interface enhancements (3D visualization)

---

## Key Files Reference

### Configuration
- `setup.py`: Package installation
- `requirements.txt`: Dependencies
- `__init__.py`: Package initialization

### Documentation
- `README.md`: Main project documentation
- `build_heterojunctions/README.md`: Technical details
- `docs/GETTING_STARTED.md`: Quick start guide
- `PROJECT_STRUCTURE.md`: This file

### Scripts
- `examples/example_workflow.sh`: Complete workflow example
- `tests/test_basic.py`: Basic functionality tests

---

## Notes

- **All code comments are in English** as requested
- **All documentation is in English** as requested
- **Original CLI usage remains unchanged** for backward compatibility
- **Web interface provides modern alternative** for interactive modeling
- **Core modules provide Python API** for programmatic access

---

Last Updated: December 2024
