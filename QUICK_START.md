# Quick Start

This guide covers a minimal setup. For a full tutorial and troubleshooting, see
`docs/GETTING_STARTED.md`.

## Install

```bash
# From the repository root
cd /path/to/InterfaceML

pip install -r requirements.txt

# Optional: install the package in editable mode
pip install -e .
```

## Verify

```bash
# Test Python import
python -c "from interfaceml.core import io, layering; print('InterfaceML import: OK')"

# Check CLI help
python build_heterojunctions/interface_builder.py --help
```

## Run a Workflow

### CLI (adsorbate example)

```bash
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination FAI \
    --distance 2.5 \
    --output my_interface.vasp

python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input my_interface.vasp \
    --n_fix_layers 3 \
    --output my_interface_fixed.vasp
```

### Web Interface

```bash
python -m interfaceml.web.app
```

Then open:
- Docs homepage: `http://localhost:5000/`
- Web app UI: `http://localhost:5000/app`

### Python API

```python
from interfaceml.core import io, layering

structure = io.load_structure("my_structure.vasp")
layers, _ = layering.split_layers_by_z(structure)
fixed = [i for layer in layers[:3] for i in layer]
fixed = layering.include_whole_molecules(structure, fixed)

selective = [
    (False, False, False) if i in fixed else (True, True, True)
    for i in range(len(structure))
]
io.write_poscar(structure, "output.vasp", selective_dynamics=selective)
```

## Next Steps

- `docs/GETTING_STARTED.md` for the full tutorial and troubleshooting
- `build_heterojunctions/README.md` for CLI reference
- `docs/MATHEMATICAL_OVERVIEW.md` for algorithm details
- `README.md` for the project overview
