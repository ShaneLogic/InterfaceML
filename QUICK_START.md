# Quick Start Guide

Get InterfaceML up and running in 5 minutes!

## 🚀 Installation

```bash
# 1. Navigate to the project directory
cd /path/to/InterfaceML

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install package for easy access
pip install -e .
```

---

## ✅ Verify Installation

```bash
# Test Python import
python -c "from interfaceml.core import io, layering; print('✓ InterfaceML ready!')"

# Check CLI tools
python build_heterojunctions/interface_builder.py --help
```

---

## 🎯 Choose Your Interface

### Option 1: Command-Line (Traditional)

Best for: Automation, scripting, HPC workflows

```bash
# Build adsorbate model
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination FAI \
    --distance 2.5 \
    --output my_interface.vasp

# Fix layers for relaxation
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input my_interface.vasp \
    --n_fix_layers 3 \
    --output my_interface_fixed.vasp
```

### Option 2: Web Interface (Interactive)

Best for: Visualization, parameter exploration, quick testing

```bash
# Start web server
python -m interfaceml.web.app

# Open browser to http://localhost:5000
# Use drag-and-drop interface!
```

### Option 3: Python API (Programmatic)

Best for: Custom workflows, Jupyter notebooks, integration

```python
from interfaceml.core import io, layering

# Load structure
structure = io.load_structure("my_structure.vasp")

# Detect and fix bottom 3 layers
layers, tol = layering.split_layers_by_z(structure)
fixed_indices = []
for layer in layers[:3]:
    fixed_indices.extend(layer)

# Include whole molecules
fixed_indices = layering.include_whole_molecules(structure, fixed_indices)

# Write output
selective_dynamics = [
    (False, False, False) if i in fixed_indices else (True, True, True)
    for i in range(len(structure))
]
io.write_poscar(structure, "output.vasp", selective_dynamics=selective_dynamics)
```

---

## 📖 Next Steps

1. **Read the main README**: [`README.md`](README.md)
2. **Try the example workflow**: `bash examples/example_workflow.sh`
3. **Explore detailed docs**: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)
4. **Check technical details**: [`build_heterojunctions/README.md`](build_heterojunctions/README.md)

---

## 🆘 Common Issues

### "No module named 'pymatgen'"

```bash
pip install pymatgen
# or
conda install -c conda-forge pymatgen
```

### "No module named 'flask'"

```bash
pip install flask flask-cors
```

### Web interface won't start

```bash
# Try different port
python -m interfaceml.web.app --port 8000

# Check firewall settings
```

### Import errors

```bash
# Make sure you're in the project directory
cd /path/to/InterfaceML

# Or install package
pip install -e .
```

---

## 💡 Quick Examples

### Build FAPbI3/C60 Interface

```bash
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination FAI \
    --distance 2.5 \
    --supercell auto \
    --output fapbi3_c60.vasp
```

### Build Multi-Layer Stack

```bash
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C70-D5h.cif structures/etl/C60-Ih.cif \
    --termination AI \
    --layer_gaps 2.0 2.0 \
    --supercell auto \
    --output trilayer.vasp
```

### Fix Layers with Debug Info

```bash
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input structure.vasp \
    --n_fix_layers 3 \
    --debug_layers \
    --print_only
```

---

## 🎨 Web Interface Features

After starting `python -m interfaceml.web.app`:

1. **Upload Tab** 📁
   - Drag and drop structure files
   - Automatic format detection
   - Structure info display

2. **Build Tab** 🔨
   - Configure parameters
   - Real-time validation
   - One-click generation

3. **Download Tab** ⬇️
   - Download results
   - View atom counts
   - Fixed indices list

---

## 📚 Documentation Structure

```
README.md                           ← Start here!
├── QUICK_START.md                 ← You are here
├── docs/GETTING_STARTED.md         ← Detailed tutorial
├── build_heterojunctions/README.md ← Technical reference
└── examples/example_workflow.sh    ← Complete example
```

---

## 🤝 Getting Help

- 📖 Read the [full documentation](README.md)
- 🐛 [Report bugs](https://github.com/yourusername/InterfaceML/issues)
- 💬 [Ask questions](https://github.com/yourusername/InterfaceML/discussions)
- 📧 Email: interface@example.com

---

**Ready to build amazing interfaces? Let's go! 🚀**
