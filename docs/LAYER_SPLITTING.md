# Layer Splitting Feature Documentation

## Overview

The **Layer Splitting** feature allows you to separate multi-layer heterostructures into individual layer files. This is particularly useful for:
- Analyzing individual layers separately
- Preparing layer-by-layer DFT calculations
- Computing charge density differences between layers
- Studying interface effects in multi-layer stacks

## Key Features

### Preserves Original Lattice
Each separated layer maintains the **exact same lattice parameters** as the input structure. Only the atomic positions are filtered by layer.

### Automatic Interface Detection
Two modes are available:
- Simple gap-based detection (legacy CLI)
- Composition-aware smart detection (core, default in `interfaceml.core.splitting`)

### Flexible Layer Count
- **1 interface** -> 2 layers
- **2 interfaces** -> 3 layers
- **N interfaces** -> N+1 layers

---

## Usage Methods

### Method 1: Web Interface (GUI)

1. **Open web interface**:
```bash
python -m interfaceml.web.app --port 8000
```

2. **Navigate to "Layer Fixing" tab**

3. **Switch to "Split into Layers" mode**:
   - Click the "Split into Layers" button

4. **Upload structure**:
   - Drag and drop or click to upload your multi-layer structure
   - Supports: .cif, .vasp, .poscar, .xyz

5. **Configure parameters**:
   - **Number of Interfaces**: Set to 1 for 2-layer, 2 for 3-layer, etc.
   - **Minimum Gap (Angstrom)**: Minimum gap to recognize as interface (default: 0.0)

6. **Click "Split into Layers"**

7. **Download results**:
   - Each layer appears as a separate card
   - Click download link for each layer file
   - Files are named: `input_layer1.vasp`, `input_layer2.vasp`, etc.

### Method 2: Command-Line Interface

```bash
# For a 2-layer structure (1 interface)
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input trilayer.vasp \
    --n_interfaces 1 \
    --output layer1.vasp \
    --print_only

# For a 3-layer structure (2 interfaces)
python build_heterojunctions/fix_interface_layers.py \
    --by_layers \
    --input trilayer.vasp \
    --n_interfaces 2 \
    --output layer1.vasp \
    --print_only
```

### Method 3: Python API

```python
from interfaceml.core import io, splitting

# Load multi-layer structure
structure = io.load_structure("perovskite_c70_c60.vasp")

# Split into layers (2 interfaces -> 3 layers)
layers = splitting.split_structure_into_layers(
    structure,
    n_interfaces=2,
    min_gap=1.5,
    use_smart_detection=True
)

print(f"Split into {len(layers)} layers")
for i, layer in enumerate(layers, 1):
    print(f"Layer {i}: {len(layer)} atoms, {layer.composition.formula}")
    
    # Save each layer
    io.write_poscar(layer, f"layer{i}.vasp", comment=f"Layer {i}")
```

---

## Algorithm Details

The canonical algorithm descriptions live in:
- `docs/SMART_SPLITTING_ALGORITHM.md` (smart, composition-aware splitting)
- `docs/MATHEMATICAL_OVERVIEW.md` (math summary and interface matching)

At a high level, the split workflow is:
1. Project atoms onto the interface normal and unwrap periodic coordinates.
2. Identify interfaces using gap-only or composition-aware scoring.
3. Split into N+1 layers while preserving the original lattice.

---

## Examples

### Example 1: Perovskite/C60 Stack (1 Interface -> 2 Layers)

**Input**: `fapbi3_c60.vasp` (180 perovskite atoms + 60 C60 atoms)

**Command**:
```bash
# Via web: Upload -> Set n_interfaces=1 -> Split
# Via CLI:
python -m interfaceml.core.splitting split fapbi3_c60.vasp output/ --n-interfaces 1
```

**Output**:
- `fapbi3_c60_layer1.vasp`: 180 atoms (FAPbI3 slab)
- `fapbi3_c60_layer2.vasp`: 60 atoms (C60 molecule)

Both layers have the **same lattice** as the input file.

### Example 2: Perovskite/C70/C60 Stack (2 Interfaces -> 3 Layers)

**Input**: `perovskite_c70_c60.vasp` (180 + 70 + 60 = 310 atoms)

**Web Interface**:
1. Upload structure
2. Set "Number of Interfaces" = 2
3. Click "Split into Layers"
4. Download 3 layer files

**Output**:
- `perovskite_c70_c60_layer1.vasp`: 180 atoms (perovskite)
- `perovskite_c70_c60_layer2.vasp`: 70 atoms (C70)
- `perovskite_c70_c60_layer3.vasp`: 60 atoms (C60)

---

## Use Cases

### 1. Charge Density Analysis
Split layers to compute individual charge densities, then calculate:
```
Delta rho = rho(interface) - rho(layer1) - rho(layer2)
```

### 2. Layer-by-Layer Optimization
Optimize each layer separately before building the full interface.

### 3. Interface Distance Studies
Vary the distance between layers by modifying coordinates while keeping lattice fixed.

### 4. Separate DFT Calculations
Run independent calculations for each layer, then analyze coupling effects.

---

## Technical Notes

### Lattice Preservation

**Important**: Each layer file has the **identical lattice parameters** as the input:

```python
# Original structure
original_lattice = structure.lattice
# After splitting
layer1_lattice = layers[0].lattice
layer2_lattice = layers[1].lattice

# These are all the same:
assert original_lattice.matrix.all() == layer1_lattice.matrix.all()
assert original_lattice.matrix.all() == layer2_lattice.matrix.all()
```

This means:
- Same cell dimensions (a, b, c)
- Same cell angles (alpha, beta, gamma)
- Fractional coordinates are preserved
- Vacuum/empty space is maintained in each layer file

### Interface Detection

The algorithm identifies interfaces by:
1. Computing heights along interface normal
2. Finding the N largest gaps in the height distribution
3. Using these gaps as layer boundaries

**Tip**: If automatic detection fails, try adjusting `min_gap` parameter to filter out small intra-layer gaps.

### Ordering

Layers are always ordered **from bottom to top**:
- Layer 1 = lowest (bottom)
- Layer N = highest (top)

---

## Parameters Reference

### Web Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Number of Interfaces | int | 1 | Number of interfaces to split by |
| Minimum Gap (Angstrom) | float | 0.0 | Minimum gap to consider as interface |

### CLI Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n_interfaces` | int | required | Number of interfaces |
| `--min_gap` | float | 0.0 | Minimum gap threshold |
| `--output` | str | required | Output directory or file |

### Python API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure` | Structure | required | Input structure object |
| `n_interfaces` | int | required | Number of interfaces |
| `min_gap` | float | 0.0 | Minimum gap threshold |

---

## Troubleshooting

### Issue: Wrong Number of Layers

**Problem**: Got 2 layers instead of expected 3.

**Solution**:
- Increase `n_interfaces` parameter
- Or: Check if structure actually has that many interfaces
- Try with `--debug_layers` in CLI to see gap sizes

### Issue: Layers Not Separated Correctly

**Problem**: Some atoms from different layers are mixed.

**Solution**:
- Adjust `min_gap` to filter small gaps
- Verify structure has clear interface gaps (visualize in VESTA)
- Check that layers are actually separated by vacuum/low density

### Issue: Empty Layer Files

**Problem**: One or more layer files have 0 atoms.

**Solution**:
- Check input structure is valid
- Verify n_interfaces matches structure
- Try reducing `min_gap` to 0.0

---

## API Reference

### Core Function

```python
def split_structure_into_layers(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
) -> List[Structure]:
    """
    Split stacked structure into separate layer structures.
    
    Returns list of Structure objects, one per layer.
    Each structure maintains original lattice parameters.
    """
```

### Convenience Function

```python
def split_and_save_layers(
    input_file: str,
    output_dir: str,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
    base_name: str | None = None,
) -> List[str]:
    """
    Split and save to files in one step.
    
    Returns list of output file paths.
    """
```

---

## Integration with Other Tools

### With Charge Density Analysis

```bash
# 1. Split structure
# (via web or CLI)

# 2. Run DFT on each layer
# vasp_std < layer1.vasp
# vasp_std < layer2.vasp
# vasp_std < interface.vasp

# 3. Compute delta density
python build_heterojunctions/delta_density_cube.py \
    --interface interface_density.cube \
    --layer1 layer1_density.cube \
    --layer2 layer2_density.cube \
    --output delta_charge.cube
```

### With Selective Dynamics

```bash
# 1. Split into layers
# (produces layer1.vasp, layer2.vasp, ...)

# 2. Add selective dynamics to specific layer
python build_heterojunctions/fix_interface_layers.py \
    --by_z_layers \
    --input layer1.vasp \
    --n_fix_layers 3 \
    --output layer1_fixed.vasp
```

---

## Web Interface Screenshots

The Layer Splitting feature appears in the **Layer Fixing** tab with two modes:

1. **Fix Layers (Selective Dynamics)** - Add T/F flags for VASP
2. **Split into Layers** - Separate multi-layer structures

When in Split mode:
- Upload your multi-layer structure
- Set number of interfaces
- Click "Split into Layers"
- Download each layer individually

---

## FAQ

**Q: Do the layer files still have the same cell size?**  
A: Yes. The lattice parameters are **identical** to the input structure. Only atoms are filtered.

**Q: Can I split a 2-layer structure?**  
A: Yes, use `n_interfaces = 1` to split into 2 layers.

**Q: What about coordinates - Cartesian or Fractional?**  
A: The output VASP files use fractional coordinates (Direct mode), matching standard VASP format.

**Q: Can I use this with CP2K XYZ files?**  
A: Yes. The web interface accepts .xyz files with CP2K-style cell vectors.

**Q: Will this work for non-interface structures?**  
A: It's designed for stacked structures with clear gaps. For general layer detection, use the "Fix Layers" mode with `--by_z_layers`.

---

## Related Documentation

- [Main README](../README.md) - Overall project documentation
- [Quick Start](../QUICK_START.md) - Fast setup and usage
- [CLI Reference](../build_heterojunctions/README.md) - Command-line details

---

Use the contact information in the main README for questions or support.
