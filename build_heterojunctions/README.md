# Auto Interface Builder (and related tools)

This folder contains utilities to build heterojunction interfaces and prepare inputs for DFT relaxation, plus a helper script to generate **difference charge density** from CP2K electron-density `.cube` files (visualizable in **VESTA**).

## Contents

- [Quick start (recommended workflow)](#quick-start-recommended-workflow)
- [1) Build heterojunction interface (`interface_builder.py`)](#1-build-heterojunction-interface-interface_builderpy)
- [2) Add selective dynamics (`fix_interface_layers.py`)](#2-add-selective-dynamics-fix_interface_layerspy)
- [3) Split an existing heterojunction into layers (for differential charge)](#3-split-an-existing-heterojunction-into-layers-for-differential-charge)
- [4) Difference charge density from CP2K cubes (`delta_density_cube.py`)](#4-difference-charge-density-from-cp2k-cubes-delta_density_cubepy)
- [5) Stack pre-relaxed slabs (alternative) (`stack_slabs_relax.py`)](#5-stack-pre-relaxed-slabs-alternative-stack_slabs_relaxpy)
- [Tips](#tips)
- [Troubleshooting](#troubleshooting)

## Quick start (recommended workflow)

```bash
# 1) Build interface
python build_heterojunctions/interface_builder.py \
  --a <structure_A.(cif|vasp)> \
  --b <structure_B.(cif|vasp)> \
  --miller_a <h,k,l> \
  --miller_b <h,k,l> \
  --use_builder_interface \
  --max_atoms 300

# 2) Add selective dynamics
python build_heterojunctions/fix_interface_layers.py \
  <A>@<B>.vasp \
  -o <A>@<B>_relaxed.vasp \
  -n 2 \
  -t 2.0
```

## 1) Build heterojunction interface (`interface_builder.py`)

### Basic usage

```bash
python build_heterojunctions/interface_builder.py \
  --a structures/perovskites/H6PbCI3N.cif \
  --b structures/etl/TiO2.cif \
  --miller_a 0,0,1 \
  --miller_b 1,0,1 \
  --vacuum 20 \
  --sep 3.2 \
  --strain_target A \
  --use_builder_interface \
  --max_atoms 300
```

### Required arguments

- `--a`: Path to structure file A (CIF/POSCAR format)
- `--b`: Path to structure file B (CIF/POSCAR format)
- `--miller_a`: Miller indices for material A (e.g., `0,0,1`)
- `--miller_b`: Miller indices for material B (e.g., `1,0,1`)

### Common optional arguments

- `--slab_thickness_a`: Slab thickness for A in Å (default: 20.0)
- `--slab_thickness_b`: Slab thickness for B in Å (default: 12.0)
- `--vacuum`: Vacuum size in Å (default: 20.0)
- `--sep`: Initial separation between surfaces in Å (default: 3.2)
- `--tol`: Matching strain tolerance (default: 0.03)
- `--max_area`: Maximum interface area in Å² for searching matches (default: 800.0)
- `--strain_target`: Which material to strain: `A`, `B`, or `both` (default: `A`)
- `--use_builder_interface`: Use CoherentInterfaceBuilder method (recommended for ordered interfaces)
- `--max_atoms`: Maximum number of atoms in final structure (default: 400)

### Outputs

The script generates three POSCAR files (base name is `<A>@<B>` derived from filenames):

1. `<A>@<B>_bottom_strained.vasp` - Bottom slab (substrate)
2. `<A>@<B>_top_strained.vasp` - Top slab (film)
3. `<A>@<B>.vasp` - Combined heterojunction structure (input for Step 2)

## 2) Add selective dynamics (`fix_interface_layers.py`)

### Basic usage

```bash
python build_heterojunctions/fix_interface_layers.py \
  <A>@<B>.vasp \
  -o <A>@<B>_relaxed.vasp \
  -n 2 \
  -t 2.0
```

### Parameters

- `input_file`: Input POSCAR file (combined heterojunction from Step 1)
- `-o, --output`: Output POSCAR file (default: input_file with `_relaxed` suffix)
- `-n, --n_layers`: Number of layers to relax on each side of interface (default: 1)
- `-t, --thickness`: Layer thickness threshold in Å (default: 2.0)
- `-m, --method`: Interface identification method: `density_gap`, `median`, or `max_gap` (default: `density_gap`)

### Selective dynamics format (in output POSCAR)

```
Selective dynamics
direct
  0.1234  0.5678  0.9012  T T T  Ti4+
  0.2345  0.6789  0.0123  F F F  O2-
  ...
```

- `T T T`: Atom is free to relax (interface layers)
- `F F F`: Atom is fixed (bulk layers)

## 3) Split an existing heterojunction into layers (for differential charge)

If you already have a heterojunction model (e.g. a CIF exported from Multiwfn) and need to split it into independent layers while keeping the **relative positions inside the unit cell unchanged**, use:

```bash
python build_heterojunctions/interface_builder.py \
  --split_layers \
  --input structures/heterojunctions/PROJECT-1.cif
```

### Optional parameters

- `--out_base_dir`: Base directory to place the output folder `<input_stem>/` (default: input's parent directory)
- `--layer1_elements`: Comma-separated element symbols for layer 1 (e.g., `Ti,O`)
- `--layer2_elements`: Comma-separated element symbols for layer 2 (optional; inferred if not provided)

### Outputs

For an input file `PROJECT-1.cif`, the script creates:

- Output folder: `structures/heterojunctions/PROJECT-1/`
- A copy of the original file: `PROJECT-1.cif`
- Layer structures (VASP): `PROJECT-1_layer1.vasp`, `PROJECT-1_layer2.vasp`, and `PROJECT-1.vasp`
- Layer structures (CIF): `PROJECT-1_layer1.cif`, `PROJECT-1_layer2.cif`, and `PROJECT-1_combined.cif`

### Important notes (position preservation)

- The split-layer mode is designed for differential charge workflows: it does **not** translate, center, or wrap fractional coordinates to `[0, 1)`.
- If `--layer1_elements/--layer2_elements` are not provided, the script tries to auto-detect common oxide/perovskite interfaces (e.g. `Ti/O` as one layer).

## 4) Difference charge density from CP2K cubes (`delta_density_cube.py`)

Compute difference charge density (best for heterojunctions):

- **Formula**: Δρ(r) = ρ_interface − ρ_layerA − ρ_layerB
- **Output**: `delta-density.cube` (Gaussian cube format; open directly in **VESTA**)

### Example (PROJECT-1)

```bash
python build_heterojunctions/delta_density_cube.py \
  --interface structures/heterojunctions/PROJECT-1/mapbi3_tio2-ELECTRON_DENSITY-1_0.cube \
  --layerA structures/heterojunctions/PROJECT-1/mapbi3-ELECTRON_DENSITY-1_0.cube \
  --layerB structures/heterojunctions/PROJECT-1/tio2-ELECTRON_DENSITY-1_0.cube \
  --output structures/heterojunctions/PROJECT-1/delta-density.cube
```

### Notes

- The three input `.cube` files must share the **same volumetric grid** (grid size + origin + 3 axis lines). The script tolerates small rounding differences in header floats (common in CP2K output).
- The implementation is **streaming** (constant memory) and suitable for very large cube files.

## 5) Stack pre-relaxed slabs (alternative) (`stack_slabs_relax.py`)

If you have pre-relaxed slab structures and want to stack them with a specific gap, you can use `stack_slabs_relax.py`:

### Basic Usage

```bash
python build_heterojunctions/stack_slabs_relax.py <slab1_file> <slab2_file> [gap_in_angstrom] [vacuum_in_angstrom] [output_file]
```

### Parameters

- `slab1_file`: Path to first slab file (bottom slab)
- `slab2_file`: Path to second slab file (top slab)
- `gap_in_angstrom`: Gap between two slabs in Angstroms (default: 3.0)
- `vacuum_in_angstrom`: Total vacuum layer thickness in Angstroms (default: 20.0), split equally between bottom and top
- `output_file`: Output filename (optional), if not specified will be auto-generated and saved to `relax_slab/` folder

### Examples

**Example 1**: Stack two pre-relaxed slabs with 3 Å gap and 20 Å vacuum

```bash
python build_heterojunctions/stack_slabs_relax.py \
    relax_slab/fapbi3_slab_relax.vasp \
    relax_slab/tio2_fa_slab_relax.vasp \
    3.0 \
    20.0
```

This will create `relax_slab/fapbi3@tio2_fa_stacked.vasp` with structure:
- Bottom vacuum (10 Å, half of total 20 Å)
- Slab 1 (fapbi3)
- Gap (3 Å)
- Slab 2 (tio2_fa)
- Top vacuum (10 Å, half of total 20 Å)

**Example 2**: Stack slabs with custom gap and vacuum

```bash
python build_heterojunctions/stack_slabs_relax.py \
    relax_slab/fapbi3_slab_relax.vasp \
    relax_slab/tio2_fa_slab_relax.vasp \
    5.0 \
    15.0
```

**Example 3**: Stack with different spacing and custom output name

```bash
python build_heterojunctions/stack_slabs_relax.py \
    relax_slab/mapbi3_slab_relax.vasp \
    relax_slab/tio2_ma_slab_relax.vasp \
    3.2 \
    20.0 \
    custom_name.vasp
```

This will create `relax_slab/custom_name.vasp` (still saved in relax_slab folder)

### How It Works

1. Reads two VASP-format slab files
2. Calculates the thickness of each slab in the z-direction
3. Creates a structure with the following layout:
   - Bottom vacuum layer
   - Slab 1 (bottom slab)
   - Gap between slabs
   - Slab 2 (top slab)
   - Top vacuum layer
4. Adjusts z-coordinates of both slabs to fit in the new structure
5. Merges atomic coordinates and element information from both slabs
6. Updates the z-component of lattice vectors to accommodate both slabs, gap, and vacuum layers
7. Outputs the combined VASP file

### Output Information

The script displays:
- z-coordinate range and thickness for each slab
- Total number of atoms
- New lattice z-direction length
- Gap between the two slabs

### Notes

- Both slabs should have the same (or compatible) xy-plane lattice vectors
- Output file uses Direct coordinates (fractional coordinates)
- The script automatically merges elements of the same type
- Output file includes velocity information (initialized to zero)
- **Output location**: By default, stacked structures are saved to the `relax_slab/` folder
- **File naming**: Output files are named as `{slab1}@{slab2}_stacked.vasp` (e.g., `fapbi3@tio2_fa_stacked.vasp`)
- **Duplicate handling**: If a file with the same name already exists, a number suffix will be added (e.g., `fapbi3@tio2_fa_stacked_1.vasp`)
- **Vacuum layers**: The structure includes vacuum layers at both bottom and top. The specified vacuum thickness is the total (default: 20 Å), which is split equally between bottom and top (10 Å each) to ensure proper isolation for DFT calculations
- **Structure layout**: The final structure follows: [bottom vacuum] → [slab1] → [gap] → [slab2] → [top vacuum]

### Use Cases

This tool is useful when:
- You have pre-relaxed slab structures from separate calculations
- You want to create heterojunctions by stacking relaxed slabs
- You need precise control over the gap between slabs
- You want to combine slabs without going through the full interface builder workflow

## Tips

1. **Use `--use_builder_interface`**: Recommended for ordered, high-symmetry interfaces
2. **Atom count control**: Use `--max_atoms` to limit system size (the script may adjust thickness if needed)
3. **Selective dynamics**: `-n 2` (2 layers per side) is a good default for many DFT relaxations
4. **Layer thickness**: Tune `-t` (often 2.0–3.0 Å) based on your material’s layer spacing
5. **Strain target**:
   - `A`: Strain material A to match B (good when A is more flexible)
   - `B`: Strain material B to match A (good when B is more flexible)
   - `both`: Split strain evenly (balanced)

## Troubleshooting

- **No matches found**: Increase `--tol` or `--max_area`
- **Too many atoms**: Reduce `--max_atoms`, `--slab_thickness_a/b`, or `--max_area`
- **Poor matching**: Try different `--miller_a` and `--miller_b` combinations
- **Interface layers not relaxed**: Check `-n` and `-t` parameters in `fix_interface_layers.py`
- **Interface identification failed**: Try different `-m` method (`density_gap`, `median`, or `max_gap`)
- **Difference-density cube errors**: Ensure the three `.cube` files share the same grid and correspond to (interface, layerA, layerB)
