# Quick Start Guide - Perovskite Dataset

## Dataset Summary

- **Total Structures**: 9,232 perovskite CIF files
- **Sources**: Materials Project (7,886) + Dryad HOIP (1,346)
- **Format**: Crystallographic Information File (CIF)
- **Applications**: Heterojunction design, solar cells, optoelectronics, catalysis

## Directory Structure

```
perovskite_dataset/
├── mp_perovskite_cifs/          # 7,886 Materials Project structures
├── cif_merge/                    # 1,346 Dryad HOIP structures
├── mp_perovskite_ids.txt        # List of all MP perovskite IDs
├── mp_perovskite_metadata.txt   # Metadata (formula, space group)
├── PEROVSKITE_DATASET_README.md # Detailed documentation
└── QUICK_START.md               # This file
```

## Quick Access Examples

### 1. Browse CIF Files

**Materials Project structures:**
```bash
cd mp_perovskite_cifs/
ls *.cif | head -10
```

**Dryad HOIP structures:**
```bash
cd cif_merge/
ls *.cif | head -10
```

### 2. View a CIF File

```bash
# View with text editor
cat mp_perovskite_cifs/mp-5986.cif

# Or for HOIP dataset
cat cif_merge/0001.cif
```

### 3. Search for Specific Elements

**Find all structures containing Lead (Pb):**
```bash
grep -l "Pb" mp_perovskite_cifs/*.cif | wc -l
```

**Find halide perovskites (containing I, Br, or Cl):**
```bash
grep -l "_atom_site_type_symbol.*[IBr|Cl]" mp_perovskite_cifs/*.cif
```

### 4. Python Usage (pymatgen)

**Install pymatgen:**
```bash
pip install pymatgen
```

**Load and analyze a structure:**
```python
from pymatgen.core import Structure

# Load BaTiO3 (mp-5986)
structure = Structure.from_file("mp_perovskite_cifs/mp-5986.cif")

# Print basic information
print(f"Formula: {structure.composition.reduced_formula}")
print(f"Space Group: {structure.get_space_group_info()}")
print(f"Lattice: {structure.lattice}")
print(f"Volume: {structure.volume:.2f} Å³")

# Get atomic positions
for site in structure:
    print(f"{site.specie}: {site.frac_coords}")
```

### 5. Visualize Structures

**Using VESTA (recommended):**
1. Download VESTA: https://jp-minerals.org/vesta/en/
2. Open CIF file: File → Open → Select .cif file
3. Visualize crystal structure in 3D

**Using ASE (Python):**
```python
from ase.io import read
from ase.visualize import view

atoms = read("mp_perovskite_cifs/mp-5986.cif")
view(atoms)
```

## Common Use Cases

### Finding Specific Perovskite Types

**1. Halide Perovskites (e.g., for solar cells):**
```bash
# Search for Pb-I perovskites
grep -l "Pb" mp_perovskite_cifs/*.cif | xargs grep -l "I" | head -20
```

**2. Oxide Perovskites (e.g., for electronics):**
```bash
# Search for Ti-O perovskites
grep -l "Ti" mp_perovskite_cifs/*.cif | xargs grep -l "O" | head -20
```

**3. Hybrid Organic-Inorganic Perovskites:**
```bash
# All files in cif_merge/ are hybrid perovskites
ls cif_merge/*.cif | wc -l
```

### Extracting Properties from Dryad HOIP Dataset

```python
import re

def extract_properties(cif_file):
    """Extract calculated properties from Dryad HOIP CIF files"""
    with open(cif_file, 'r') as f:
        content = f.read()
    
    properties = {}
    
    # Extract bandgap
    match = re.search(r'Bandgap, HSE06 \(eV\):\s+([\d.]+)', content)
    if match:
        properties['bandgap_hse06'] = float(match.group(1))
    
    # Extract dielectric constant
    match = re.search(r'Dielectric constant, total:\s+([\d.]+)', content)
    if match:
        properties['dielectric_total'] = float(match.group(1))
    
    # Extract formula
    match = re.search(r'Label:\s+(.+)', content)
    if match:
        properties['name'] = match.group(1).strip()
    
    return properties

# Example usage
props = extract_properties("cif_merge/0001.cif")
print(props)
```

### Creating a Heterojunction Model

```python
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

# Load two perovskite structures
perov_a = Structure.from_file("mp_perovskite_cifs/mp-5986.cif")  # BaTiO3
perov_b = Structure.from_file("mp_perovskite_cifs/mp-1977728.cif")  # NaNbO3

# Create (001) surface slabs
slab_gen_a = SlabGenerator(perov_a, (0, 0, 1), min_slab_size=15, min_vacuum_size=20)
slab_gen_b = SlabGenerator(perov_b, (0, 0, 1), min_slab_size=15, min_vacuum_size=20)

slab_a = slab_gen_a.get_slab()
slab_b = slab_gen_b.get_slab()

# Save slabs for further processing
slab_a.to(filename="slab_a.cif")
slab_b.to(filename="slab_b.cif")

print(f"Slab A: {slab_a.composition.reduced_formula}")
print(f"Slab B: {slab_b.composition.reduced_formula}")
```

## Filtering and Analysis Scripts

### Count structures by element:

```python
import os
from pymatgen.core import Structure

def count_by_element(directory, element):
    """Count structures containing a specific element"""
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.cif'):
            try:
                struct = Structure.from_file(os.path.join(directory, filename))
                if element in [str(el) for el in struct.composition.elements]:
                    count += 1
            except:
                pass
    return count

# Count Pb-containing perovskites
pb_count = count_by_element("mp_perovskite_cifs", "Pb")
print(f"Structures containing Pb: {pb_count}")
```

### Generate statistics:

```python
from pymatgen.core import Structure
import os

def analyze_dataset(directory, max_files=100):
    """Analyze basic statistics of the dataset"""
    
    formulas = []
    space_groups = []
    volumes = []
    
    for i, filename in enumerate(os.listdir(directory)):
        if i >= max_files:
            break
        if filename.endswith('.cif'):
            try:
                struct = Structure.from_file(os.path.join(directory, filename))
                formulas.append(struct.composition.reduced_formula)
                space_groups.append(struct.get_space_group_info()[0])
                volumes.append(struct.volume)
            except:
                pass
    
    print(f"Analyzed {len(formulas)} structures")
    print(f"Unique formulas: {len(set(formulas))}")
    print(f"Unique space groups: {len(set(space_groups))}")
    print(f"Average volume: {sum(volumes)/len(volumes):.2f} Å³")
    
    return formulas, space_groups, volumes

# Analyze Materials Project dataset
analyze_dataset("mp_perovskite_cifs", max_files=100)
```

## Tips and Best Practices

1. **Start Small**: Test your analysis on a subset of structures before processing all 9,232 files
2. **Check CIF Validity**: Some CIF files may have formatting issues; use try-except blocks
3. **Lattice Matching**: For heterojunctions, check lattice parameters to minimize strain
4. **Property Extraction**: Dryad HOIP files contain rich metadata in comments
5. **Visualization**: Always visualize structures before using them in calculations
6. **Version Control**: Keep track of which structures you use in your research

## Common Issues and Solutions

**Issue**: CIF file won't load in pymatgen
- **Solution**: Try using ASE or manually inspect the file for formatting errors

**Issue**: Too many structures to analyze
- **Solution**: Filter by composition, space group, or other criteria first

**Issue**: Need specific properties not in CIF
- **Solution**: Use Materials Project API to fetch additional data using material IDs

## Next Steps

1. Read the full `PEROVSKITE_DATASET_README.md` for detailed information
2. Explore the Materials Project website for additional properties
3. Check the Dryad repository for the original HOIP dataset paper
4. Use computational tools (VASP, Quantum ESPRESSO) for DFT calculations on heterojunctions

## Support and Resources

- **Pymatgen Documentation**: https://pymatgen.org/
- **Materials Project Forum**: https://matsci.org/
- **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/
- **VESTA Manual**: https://jp-minerals.org/vesta/en/

---

**Happy researching!** 🔬✨
