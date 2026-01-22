# Perovskite Heterojunction Dataset - CIF Files Collection

## Overview

This dataset contains **9,232 perovskite crystal structures** in CIF (Crystallographic Information File) format, retrieved from two major sources:

1. **Materials Project Database**: 7,886 structures
2. **Dryad HOIP Dataset**: 1,346 hybrid organic-inorganic perovskite structures

## Dataset Contents

### 1. Materials Project Perovskite Structures (7,886 structures)

**Source**: Materials Project Database (https://materialsproject.org/)  
**Query Method**: Robocrystallographer keyword search for "perovskite"  
**Location**: `mp_perovskite_cifs/`

**Characteristics**:
- Includes oxide perovskites, halide perovskites, and other perovskite-type structures
- Covers a wide range of compositions: ABX₃, A₂BB'X₆, layered perovskites, etc.
- Contains both inorganic and hybrid organic-inorganic perovskites
- DFT-calculated structures with optimized geometries
- Includes various perovskite polymorphs and derivatives

**File Naming**: `mp-XXXXXX.cif` (where XXXXXX is the Materials Project ID)

**Additional Data**:
- Material IDs list: `mp_perovskite_ids.txt` (8,509 total IDs)
- Metadata: `mp_perovskite_metadata.txt` (formula, space group information)

### 2. Dryad Hybrid Organic-Inorganic Perovskite Dataset (1,346 structures)

**Source**: Dryad Digital Repository  
**DOI**: https://doi.org/10.5061/dryad.gq3rg  
**Publication**: Kim et al., Scientific Data 4, 170057 (2017)  
**Location**: `cif_merge/`

**Characteristics**:
- 16 organic cations (e.g., CH₃NH₃⁺, CH₃CH₂NH₃⁺, etc.)
- 3 group-IV cations (Ge, Sn, Pb)
- 4 halide anions (F, Cl, Br, I)
- DFT-optimized structures (VASP, PAW pseudopotentials)
- Each CIF file includes calculated properties:
  - Bandgap (HSE06 and GGA)
  - Dielectric constant (electronic, ionic, total)
  - Refractive index
  - Atomization energy
  - Relative energies
  - Density

**File Naming**: `XXXX.cif` (where XXXX is a sequential number 0001-1346)

**Example Properties** (from 0001.cif - Ethylammonium Germanium Fluoride):
```
Formula: CH₃CH₂NH₃GeF₃
Bandgap (HSE06): 5.9448 eV
Bandgap (GGA): 4.7308 eV
Dielectric constant (total): 5.3973
Refractive index: 1.6724
Density: 2.5647 g/cm³
```

## Heterojunction Relevance

While this dataset contains **bulk perovskite structures** rather than pre-constructed heterojunction interfaces, these structures are highly relevant for heterojunction research:

### Potential Applications:

1. **Type-II Heterojunctions**: Combine perovskites with different band gaps for charge separation
2. **Perovskite/Perovskite Heterojunctions**: Interface two different perovskite compositions
3. **Phase Heterojunctions**: Utilize different polymorphs of the same perovskite
4. **Dimensional Heterojunctions**: 3D/2D or 3D/quasi-2D perovskite interfaces
5. **Graded Heterojunctions**: Compositional gradients using multiple perovskite structures

### Construction Methods:

To create heterojunction models from these structures:
- Use computational tools (VESTA, ASE, pymatgen) to build interface models
- Stack two different perovskite structures along specific crystallographic planes
- Optimize the interface geometry using DFT calculations
- Consider lattice matching and strain effects

## File Structure

```
/home/ubuntu/
├── mp_perovskite_cifs/          # Materials Project structures (7,886 files)
│   ├── mp-1517942.cif
│   ├── mp-1222721.cif
│   └── ...
├── cif_merge/                    # Dryad HOIP structures (1,346 files)
│   ├── 0001.cif
│   ├── 0002.cif
│   └── ...
├── mp_perovskite_ids.txt        # All Materials Project perovskite IDs (8,509 entries)
├── mp_perovskite_metadata.txt   # Metadata for downloaded MP structures
├── HOIP_cif.tar                 # Original Dryad dataset archive
└── PEROVSKITE_DATASET_README.md # This file
```

## Data Quality and Validation

### Materials Project Data:
- All structures are DFT-calculated and optimized
- Peer-reviewed computational methodology
- Consistent calculation parameters across the database
- Regular updates and corrections

### Dryad HOIP Data:
- Validated against experimental data where available
- Systematic DFT calculations with uniform parameters
- Published in peer-reviewed journal (Scientific Data)
- Includes energy convergence criteria

## Usage Examples

### Loading CIF Files with Python (pymatgen):

```python
from pymatgen.core import Structure

# Load a Materials Project structure
structure = Structure.from_file("mp_perovskite_cifs/mp-5986.cif")
print(structure)

# Load a Dryad HOIP structure
hoip_structure = Structure.from_file("cif_merge/0001.cif")
print(hoip_structure)
```

### Creating a Simple Heterojunction:

```python
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

# Load two different perovskites
perov1 = Structure.from_file("mp_perovskite_cifs/mp-5986.cif")  # BaTiO3
perov2 = Structure.from_file("mp_perovskite_cifs/mp-1977728.cif")  # NaNbO3

# Generate slabs
slab_gen1 = SlabGenerator(perov1, (0, 0, 1), min_slab_size=10, min_vacuum_size=15)
slab_gen2 = SlabGenerator(perov2, (0, 0, 1), min_slab_size=10, min_vacuum_size=15)

slab1 = slab_gen1.get_slab()
slab2 = slab_gen2.get_slab()

# Stack slabs to create heterojunction (requires additional processing)
```

## Filtering for Specific Applications

### Halide Perovskites (Solar Cells):
Look for structures containing Pb, Sn, or Cs with I, Br, or Cl

### Oxide Perovskites (Electronics):
Search for structures with Ti, Zr, Nb, or Ta with oxygen

### Hybrid Perovskites (Optoelectronics):
Use the Dryad dataset (`cif_merge/`) which focuses on organic-inorganic hybrids

## Citations

### Materials Project:
```
Jain, A. et al. Commentary: The Materials Project: A materials genome approach to 
accelerating materials innovation. APL Materials 1, 011002 (2013).
DOI: 10.1063/1.4812323
```

### Dryad HOIP Dataset:
```
Kim, C., Huan, T. D., Krishnan, S. & Ramprasad, R. A hybrid organic-inorganic 
perovskite dataset. Scientific Data 4, 170057 (2017).
DOI: 10.1038/sdata.2017.57
```

### Robocrystallographer (MP Keyword Search):
```
Ganose, A. M., Jain, A. Robocrystallographer: automated crystal structure text 
descriptions and analysis. MRS Communications 9, 874–881 (2019).
DOI: 10.1557/mrc.2019.94
```

## Notes and Limitations

1. **Not Pre-constructed Heterojunctions**: These are bulk crystal structures, not interface models
2. **Deprecated Materials**: Some Material IDs from the original query (8,509) were not available for download (623 structures), likely due to deprecation or updates
3. **Computational vs. Experimental**: All structures are computationally optimized; experimental structures may differ slightly
4. **Heterojunction Construction**: Users need to manually construct heterojunction interfaces from these bulk structures
5. **Lattice Mismatch**: When creating heterojunctions, consider lattice matching and strain effects

## Additional Resources

- **Materials Project Website**: https://materialsproject.org/
- **Dryad Repository**: https://datadryad.org/dataset/doi:10.5061/dryad.gq3rg
- **Khazana Repository**: http://khazana.uconn.edu/
- **Pymatgen Documentation**: https://pymatgen.org/
- **ASE (Atomic Simulation Environment)**: https://wiki.fysik.dtu.dk/ase/

## Contact and Support

For questions about:
- **Materials Project data**: https://matsci.org/
- **Dryad HOIP dataset**: Contact authors via publication
- **This compilation**: Refer to the query methods documented in this README

---

**Dataset Compiled**: November 2, 2025  
**Total Structures**: 9,232 CIF files  
**Total Size**: ~15 MB (compressed), ~50 MB (uncompressed)
