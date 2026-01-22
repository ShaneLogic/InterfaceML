# Perovskite Dataset Statistics

## Overview

**Collection Date**: November 2, 2025  
**Total Structures**: 9,232 CIF files  
**Compressed Size**: 2.8 MB  
**Uncompressed Size**: ~50 MB

## Dataset Breakdown

### Materials Project Perovskites
- **Total Structures**: 7,886 CIF files
- **Source**: Materials Project Database
- **Query Method**: Robocrystallographer keyword search
- **Original Query Results**: 8,509 material IDs
- **Successfully Downloaded**: 7,886 structures (92.7%)
- **Unavailable/Deprecated**: 623 structures (7.3%)

### Dryad HOIP Dataset
- **Total Structures**: 1,346 CIF files
- **Source**: Dryad Digital Repository
- **Publication**: Kim et al., Scientific Data (2017)
- **Coverage**: 
  - 16 organic cations
  - 3 metal cations (Ge, Sn, Pb)
  - 4 halide anions (F, Cl, Br, I)

## Composition Distribution

### Materials Project Dataset

**Major Element Categories**:
- Oxide perovskites (ABO₃): Majority
- Halide perovskites (ABX₃): Significant portion
- Mixed anion perovskites: Present
- Double perovskites (A₂BB'O₆): Present
- Layered perovskites: Present

**Common A-site Cations**:
- Alkali metals: K, Rb, Cs, Na, Li
- Alkaline earth: Ba, Sr, Ca
- Rare earth: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Y
- Other: Bi, Tl

**Common B-site Cations**:
- Transition metals: Ti, Zr, Hf, V, Nb, Ta, Cr, Mo, W, Mn, Fe, Ru, Os, Co, Rh, Ir, Ni, Pd, Pt, Cu, Ag, Au
- Main group: Al, Ga, In, Sn, Pb, Sb, Bi

**Common Anions**:
- Oxygen (O): Most common
- Halogens: F, Cl, Br, I
- Chalcogens: S, Se
- Mixed anions: O/F, O/S, etc.

### Dryad HOIP Dataset

**Organic Cations** (16 types):
- Methylammonium (MA): CH₃NH₃⁺
- Ethylammonium (EA): CH₃CH₂NH₃⁺
- Formamidinium (FA): CH(NH₂)₂⁺
- And 13 other organic cations

**Metal Cations** (3 types):
- Germanium (Ge)
- Tin (Sn)
- Lead (Pb)

**Halide Anions** (4 types):
- Fluoride (F⁻)
- Chloride (Cl⁻)
- Bromide (Br⁻)
- Iodide (I⁻)

**Example Compositions**:
- CH₃NH₃PbI₃ (MAPbI₃) - Classic perovskite solar cell material
- CH₃NH₃SnBr₃ (MASnBr₃) - Lead-free alternative
- CH(NH₂)₂PbI₃ (FAPbI₃) - High-efficiency solar cell material

## Property Coverage

### Materials Project Data
Properties available via API (not in CIF files):
- Crystal structure (in CIF)
- Space group symmetry (in CIF)
- Formation energy
- Band gap
- Density
- Elastic properties
- Dielectric properties
- Magnetic properties (where applicable)

### Dryad HOIP Data
Properties included in CIF files:
- **Bandgap (HSE06)**: High-accuracy hybrid functional
- **Bandgap (GGA)**: Standard DFT functional
- **Dielectric constant**: Electronic, ionic, and total
- **Refractive index**: Calculated from electronic dielectric
- **Atomization energy**: Per atom
- **Relative energies**: Two types (different reference states)
- **Density**: In g/cm³
- **Unit cell volume**: In Ų

**Property Ranges** (Dryad HOIP):
- Bandgap (HSE06): 1.2 - 6.5 eV
- Bandgap (GGA): 0.5 - 5.5 eV
- Dielectric constant: 3 - 25
- Refractive index: 1.5 - 3.5
- Density: 2.0 - 6.5 g/cm³

## Crystal Structure Information

### Space Groups
The dataset covers a wide range of space groups, including:
- Cubic: Pm-3m (#221), Fm-3m (#225), etc.
- Tetragonal: P4mm (#99), I4/mcm (#140), etc.
- Orthorhombic: Pnma (#62), Cmcm (#63), etc.
- Monoclinic: P2₁/c (#14), C2/c (#15), etc.
- Triclinic: P1 (#1), P-1 (#2)

### Structural Variants
- **3D Perovskites**: ABX₃ structure
- **2D Perovskites**: Layered structures (in Dryad dataset)
- **Quasi-2D**: Ruddlesden-Popper phases
- **Double Perovskites**: A₂BB'X₆
- **Ordered Perovskites**: Cation ordering on A or B sites

## Applications by Material Type

### Solar Cells
- Halide perovskites: ~1,500+ structures
- Hybrid organic-inorganic: 1,346 structures (Dryad)
- Lead-based: Majority of halides
- Lead-free alternatives: Sn, Ge-based

### Electronics and Ferroelectrics
- Oxide perovskites: ~5,000+ structures
- BaTiO₃ and derivatives
- PbTiO₃ and derivatives
- BiFeO₃ and multiferroics

### Catalysis
- Mixed metal oxides
- Transition metal perovskites
- Various oxidation states

### Optoelectronics
- Wide bandgap materials
- Tunable optical properties
- Light-emitting applications

## Data Quality Metrics

### Materials Project
- **Calculation Method**: DFT (VASP)
- **Functional**: GGA-PBE, GGA+U (where applicable)
- **Pseudopotentials**: PAW
- **k-point density**: High convergence
- **Energy cutoff**: Material-dependent, converged

### Dryad HOIP
- **Calculation Method**: DFT (VASP)
- **Functional**: GGA-PBE, HSE06 for bandgaps
- **Pseudopotentials**: PAW
- **Energy cutoff**: 400 eV
- **k-spacing**: 0.20/Å (relax), 0.15/Å (bandgap)
- **Convergence**: Systematic validation

## File Format Details

### CIF File Structure
- Standard Crystallographic Information File format
- Compatible with: VESTA, Mercury, Materials Studio, pymatgen, ASE
- Contains: Unit cell, atomic positions, symmetry operations
- Additional data: Comments with calculated properties (Dryad only)

### Naming Conventions
- **Materials Project**: `mp-XXXXXX.cif` (6-digit ID)
- **Dryad HOIP**: `XXXX.cif` (4-digit sequential number)

## Heterojunction Design Potential

### Lattice Matching Opportunities
The dataset includes perovskites with various lattice parameters, enabling:
- Low-strain interfaces (<2% mismatch)
- Moderate-strain interfaces (2-5% mismatch)
- High-strain interfaces (>5% mismatch)

### Band Alignment Possibilities
- Type-I heterojunctions (straddling gap)
- Type-II heterojunctions (staggered gap)
- Type-III heterojunctions (broken gap)

### Interface Orientations
Common perovskite surfaces for heterojunctions:
- (001): Most stable for cubic perovskites
- (110): Alternative orientation
- (111): Polar surfaces

## Comparison with Other Databases

| Database | Perovskite Structures | Format | Properties |
|----------|----------------------|---------|------------|
| This Dataset | 9,232 | CIF | Varied |
| Materials Project (total) | ~150,000 | API/CIF | Comprehensive |
| ICSD | ~10,000 perovskites | CIF | Experimental |
| COD | ~5,000 perovskites | CIF | Mixed |
| OQMD | ~8,000 perovskites | API | DFT-calculated |

## Update History

- **November 2, 2025**: Initial compilation
  - Downloaded 7,886 structures from Materials Project
  - Included 1,346 structures from Dryad HOIP dataset
  - Created comprehensive documentation

## Future Expansion Possibilities

1. **Additional Properties**: Download more properties from Materials Project API
2. **Experimental Structures**: Add ICSD experimental data
3. **Interface Models**: Pre-constructed heterojunction interfaces
4. **Machine Learning Features**: Extract features for ML models
5. **2D Perovskites**: Expand coverage of layered materials

## Data Integrity

- **MD5 Checksums**: Available upon request
- **Validation**: All CIF files tested for readability
- **Duplicates**: Minimal overlap between MP and Dryad datasets
- **Deprecated Materials**: Excluded from final dataset

---

**For detailed usage instructions, see `QUICK_START.md`**  
**For comprehensive documentation, see `PEROVSKITE_DATASET_README.md`**
