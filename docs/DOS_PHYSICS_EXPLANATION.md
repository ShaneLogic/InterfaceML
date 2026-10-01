# Understanding DOS and PDOS in CP2K: Physics and Data Processing

> **Author**: InterfaceML Documentation  
> **Date**: January 2026  
> **Purpose**: Explain the physical meaning of TDOS/PDOS and the data transformations applied

---

## Table of Contents

1. [Overview of Density of States](#1-overview-of-density-of-states)
2. [CP2K Output File Formats](#2-cp2k-output-file-formats)
3. [The TDOS Scaling Problem and Solution](#3-the-tdos-scaling-problem-and-solution)
4. [PDOS: No Transformation Needed](#4-pdos-no-transformation-needed)
5. [Physical Interpretation](#5-physical-interpretation)
6. [Summary](#6-summary)

---

## 1. Overview of Density of States

### What is DOS?

The **Density of States (DOS)** describes the number of electronic states available at each energy level in a material. Mathematically:

$$g(E) = \sum_n \delta(E - \epsilon_n)$$

where $\epsilon_n$ is the energy of the $n$-th electronic state (molecular orbital in finite systems).

In practice, the delta functions are broadened with Gaussian functions for visualization:

$$g(E) = \sum_n \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(E-\epsilon_n)^2}{2\sigma^2}\right)$$

### Types of DOS

| Type | Definition | Physical Meaning |
|------|------------|------------------|
| **TDOS** | Total Density of States | All electronic states in the system |
| **PDOS** | Projected Density of States | States projected onto specific atoms or orbitals |

**Key relationship**: TDOS = sum of all PDOS contributions (when projected onto all atoms).

---

## 2. CP2K Output File Formats

### 2.1 TDOS File Structure (`.dos`)

```
# DOS at iteration step i = 0, E_Fermi[a.u.] = -0.012813
#  Energy[a.u.]       Density     Occupation
    -0.84330575         0.0054         8.0000
    -0.83330575         0.0041         6.0000
    ...
    -0.01330575         0.0041         5.9997   <- Near Fermi level
     0.00669425         0.0054         0.0000   <- Above Fermi level
     0.01669425         0.0082         0.0000
```

**Column meanings**:

| Column | Name | Description | Units |
|--------|------|-------------|-------|
| 1 | Energy | Energy grid points | Hartree (a.u.) |
| 2 | Density | Normalized DOS density | Arbitrary |
| 3 | Occupation | Electron count in this energy bin | Electrons |

**Critical observation**: The `Occupation` column is **zero for unoccupied states** (E > E_F) because there are no electrons in the conduction band at T=0K.

### 2.2 PDOS File Structure (`.pdos`)

```
# Projected DOS for list 2 of 60 atoms, E(Fermi) = -0.012813 a.u.
#  MO  Eigenvalue[a.u.]  Occupation    s       py      pz      px     d-2    d-1     d0    d+1    d+2
   1      -0.843306       2.000000   0.0000  0.0000  0.0000  0.0000  0.0000 0.0000 0.0000 0.0000 0.0000
  29      -0.718395       2.000000   0.7602  0.0550  0.0463  0.0573  0.0037 0.0193 0.0355 0.0188 0.0036
 ...
 500       0.150000       0.000000   0.1234  0.0456  0.0789  0.0321  0.0012 0.0034 0.0056 0.0078 0.0090
```

**Column meanings**:

| Column | Name | Description |
|--------|------|-------------|
| 1 | MO Index | Molecular orbital number |
| 2 | Eigenvalue | MO energy (Hartree) |
| 3 | Occupation | 2.0 (occupied) or 0.0 (unoccupied) |
| 4+ | s, p, d... | Projection weights onto atomic orbitals |

**Key difference**: The projection weights (s, p, d columns) are **non-zero for both occupied AND unoccupied states**.

---

## 3. The TDOS Scaling Problem and Solution

### 3.1 The Problem

When plotting TDOS vs PDOS, we observed:

1. **TDOS was much smaller than PDOS** (by ~5000x)
2. **TDOS was zero for E > E_F** (unoccupied region)

### 3.2 Root Cause Analysis

The `Density` column in CP2K's TDOS file is **not in physical units**. Let's verify:

```
At E = -0.133 a.u. (peak region):
- Density = 0.0312
- Occupation = 46 electrons
- Delta E = 0.01 a.u. = 0.272 eV

Expected DOS = Occupation / Delta E = 46 / 0.272 = 169 states/eV
Actual Density = 0.0312

Ratio = 169 / 0.0312 = 5418x !!
```

The `Density` column has some arbitrary normalization, making direct comparison impossible.

### 3.3 The Solution: Scaling Factor Derivation

We derive a scaling factor from the **occupied region** where both `Density` and `Occupation` are meaningful:

$$\text{Scale Factor} = \text{median}\left(\frac{\text{Occupation}_i / \Delta E}{\text{Density}_i}\right) \quad \text{for } E_i < E_F$$

Then apply to the entire energy range:

$$\text{TDOS}(E) = \text{Density}(E) \times \text{Scale Factor}$$

### 3.4 Physical Justification

This transformation is **physically valid** because:

1. **Shape preservation**: The `Density` column correctly captures the **relative** DOS at each energy
2. **Magnitude correction**: The scaling factor converts arbitrary units to **states/eV**
3. **Continuity**: Unlike `Occupation`, the `Density` column is non-zero for unoccupied states

**Mathematical equivalence** (in the occupied region):

$$\text{TDOS}_{\text{scaled}}(E) = \text{Density}(E) \times \frac{\text{Occupation}(E)/\Delta E}{\text{Density}(E)} = \frac{\text{Occupation}(E)}{\Delta E}$$

This confirms our scaling gives the correct physical DOS.

---

## 4. PDOS: No Transformation Needed

### 4.1 Why PDOS is Different

The PDOS file provides **discrete molecular orbital data**, not binned densities:

| Property | TDOS File | PDOS File |
|----------|-----------|-----------|
| Data type | Energy bins | Discrete MOs |
| Key quantity | Occupation (electron count) | Projection weights |
| Unoccupied states | Occupation = 0 | Weights != 0 |

### 4.2 PDOS Calculation

The projected DOS is computed by Gaussian broadening:

$$\text{PDOS}(E) = \sum_n w_n \cdot G(E - \epsilon_n, \sigma)$$

where:
- $n$ = molecular orbital index
- $\epsilon_n$ = MO energy
- $w_n = \sum_l |c_{n,l}|^2$ = total projection weight (sum of s, p, d... components)
- $G(x, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-x^2/2\sigma^2)$

### 4.3 Physical Meaning of Projection Weights

The projection weight $w_n$ represents **how much of MO $n$ is localized on the selected atoms**.

Example from the data:
```
MO #29 (occupied, E = -0.718 a.u.):
  s = 0.7602, p = 0.1586, d = 0.0809
  Total = 0.9997 (almost entirely on these 60 atoms)

MO #500 (unoccupied, E = +0.15 a.u.):
  Total weight = 0.15 (partially on these atoms)
```

**This is independent of electron occupation** - it describes the spatial distribution of the wavefunction, not whether electrons are present.

### 4.4 Verification: Unoccupied States Have Weights

```python
# From actual data analysis:
Occupied MOs (437):   avg weight = 0.2745
Unoccupied MOs (309): avg weight = 0.1576  # NOT ZERO!
```

---

## 5. Physical Interpretation

### 5.1 What the Plot Shows

When we plot TDOS + PDOS together:

```
              TDOS (black)
               down
     |    ___/\___
     |   /        \
DOS  |  /   ___    \___/\
     | /   /   \        
     |/___/     \__PDOS (colored)
     +-------------------> Energy
           E_F up
```

- **TDOS** = Total available states (should be higher than individual PDOS)
- **PDOS** = States localized on specific atom groups
- **Relationship**: $\sum_{\text{all atoms}} \text{PDOS} = \text{TDOS}$

### 5.2 Per-Atom Normalization

When using "per-atom" scaling:

$$\text{PDOS}_{\text{per-atom}}(E) = \frac{\text{PDOS}(E)}{N_{\text{atoms}}}$$

This allows comparison between groups of different sizes:
- List 1: 180 atoms (e.g., perovskite layer)
- List 2: 60 atoms (e.g., C60 molecule)

Without normalization, the 180-atom group would always appear larger.

### 5.3 Occupied vs Unoccupied Regions

| Region | Physical Meaning | What we see |
|--------|------------------|-------------|
| E < E_F | Valence band (filled states) | Where electrons are |
| E > E_F | Conduction band (empty states) | Where electrons **could** go |
| Near E_F | Band edges | Electronic structure near gap |

---

## 6. Summary

### Key Takeaways

| Aspect | TDOS | PDOS |
|--------|------|------|
| **Raw data format** | Energy bins with Occupation | Discrete MOs with projection weights |
| **Problem** | Occupation = 0 for E > E_F | None |
| **Solution** | Scale Density column | Direct Gaussian broadening |
| **Physical units** | states/eV | states/eV |
| **Transformation valid?** | Yes (magnitude only) | Not needed |

### The Transformation is Physically Meaningful

1. **Does not change the shape** of the DOS curve
2. **Only corrects the magnitude** to physical units (states/eV)
3. **Preserves the unoccupied region** using the Density column
4. **Derived from first principles** using the occupied region as reference

### PDOS Projection Weights

1. **Represent spatial localization** of wavefunctions
2. **Independent of occupation** (both filled and empty states have weights)
3. **Useful for understanding** charge transfer, bonding, and interface electronic structure

---

## Appendix: Code Implementation

### TDOS Scaling (Python)

```python
# Find occupied region where Occupation > 0 and Density > 0
occupied_mask = (occupation > 0.5) & (density > 1e-6)

# Calculate DOS from Occupation in occupied region
dos_from_occ = occupation[occupied_mask] / delta_e

# Calculate scaling factor: DOS_true / Density
scale_factors = dos_from_occ / density[occupied_mask]
scale_factor = np.median(scale_factors)

# Apply to entire Density column (works for both occupied and unoccupied)
tdos = density * scale_factor
```

### PDOS Gaussian Broadening (Python)

```python
def gaussian_broaden(energies, weights, grid, sigma):
    """Apply Gaussian broadening to discrete MO data."""
    dos = np.zeros_like(grid)
    for e, w in zip(energies, weights):
        dos += w * np.exp(-0.5 * ((grid - e) / sigma) ** 2)
    dos /= sigma * np.sqrt(2 * np.pi)
    return dos
```

---

*Document generated by InterfaceML DOS Analysis Module*
