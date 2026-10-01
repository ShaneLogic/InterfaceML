# Mathematical and Algorithmic Overview

This document summarizes the core math and algorithms used by InterfaceML. It is
implementation-driven and references the canonical modules in `interfaceml/core`
and the legacy workflows in `build_heterojunctions/`.

## Notation

- Lattice vectors: `a`, `b`, `c` (Cartesian).
- Interface normal: `n = (a x b) / ||a x b||`, oriented so `n dot c > 0`.
- Atomic positions: `r_i` (Cartesian), heights `h_i = r_i dot n`.
- Period along `n`: `P = |c dot n|` (fallback to `max(h) - min(h) + 1.0` if degenerate).
- In-plane vectors for matching: film vectors `f1, f2`, substrate vectors `s1, s2`.

## 1) Height Projection and Periodic Unwrapping

Goal: avoid splitting a physical layer across the periodic boundary.

Algorithm:

1. Compute `h_i` and map to `[0, P)` with modulo.
2. Sort heights and compute consecutive gaps, including the wrap-around gap.
3. Cut at the largest gap and rotate the ordering so the cut is at the start.

This is implemented in `interfaceml/core/layering.py::unwrap_periodic_1d`.

## 2) Layer Clustering for Selective Dynamics

Given sorted, unwrapped heights:

- Successive gaps: `d_i = h_{i+1} - h_i`.
- Automatic tolerance: `tol = clamp(6 * median(d_i), 0.20, 1.00)` (Angstrom).
- Start a new layer when `d_i > tol`.

Complexity: `O(N log N)` for sorting plus `O(N)` clustering.

## 3) Stack Splitting (Simple Gap-Based)

Given `n_interfaces`:

- Select the `n_interfaces` largest gaps that exceed `min_gap`.
- Cut the ordered atom list at those positions.
- Return `n_interfaces + 1` layers.

This is suitable for well-separated multilayers with clean vacuum gaps.

## 4) Smart Splitting (Composition-Aware)

Default for `interfaceml.core.splitting.split_structure_into_layers`.

### Step A: Carbon Component Labeling

- Build a graph over carbon atoms using a C-C cutoff (1.85 Angstrom).
- Connected components help avoid splitting a single fullerene.

### Step B: Score Each Candidate Cut

For each cut position `k`, using a window of `W` atoms on each side (default `W = 24`):

1. Compute element sets and counts on both sides.
2. Infer a material family by dominant composition. Criteria: **Fullerene** if
   carbon fraction >= 0.92; **Perovskite** if halide/metal fraction >= 0.05 or
   cation fraction >= 0.05; **Oxide** if oxygen fraction >= 0.15 and a known
   oxide metal is present; **Other** otherwise.

Environment score (`env`):

- Base term: `1 - Jaccard(elements_before, elements_after)`.
- Bonuses for large carbon-fraction jumps and for family transitions.
- Penalties for same-family cuts, especially Perovskite -> Perovskite.
- Guardrail: cuts that are clearly inside a perovskite region are rejected.
- Carbon component guard: penalize cuts that split a single carbon component.

Contact score:

- `contact_fraction` = fraction of inter-window atom pairs within 3.2 Angstrom
  (PBC-aware distances).
- `contact_score = 1 - contact_fraction`.

Gap score (soft preference):

- `gap_scaled = gap / min_gap`, clipped to `[-1, 2]`.
- Small gaps are allowed but penalized.

Combined score:

`score = w_env * env + w_gap * gap + w_contact * contact`

Default weights:

- `w_env = 0.85`
- `w_gap = w_contact = (1 - w_env) / 2 = 0.075`

### Step C: Select Interfaces

- Choose the top `n_interfaces` cut positions with a minimum spacing
  (>= 12 positions in the sorted list).
- If the spacing constraint is too strict, fill remaining cuts by score.

Implementation: `interfaceml/core/splitting.py`.

## 5) Whole-Molecule Inclusion (Selective Dynamics)

To avoid fixing only part of an organic cation:

- Build a distance graph on `{C, N, H}` atoms with cutoffs
  (C-N 1.75, C-H 1.25, N-H 1.25 Angstrom).
- Compute connected components via DFS.
- If any atom in a component is fixed, fix the entire component.

Implementation: `interfaceml/core/layering.py::include_whole_molecules`.

## 6) Commensurate Interface Matching (ZSL)

Interface matching uses `pymatgen.analysis.interfaces.zsl.ZSLGenerator` to search
integer 2x2 in-plane supercell matrices `M_f`, `M_s` such that transformed film
and substrate lattices are commensurate within tolerances.

Mismatch metrics used for ranking:

- Length mismatch for each in-plane vector:
  `epsilon_i = (|f_i| - |s_i|) / ((|f_i| + |s_i|) / 2)`
- Angle mismatch:
  `delta_theta = |angle(f1, f2) - angle(s1, s2)|`

Candidate matches are filtered by `max_length_tol`, `max_angle_tol`, and
`max_area`. The selected match is used to build slabs and apply strain to the
chosen target(s).

Implementation: `build_heterojunctions/interface_builder.py`.

## 7) Charge Density Difference

Difference density for heterojunctions:

`Delta rho(r) = rho_interface(r) - rho_A(r) - rho_B(r)`

The cube grids must be identical (origin, axes, voxel counts). The implementation
streams through voxels to keep memory usage constant.

Implementation: `build_heterojunctions/delta_density_cube.py`.

## Appendix: ZSL and Coherent Interface Matching

This appendix summarizes the lattice matching math used by the ZSL algorithm and
the coherent interface builder in pymatgen.

### ZSL matching criteria

Let the film in-plane lattice vectors be `f1, f2` and the substrate vectors be
`s1, s2`. For integer 2x2 supercell matrices `M_f` and `M_s`, define the matched
superlattice vectors:

`f_i' = M_f * f_i`, `s_i' = M_s * s_i`

The relative length mismatch for each in-plane vector is:

`epsilon_i = (|f_i'| - |s_i'|) / (0.5 * (|f_i'| + |s_i'|))`

The angle mismatch is:

`delta_theta = |angle(f1', f2') - angle(s1', s2')|`

Candidate matches are accepted when `|epsilon_i| <= max_length_tol`,
`delta_theta <= max_angle_tol`, and the matched superlattice area is within
`max_area` (and `max_area_ratio_tol` when enabled). The ZSL search enumerates
superlattice transformations within the area limit, reduces vectors, and checks
length and angle equivalence for each candidate. [1][2]

### Coherent interface construction

`CoherentInterfaceBuilder` uses ZSL matches to construct coherent interfaces
between film and substrate slabs. It enumerates terminations and generates
interfaces with user-controlled gap, vacuum, and thickness parameters. [2]

### References

1. A. Zur and T. C. McGill, "Lattice match: An application to heteroepitaxy,"
   Journal of Applied Physics 55, 378-386 (1984). DOI: 10.1063/1.333084.
2. pymatgen analysis interfaces documentation (ZSLGenerator and
   CoherentInterfaceBuilder). https://pymatgen.org/pymatgen.analysis.interfaces.html
3. Materials Project Suggested Substrates methodology (ZSL-based lattice
   matching overview). https://docs.materialsproject.org/methodology/materials-project/heterojunctions-interface-science/suggested-substrates

## Implementation Map

- Layering, unwrapping, selective dynamics: `interfaceml/core/layering.py`
- Smart splitting: `interfaceml/core/splitting.py`
- Interface building + ZSL matching: `build_heterojunctions/interface_builder.py`
- Delta density: `build_heterojunctions/delta_density_cube.py`
