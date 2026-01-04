"""
Layer splitting module for InterfaceML.

This module provides functionality to split multi-layer heterostructures
into separate layer files while preserving the original lattice parameters.

Key features:
- Split stacked structures by interface gaps
- Maintain original lattice for each layer
- Support multi-interface structures
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from interfaceml.core.layering import interface_normal_unit, unwrap_periodic_1d
from interfaceml.core.io import get_element_symbols


def identify_layer_type(structure: Structure) -> str:
    """
    Identify the type of layer based on composition.

    Parameters
    ----------
    structure
        Layer structure to identify.

    Returns
    -------
    layer_type
        Descriptive label: "Perovskite", "Fullerene C60", "Fullerene C70", 
        "Oxide", or generic composition.
    """
    elements = {str(el) for el in structure.composition.elements}
    n_atoms = len(structure)
    
    # Check for perovskite (contains Pb, I, and organic cations)
    if 'Pb' in elements and 'I' in elements and ('N' in elements or 'C' in elements):
        return "Perovskite"
    
    # Check for pure carbon structures (fullerenes)
    if elements == {'C'}:
        if 58 <= n_atoms <= 62:
            return "Fullerene C60"
        elif 68 <= n_atoms <= 72:
            return "Fullerene C70"
        elif 78 <= n_atoms <= 82:
            return "Fullerene C80"
        else:
            return f"Carbon ({n_atoms} atoms)"
    
    # Check for oxides
    if 'O' in elements and len(elements) <= 3:
        metals = elements - {'O', 'H'}
        if metals:
            return f"{','.join(sorted(metals))}O_x Oxide"
    
    # Generic composition
    return structure.composition.reduced_formula


def _compute_composition_similarity(elems1: set, elems2: set) -> float:
    """
    Compute similarity between two element sets (0=different, 1=identical).
    
    Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
    """
    if not elems1 or not elems2:
        return 0.0
    intersection = len(elems1 & elems2)
    union = len(elems1 | elems2)
    return intersection / union if union > 0 else 0.0


def _infer_material_family_from_counts(counts, n_atoms: int) -> tuple[str, float, dict]:
    """Infer a broad "material family" from element counts.

    This is intentionally fuzzy and based on *dominant* composition, not mere presence.
    That prevents a window containing a few stray atoms from being misclassified.

    Returns
    -------
    family
        One of: 'Perovskite', 'Fullerene', 'Oxide', 'Other'.
    confidence
        Rough dominance score in [0, 1].
    features
        Useful fractions for downstream scoring.
    """
    if n_atoms <= 0:
        return 'Other', 0.0, {
            'frac_c': 0.0,
            'frac_o': 0.0,
            'frac_halide_metal': 0.0,
            'frac_cation': 0.0,
        }

    n = float(n_atoms)
    frac_c = float(counts.get('C', 0)) / n
    frac_o = float(counts.get('O', 0)) / n
    # Inorganic perovskite framework markers
    framework = {'Pb', 'Sn', 'Ge'}
    halides = {'I', 'Br', 'Cl'}
    frac_framework_halide = float(sum(counts.get(e, 0) for e in (framework | halides))) / n
    # Common perovskite A-site/cation markers (MA/FA/Cs etc.)
    cations = {'N', 'Cs'}
    frac_cation = float(sum(counts.get(e, 0) for e in cations)) / n

    metals_for_oxides = {'Ti', 'Zn', 'Sn', 'Ni', 'Cu', 'Al', 'In', 'Ga', 'Fe', 'Co', 'Mn'}
    has_oxide_metal = any(counts.get(m, 0) > 0 for m in metals_for_oxides)

    features = {
        'frac_c': frac_c,
        'frac_o': frac_o,
        'frac_halide_metal': frac_framework_halide,
        'frac_cation': frac_cation,
    }

    # Pure-carbon (fullerene) window
    if frac_c >= 0.92:
        return 'Fullerene', frac_c, features

    # Perovskite window (either inorganic framework/halide OR organic/cation dominated)
    # Using a low threshold allows organic-only slices to be treated as Perovskite family.
    if frac_framework_halide >= 0.05 or frac_cation >= 0.05:
        return 'Perovskite', max(frac_framework_halide, frac_cation), features

    # Oxide-like window
    if frac_o >= 0.15 and has_oxide_metal:
        return 'Oxide', frac_o, features

    return 'Other', 0.0, features


def _get_material_family(elems: set) -> str:
    """Backward-compatible family inference using only an element set."""
    from collections import Counter

    counts = Counter({e: 1 for e in elems})
    family, _, _ = _infer_material_family_from_counts(counts, max(len(elems), 1))
    return family


def _contact_fraction_across_cut(
    structure: Structure,
    before_indices: np.ndarray,
    after_indices: np.ndarray,
    *,
    cutoff: float = 3.2,
) -> float:
    """Fraction of inter-window atom pairs within a distance cutoff (PBC-aware)."""
    if before_indices.size == 0 or after_indices.size == 0:
        return 0.0
    frac_before = np.asarray([structure[int(i)].frac_coords for i in before_indices], dtype=float)
    frac_after = np.asarray([structure[int(i)].frac_coords for i in after_indices], dtype=float)
    dists = structure.lattice.get_all_distances(frac_before, frac_after)
    if dists.size == 0:
        return 0.0
    return float(np.count_nonzero(dists <= cutoff)) / float(dists.size)


def _label_carbon_components(
    structure: Structure,
    symbols: List[str],
    *,
    cutoff: float = 1.85,
) -> np.ndarray:
    """Label connected components among carbon atoms using a C–C bond cutoff.

    Non-carbon atoms are labeled as -1.

    Notes
    -----
    This is used to avoid selecting cut positions that split a single fullerene
    molecule into fragments.
    """
    n_atoms = len(symbols)
    comp = np.full(n_atoms, -1, dtype=int)
    carbon_indices = np.array([i for i, s in enumerate(symbols) if s == 'C'], dtype=int)
    if carbon_indices.size == 0:
        return comp

    frac = np.asarray([structure[int(i)].frac_coords for i in carbon_indices], dtype=float)
    d = structure.lattice.get_all_distances(frac, frac)
    # adjacency: within cutoff, excluding self
    adj = (d <= float(cutoff)) & (d > 1e-8)

    visited = np.zeros(carbon_indices.size, dtype=bool)
    current_label = 0
    for start in range(carbon_indices.size):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp[int(carbon_indices[u])] = current_label
            # neighbors
            neigh = np.flatnonzero(adj[u] & (~visited))
            for v in neigh.tolist():
                visited[v] = True
                stack.append(int(v))
        current_label += 1

    return comp


def _score_cut_positions_environment_aware(
    structure: Structure,
    order_unwrapped: np.ndarray,
    h_unwrapped: np.ndarray,
    *,
    min_gap: float,
    carbon_component_ids: np.ndarray | None = None,
    window_size: int = 24,
    env_weight: float = 0.80,
    gap_weight: float = 0.10,
    contact_weight: float = 0.10,
) -> List[Tuple[int, float]]:
    """Score all possible cut positions using environment + soft gap + contact density.

    Returns
    -------
    scores
        List of (cut_pos, score), where cut_pos is in [1, n_atoms-1].

    Why this exists
    ---------------
    Real interfaces may have *small* vertical gaps (especially relaxed interfaces),
    so filtering candidates by min_gap can miss them. Here min_gap is treated as a
    *soft* preference, while the dominant signal comes from local atomic environment.
    """
    from collections import Counter

    symbols = get_element_symbols(structure)
    n_atoms = int(order_unwrapped.size)
    if n_atoms < 4:
        return []

    scores: List[Tuple[int, float]] = []
    for cut_pos in range(1, n_atoms):
        # Window slices in the unwrapped ordering (positions, not atom indices)
        before_slice = slice(max(0, cut_pos - window_size), cut_pos)
        after_slice = slice(cut_pos, min(n_atoms, cut_pos + window_size))
        before_indices = np.asarray(order_unwrapped[before_slice], dtype=int)
        after_indices = np.asarray(order_unwrapped[after_slice], dtype=int)
        if before_indices.size < 4 or after_indices.size < 4:
            continue

        counts_before = Counter(symbols[int(i)] for i in before_indices)
        counts_after = Counter(symbols[int(i)] for i in after_indices)
        elems_before = set(counts_before.keys())
        elems_after = set(counts_after.keys())

        family_before, conf_before, feat_before = _infer_material_family_from_counts(
            counts_before, int(before_indices.size)
        )
        family_after, conf_after, feat_after = _infer_material_family_from_counts(
            counts_after, int(after_indices.size)
        )

        frac_c_before = float(feat_before.get('frac_c', 0.0))
        frac_c_after = float(feat_after.get('frac_c', 0.0))
        carbon_jump = abs(frac_c_before - frac_c_after)

        # Hard guardrail: never split inside a perovskite family region.
        # In practice, perovskites often show large *internal* gaps between
        # inorganic framework and organic/cation sublayers; these are NOT interfaces.
        # We only relax this if one side is clearly carbon-rich (near a fullerene).
        if (
            family_before == 'Perovskite'
            and family_after == 'Perovskite'
            and max(frac_c_before, frac_c_after) < 0.60
            and carbon_jump < 0.40
        ):
            # Keep it in the list but make it unselectable in practice.
            scores.append((cut_pos, -1.0e9))
            continue

        # --- Environment score ---
        # Base: set-level difference (cheap)
        similarity = _compute_composition_similarity(elems_before, elems_after)
        env_score = 1.0 - similarity

        # Strong signal: jump in "pure carbon" fraction
        if carbon_jump >= 0.45:
            env_score += 2.5 * carbon_jump

        # Family transition bonuses / penalties
        if family_before != family_after:
            # If both are confidently classified, reward strongly.
            if conf_before >= 0.10 and conf_after >= 0.10:
                # Perovskite <-> Fullerene is the common case here.
                if {'Perovskite', 'Fullerene'} == {family_before, family_after}:
                    env_score += 3.0
                else:
                    env_score += 2.0
            else:
                env_score += 1.0
        else:
            # Same family: usually *not* an interface
            if family_before == 'Perovskite':
                # Critical: do not split inside perovskite even if inorganic/organic differs.
                env_score -= 3.0
            elif family_before == 'Oxide':
                env_score -= 1.5
            elif family_before == 'Fullerene':
                # Two carbon layers: allow if the carbon windows differ significantly
                # (often indicates different molecules/packing)
                env_score += 0.5

        # Prevent splitting a single fullerene molecule: if a carbon component appears
        # on both sides of the cut, this cut likely slices through a molecule.
        if carbon_component_ids is not None:
            c_before = before_indices[np.fromiter((symbols[int(i)] == 'C' for i in before_indices), dtype=bool)]
            c_after = after_indices[np.fromiter((symbols[int(i)] == 'C' for i in after_indices), dtype=bool)]
            if c_before.size and c_after.size:
                ids_before = set(int(carbon_component_ids[int(i)]) for i in c_before if int(carbon_component_ids[int(i)]) >= 0)
                ids_after = set(int(carbon_component_ids[int(i)]) for i in c_after if int(carbon_component_ids[int(i)]) >= 0)
                if ids_before & ids_after:
                    env_score -= 6.0
                elif family_before == 'Fullerene' and family_after == 'Fullerene' and ids_before and ids_after:
                    # Reward cuts that separate different carbon components (e.g., C70 vs C60)
                    env_score += 1.5

        # --- Contact score (weak) ---
        contact_frac = _contact_fraction_across_cut(
            structure, before_indices, after_indices, cutoff=3.2
        )
        contact_score = 1.0 - contact_frac

        # --- Gap score (soft) ---
        gap = float(h_unwrapped[cut_pos] - h_unwrapped[cut_pos - 1])
        # Scale gap relative to min_gap, but never exclude small gaps.
        denom = max(float(min_gap), 1e-6)
        gap_scaled = gap / denom
        gap_score = float(np.clip(gap_scaled, -1.0, 2.0))
        if gap < min_gap:
            # soft penalty when gap is smaller than preferred
            gap_score -= 0.35 * float((min_gap - gap) / denom)

        combined = env_weight * env_score + gap_weight * gap_score + contact_weight * contact_score
        scores.append((cut_pos, float(combined)))

    return scores


def _detect_composition_changes(
    structure: Structure,
    heights: np.ndarray,
    gap_positions: List[int],
    window_size: int = 20,
) -> List[Tuple[int, float]]:
    """
    Score each gap by composition change across it.
    
    Returns list of (gap_index, composition_change_score).
    Higher score = more likely to be a real interface.
    
    Notes
    -----
    Uses enhanced composition analysis including:
    - Material family detection (Perovskite, Fullerene, Oxide)
    - Jaccard similarity for element set differences
    - Element ratio analysis for quantitative changes
    - Strong bonuses for material type transitions
    - Strong penalties for intra-material gaps (especially within Perovskite)
    """
    symbols = get_element_symbols(structure)
    sorted_indices = np.argsort(heights)
    n_atoms = len(sorted_indices)
    
    scores = []
    for gap_idx in gap_positions:
        # Use larger window for more reliable composition analysis
        before_start = max(0, gap_idx - window_size)
        before_end = gap_idx
        after_start = gap_idx
        after_end = min(gap_idx + window_size, n_atoms)
        
        before_indices = sorted_indices[before_start:before_end]
        after_indices = sorted_indices[after_start:after_end]
        
        # Skip if either side has no atoms
        if len(before_indices) == 0 or len(after_indices) == 0:
            scores.append((gap_idx, 0.0))
            continue
        
        # Get element sets and counts on each side
        elems_before = {symbols[i] for i in before_indices}
        elems_after = {symbols[i] for i in after_indices}
        
        # Count element occurrences for ratio analysis
        from collections import Counter
        counts_before = Counter(symbols[i] for i in before_indices)
        counts_after = Counter(symbols[i] for i in after_indices)
        
        # Basic Jaccard similarity
        similarity = _compute_composition_similarity(elems_before, elems_after)
        change_score = 1.0 - similarity  # Higher = more different
        
        # Identify material families
        family_before = _get_material_family(elems_before)
        family_after = _get_material_family(elems_after)
        
        # --- Scoring Logic ---
        
        # Case 1: Material Family Transition (e.g. Perovskite -> Fullerene)
        if family_before != family_after:
            if family_before != 'Other' and family_after != 'Other':
                # Strong bonus for clear transition between known families
                change_score += 2.0
            else:
                # Moderate bonus for transition involving 'Other'
                change_score += 1.0
                
        # Case 2: Same Material Family (e.g. Perovskite -> Perovskite)
        elif family_before == family_after:
            if family_before == 'Perovskite':
                # Strong penalty for splitting within Perovskite
                # Perovskites often have internal gaps between inorganic/organic layers
                # We want to treat them as a single layer
                change_score -= 2.5
            
            elif family_before == 'Fullerene':
                # Two fullerene layers (e.g. C70 -> C60)
                # Check if atom counts are different (indicating different molecules)
                count_before = len(before_indices)
                count_after = len(after_indices)
                
                # Normalize by window size if windows are full, otherwise use absolute
                # But here window might cut off molecules.
                # Better to check density or just raw count if windows are same size?
                # The windows are defined by window_size, but might be truncated at ends.
                # Let's use the ratio of counts in the window.
                
                ratio = abs(count_before - count_after) / max(count_before, count_after)
                
                if ratio > 0.15:  # Significant density/count difference
                    change_score += 1.5  # Likely different fullerenes
                else:
                    change_score -= 1.0  # Likely same fullerene
            
            elif family_before == 'Oxide':
                change_score -= 1.0
                
        # Case 3: Specific Element Checks (Refinement)
        
        # Bonus for transitions to/from pure carbon (any fullerene)
        if elems_after == {'C'} and len(elems_before) > 1 and family_before != 'Fullerene':
            change_score += 0.5
        if elems_before == {'C'} and len(elems_after) > 1 and family_after != 'Fullerene':
            change_score += 0.5
            
        scores.append((gap_idx, change_score))
    
    return scores


def split_structure_into_layers_smart(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 1.5,
    composition_weight: float = 0.85,
) -> List[Structure]:
    """
    Smart layer splitting using both gap size AND composition changes.
    
    This algorithm is more robust than simple gap-based splitting because it:
    1. Detects composition changes across gaps (e.g., perovskite → fullerene)
    2. Avoids splitting within a single material (e.g., intra-layer gaps in perovskite)
    3. Prioritizes gaps with significant composition differences
    4. Uses element ratio analysis to distinguish real interfaces from internal gaps
    
    Parameters
    ----------
    structure
        Input stacked structure.
    n_interfaces
        Number of interfaces to detect.
    min_gap
        Minimum gap size in Angstroms to consider (default: 1.5).
    composition_weight
        Weight for composition change vs gap size (0-1, default: 0.85).
        Higher values prioritize composition changes over gap size.
        Recommended: 0.85-0.95 for mixed materials like perovskite/fullerene stacks.
    
    Returns
    -------
    layer_structures
        List of Structure objects, one per layer.
        
    Notes
    -----
    The algorithm uses a combined scoring approach:
    - Composition score: Based on element set differences and material type transitions
    - Gap size score: Normalized gap size contribution
    - Final score: weighted combination with heavy emphasis on composition
    
    For perovskite/fullerene structures, this correctly identifies material boundaries
    while ignoring large internal gaps within the perovskite layer.
    """
    if n_interfaces < 1:
        raise ValueError("n_interfaces must be >= 1")
    
    n_layers = n_interfaces + 1
    
    # Compute heights along interface normal
    normal = interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), normal)
    
    # Handle periodic boundaries
    c_vec = np.asarray(structure.lattice.matrix[2], dtype=float)
    period = float(abs(np.dot(c_vec, normal)))
    if period <= 1e-8:
        period = float(np.max(heights) - np.min(heights) + 1.0)
    
    h_mod = np.mod(heights, period)
    order_unwrapped, h_unwrapped = unwrap_periodic_1d(h_mod, period)
    
    if len(order_unwrapped) == 0:
        return []

    # Precompute carbon connected components to avoid splitting a fullerene molecule.
    symbols = get_element_symbols(structure)
    carbon_component_ids = _label_carbon_components(structure, symbols, cutoff=1.85)

    # Environment-aware scoring across *all* possible cut positions.
    # This avoids missing real interfaces with small vertical gaps.
    all_scores = _score_cut_positions_environment_aware(
        structure,
        order_unwrapped,
        h_unwrapped,
        min_gap=float(min_gap),
        carbon_component_ids=carbon_component_ids,
        window_size=24,
        env_weight=float(composition_weight),
        gap_weight=float(max(0.0, 1.0 - composition_weight) * 0.5),
        contact_weight=float(max(0.0, 1.0 - composition_weight) * 0.5),
    )
    if not all_scores:
        return []

    # Select N cut positions with a minimum spacing to avoid picking adjacent cuts.
    all_scores.sort(key=lambda x: x[1], reverse=True)
    min_spacing = max(10, 24 // 2)
    selected: List[int] = []
    for cut_pos, score in all_scores:
        if all(abs(cut_pos - s) >= min_spacing for s in selected):
            selected.append(int(cut_pos))
            if len(selected) >= n_interfaces:
                break

    # Fallback: if spacing constraint was too strict
    if len(selected) < n_interfaces:
        for cut_pos, _ in all_scores:
            if int(cut_pos) not in selected:
                selected.append(int(cut_pos))
                if len(selected) >= n_interfaces:
                    break

    cuts = sorted(set(selected[:n_interfaces]))
    starts = [0] + cuts
    ends = cuts + [len(order_unwrapped)]
    
    groups: List[List[int]] = []
    for s, e in zip(starts, ends):
        groups.append([int(i) for i in order_unwrapped[s:e].tolist()])
    
    # Sort groups by mean height
    groups.sort(
        key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0
    )
    
    # Create Structure objects for each layer
    layer_structures = []
    for indices in groups:
        species = [structure[i].species for i in indices]
        frac_coords = [structure[i].frac_coords for i in indices]
        
        layer_struct = Structure(
            structure.lattice,
            species,
            frac_coords,
            coords_are_cartesian=False,
            to_unit_cell=False
        )
        layer_structures.append(layer_struct)
    
    return layer_structures


def split_structure_into_layers(
    structure: Structure,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
    use_smart_detection: bool = True,
) -> List[Structure]:
    """
    Split a stacked heterostructure into separate layer structures.

    This function uses an enhanced algorithm that considers both gap sizes
    and composition changes to correctly identify interfaces, avoiding
    false splits within a single material (e.g., intra-layer gaps in perovskites).

    Each layer maintains the original lattice parameters but contains only
    the atoms from that layer.

    Parameters
    ----------
    structure
        Input stacked structure with multiple interfaces.
    n_interfaces
        Number of interfaces in the stack. Will produce (n_interfaces + 1) layers.
        For example:
        - 1 interface → 2 layers
        - 2 interfaces → 3 layers
    min_gap
        Minimum gap size in Angstroms to consider as an interface (default: 0.0).
        For smart detection, recommend 1.5-2.0 Å to filter intra-layer gaps.
    use_smart_detection
        If True (default), uses composition-aware splitting that avoids
        splitting within a single material. If False, uses simple gap-based splitting.

    Returns
    -------
    layer_structures
        List of Structure objects, one per layer, ordered from bottom to top.
        Each structure has the same lattice as the input but contains only
        atoms from that layer.

    Examples
    --------
    >>> from interfaceml.core import io, splitting
    >>> stack = io.load_structure("perovskite_c70_c60.vasp")
    >>> # Smart detection (recommended for mixed materials)
    >>> layers = splitting.split_structure_into_layers(stack, n_interfaces=2, min_gap=1.5)
    >>> print(f"Split into {len(layers)} layers")
    >>> for i, layer in enumerate(layers):
    ...     layer_type = splitting.identify_layer_type(layer)
    ...     print(f"Layer {i+1}: {layer_type} ({len(layer)} atoms)")
    
    Notes
    -----
    The smart detection algorithm examines composition changes across gaps:
    - Perovskite → Fullerene: Large composition change, valid interface
    - Perovskite → Perovskite: Small change, likely intra-layer gap
    - Pure C → Mixed elements: Large change, valid interface
    """
    if use_smart_detection and min_gap < 1.0:
        # For smart detection, use reasonable minimum to avoid tiny gaps
        min_gap = max(min_gap, 1.5)
    
    if use_smart_detection:
        return split_structure_into_layers_smart(
            structure,
            n_interfaces,
            min_gap=min_gap,
            composition_weight=0.85  # Higher weight for better material boundary detection
        )
    else:
        # Fall back to simple gap-based splitting
        from interfaceml.core.layering import split_stack_layers
        layer_indices = split_stack_layers(structure, n_interfaces, min_gap=min_gap)
        
        layer_structures = []
        for indices in layer_indices:
            species = [structure[i].species for i in indices]
            frac_coords = [structure[i].frac_coords for i in indices]
            
            layer_struct = Structure(
                structure.lattice,
                species,
                frac_coords,
                coords_are_cartesian=False,
                to_unit_cell=False
            )
            layer_structures.append(layer_struct)
        
        return layer_structures


def split_and_save_layers(
    input_file: str,
    output_dir: str,
    n_interfaces: int,
    *,
    min_gap: float = 0.0,
    base_name: str | None = None,
) -> List[str]:
    """
    Split a structure file into separate layer files.

    Parameters
    ----------
    input_file
        Path to input structure file (CIF, VASP, or XYZ).
    output_dir
        Directory to save output layer files.
    n_interfaces
        Number of interfaces to split by.
    min_gap
        Minimum gap size in Angstroms.
    base_name
        Base name for output files. If None, derives from input filename.

    Returns
    -------
    output_paths
        List of paths to created layer files.

    Examples
    --------
    >>> from interfaceml.core import splitting
    >>> files = splitting.split_and_save_layers(
    ...     "trilayer.vasp",
    ...     "output/",
    ...     n_interfaces=2
    ... )
    >>> print(files)
    ['output/trilayer_layer1.vasp', 'output/trilayer_layer2.vasp', ...]
    """
    # Import here to avoid circular dependency
    from interfaceml.core.io import load_structure, write_poscar
    
    # Load structure
    structure = load_structure(input_file)
    
    # Split into layers
    layers = split_structure_into_layers(structure, n_interfaces, min_gap=min_gap)
    
    # Prepare output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine base name
    if base_name is None:
        base_name = Path(input_file).stem
    
    # Save each layer
    output_paths = []
    for i, layer_struct in enumerate(layers, start=1):
        output_file = out_dir / f"{base_name}_layer{i}.vasp"
        write_poscar(
            layer_struct,
            output_file,
            comment=f"{base_name} - Layer {i}/{len(layers)}"
        )
        output_paths.append(str(output_file))
    
    return output_paths


__all__ = [
    "split_structure_into_layers",
    "split_and_save_layers",
]
