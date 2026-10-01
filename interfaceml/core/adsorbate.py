"""
Adsorbate placement and stacking utilities for InterfaceML.

This module provides a lightweight workflow for building interface models
by stacking an adsorbate (e.g., fullerene) on top of a perovskite slab.
It is designed to be deterministic and fast for high-throughput generation.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator

from interfaceml.core import layering
from interfaceml.utils import geometry


@dataclass(frozen=True)
class SupercellChoice:
    nx: int
    ny: int
    target_xy: tuple[float, float]
    diameter: float
    termination: str = "auto"
    termination_top: tuple[str, ...] = ()
    termination_bottom: tuple[str, ...] = ()


def build_slab(
    structure: Structure,
    miller: tuple[int, int, int],
    slab_thickness: float,
    vacuum: float,
    *,
    center_slab: bool = True,
    termination: str | None = None,
    layer_tol: float = 1.5,
) -> Structure:
    """
    Build a surface slab from a bulk structure using pymatgen's SlabGenerator.
    """
    slabgen = SlabGenerator(
        structure,
        miller_index=tuple(int(x) for x in miller),
        min_slab_size=float(slab_thickness),
        min_vacuum_size=float(vacuum),
        center_slab=center_slab,
    )
    slabs = slabgen.get_slabs()
    if not slabs:
        raise RuntimeError(f"No slabs generated for miller={miller}")

    if termination is None or str(termination).lower() in {"auto", "any"}:
        slab, _ = _select_slab_by_termination(slabs, target=None, layer_tol=layer_tol)
        return slab

    slab, _ = _select_slab_by_termination(slabs, target=str(termination), layer_tol=layer_tol)
    return slab


def auto_supercell_xy(
    slab: Structure,
    adsorbate: Structure,
    *,
    buffer: float = 10.0,
    min_supercell: int = 1,
    max_supercell: int = 8,
) -> SupercellChoice:
    """
    Choose an in-plane supercell (nx, ny) to avoid adsorbate-adsorbate overlap.

    The target in-plane size is diameter + buffer. Each in-plane lattice
    vector is scaled by the minimum integer that meets the target.
    """
    a_len = float(np.linalg.norm(slab.lattice.matrix[0]))
    b_len = float(np.linalg.norm(slab.lattice.matrix[1]))

    diameter = geometry.estimate_molecule_diameter(np.asarray(adsorbate.cart_coords))
    # Guard against NaN diameter (e.g. from diffusion model producing NaN coords)
    if not np.isfinite(diameter):
        diameter = 0.0
    target = max(diameter + buffer, max(a_len, b_len))
    if not np.isfinite(target):
        target = max(a_len, b_len, buffer)

    nx = int(np.ceil(target / max(a_len, 1e-6)))
    ny = int(np.ceil(target / max(b_len, 1e-6)))

    nx = min(max(nx, min_supercell), max_supercell)
    ny = min(max(ny, min_supercell), max_supercell)

    return SupercellChoice(nx=nx, ny=ny, target_xy=(target, target), diameter=diameter)


def _center_adsorbate(coords: np.ndarray) -> np.ndarray:
    center = geometry.center_of_mass(coords)
    return coords - center


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def _apply_rotation(coords: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
    if rot is None:
        return coords
    return np.dot(coords, rot.T)


def prepare_adsorbate_layer(
    adsorbate: Structure,
    lattice: Lattice,
    *,
    xy_frac: tuple[float, float] = (0.5, 0.5),
    rotation: np.ndarray | None = None,
) -> Structure:
    """
    Center the adsorbate and place it at a target in-plane fractional position.
    """
    coords = np.asarray(adsorbate.cart_coords, dtype=float)
    coords = _center_adsorbate(coords)
    coords = _apply_rotation(coords, rotation)

    a_vec = np.asarray(lattice.matrix[0], dtype=float)
    b_vec = np.asarray(lattice.matrix[1], dtype=float)
    xy_cart = a_vec * float(xy_frac[0]) + b_vec * float(xy_frac[1])

    coords = coords + xy_cart

    return Structure(
        lattice,
        adsorbate.species,
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
    )


def molecular_clusters(s: Structure, cutoff: float = 1.8) -> np.ndarray:
    """Connected-component labels for atoms joined by short bonds.

    Cutoff 1.8 Å captures C-N/C-H/N-H/O-H (organic cations) but excludes
    Pb-I (~3.2 Å), I-I, Cs-I, etc., so inorganic framework atoms stay
    as singletons.
    """
    try:
        from scipy.sparse import csr_matrix  # type: ignore
        from scipy.sparse.csgraph import connected_components  # type: ignore
    except Exception:
        return np.arange(len(s))
    n_sites = len(s)
    if n_sites == 0:
        return np.zeros(0, dtype=int)
    try:
        centers, neighbors, _imgs, _d = s.get_neighbor_list(r=float(cutoff))
    except Exception:
        return np.arange(n_sites)
    if len(centers) == 0:
        return np.arange(n_sites)
    data = np.ones(len(centers), dtype=float)
    adj = csr_matrix((data, (centers, neighbors)), shape=(n_sites, n_sites))
    _ncomp, labels = connected_components(adj, directed=False)
    return labels


def molecular_unwrap(s: Structure, *, cutoff: float = 1.8) -> Structure:
    """Unwrap molecules split by the periodic boundary along c.

    Uses BFS over the PBC-aware neighbor graph: for each covalent edge
    (atom i, neighbor j, image offset img), atoms on the j side are
    translated so they connect to i without crossing the boundary.
    Only c-axis unwrap is applied (in-plane periodicity stays intact for
    surface coverage by the slab supercell).
    """
    n_sites = len(s)
    if n_sites < 2:
        return s
    try:
        centers, neighbors, images, _d = s.get_neighbor_list(r=float(cutoff))
    except Exception:
        return s
    if len(centers) == 0:
        return s

    adjacency: dict = {}
    for k in range(len(centers)):
        i = int(centers[k])
        j = int(neighbors[k])
        img = np.asarray(images[k], dtype=float)
        adjacency.setdefault(i, []).append((j, img))
        adjacency.setdefault(j, []).append((i, -img))

    fracs = np.array(s.frac_coords, dtype=float)
    unwrap = fracs.copy()
    visited = np.zeros(n_sites, dtype=bool)

    for seed in range(n_sites):
        if visited[seed] or seed not in adjacency:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            i = stack.pop()
            for j, img in adjacency.get(i, []):
                if visited[j]:
                    continue
                # neighbor j sits at fracs[j] + img in the i-frame.
                # We only unwrap c-axis to avoid breaking lateral PBC tiling.
                offset = np.array([0.0, 0.0, img[2]], dtype=float)
                unwrap[j] = unwrap[i] + (fracs[j] + offset - fracs[i])
                # snap x/y to lateral periodicity of the original lattice
                unwrap[j, 0] = fracs[j, 0]
                unwrap[j, 1] = fracs[j, 1]
                visited[j] = True
                stack.append(j)

    if np.allclose(unwrap, fracs):
        return s
    return Structure(
        s.lattice,
        s.species,
        unwrap,
        coords_are_cartesian=False,
        to_unit_cell=False,
        site_properties=s.site_properties,
    )


def _site_element(site) -> str:
    """Return bare element symbol regardless of oxidation state on the site.

    Handles `Species('I-')` → 'I', `Species('Pb2+')` → 'Pb', neutral `Element('C')` → 'C'.
    """
    sp = site.specie
    try:
        return str(sp.element.symbol)
    except AttributeError:
        try:
            return str(sp.symbol)
        except AttributeError:
            return str(sp)


def _layer_elements(
    structure: Structure,
    *,
    which: str,
    layer_tol: float = 1.5,
) -> tuple[str, ...]:
    n = layering.interface_normal_unit(structure)
    heights = np.dot(np.asarray(structure.cart_coords, dtype=float), n)
    if len(heights) == 0:
        return tuple()
    max_h = float(np.max(heights))
    min_h = float(np.min(heights))
    if which == "top":
        mask = heights >= (max_h - float(layer_tol))
    else:
        mask = heights <= (min_h + float(layer_tol))
    elems = {_site_element(site) for site, keep in zip(structure, mask) if keep}
    return tuple(sorted(elems))


def _classify_termination(elements: Iterable[str]) -> str:
    elems = set(elements)
    if not elems:
        return "unknown"
    if "Pb" in elems and "I" in elems and not ({"C", "N"} & elems):
        return "PbI"
    if {"C", "N"} & elems and "I" in elems:
        return "AI"
    if "Pb" in elems and "I" in elems:
        return "PbI"
    if {"C", "N"} & elems:
        return "AI"
    return "unknown"


def _parse_target_atoms(raw) -> set | None:
    """Parse user input into a normalized element-symbol set.

    Accepts:
      - None / "" / "auto" / "any"  → None (no target)
      - list/tuple of element symbols, e.g. ["Cs", "I"]
      - string "Cs,I" or "Cs I" or "Pb,I"
      - legacy labels: PbI/FAI/MAI/AI/CsI/SnI/GeI/... split by uppercase letters
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in {"auto", "any"}:
            return None
        organic_aliases = {
            "FA": ("C", "N"),
            "MA": ("C", "N"),
            "EA": ("C", "N"),
            "DMA": ("C", "N"),
            "GA": ("C", "N"),
            "BA": ("C", "N"),
            "IA": ("C", "N"),
            "CA": ("C", "N"),
        }
        if any(ch in s for ch in (",", " ", ";", "/")):
            raw_tokens = [
                tok
                for tok in s.replace(";", ",").replace("/", ",").replace(" ", ",").split(",")
                if tok
            ]
        else:
            # Tokenize labels like "PbI", "FAI", "CsBr", "DMAI": match longer
            # organic aliases (FA, MA, DMA, ...) before element symbols [A-Z][a-z]?
            import re as _re

            pattern = _re.compile(r"(DMA|FA|MA|EA|GA|BA|IA|CA|[A-Z][a-z]?)")
            raw_tokens = pattern.findall(s)
        items: list[str] = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok in organic_aliases:
                items.extend(organic_aliases[tok])
            else:
                items.append(tok)
    else:
        return None

    cleaned = {str(x).strip().capitalize() for x in items if str(x).strip()}
    cleaned = {x for x in cleaned if x and x[0].isalpha()}
    return cleaned or None


def _select_slab_by_termination(
    slabs: list[Structure],
    *,
    target: str | None = None,
    target_atoms: set | None = None,
    layer_tol: float = 2.5,
) -> tuple[Structure, SupercellChoice]:
    """Pick the slab whose top layer best matches the target.

    Priority: `target_atoms` (element set) > `target` (legacy label).

    Scoring (target_atoms given): F1 of top-layer element set against
    target set. Hydrogen is ignored in the comparison since most analyzers
    care about heavy-atom termination.
    """
    best = None
    best_score = None

    if target_atoms is None and target is not None:
        target_atoms = _parse_target_atoms(target)

    target_norm = target.strip() if isinstance(target, str) and target.strip() else None

    def _molecular_clusters(s: Structure, cutoff: float = 1.8) -> np.ndarray:
        """Connected-component labels for atoms joined by short bonds.

        Cutoff 1.8 Å captures C-N/C-H/N-H/O-H (organic cations) but excludes
        Pb-I (~3.2 Å), I-I, Cs-I, etc., so inorganic framework atoms stay
        as singletons and only molecular cations form clusters.
        """
        try:
            from scipy.sparse import csr_matrix  # type: ignore
            from scipy.sparse.csgraph import connected_components  # type: ignore
        except Exception:
            return np.arange(len(s))
        n_sites = len(s)
        if n_sites == 0:
            return np.zeros(0, dtype=int)
        try:
            centers, neighbors, _imgs, _d = s.get_neighbor_list(r=float(cutoff))
        except Exception:
            return np.arange(n_sites)
        if len(centers) == 0:
            return np.arange(n_sites)
        data = np.ones(len(centers), dtype=float)
        adj = csr_matrix((data, (centers, neighbors)), shape=(n_sites, n_sites))
        _ncomp, labels = connected_components(adj, directed=False)
        return labels

    def molecular_unwrap(s: Structure, *, cutoff: float = 1.8) -> Structure:
        """Move atoms so multi-atom molecules are not split across the c-boundary.

        For each molecular cluster (size >= 2), if its frac_z span > 0.5
        (signature of wrap), atoms on the minority side are translated by ±1
        in frac_z to rejoin the majority.
        """
        labels = _molecular_clusters(s, cutoff=cutoff)
        if len(labels) == 0:
            return s
        fracs = np.array(s.frac_coords, dtype=float)
        changed = False
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            if len(idx) < 2:
                continue
            z = fracs[idx, 2] % 1.0
            if z.max() - z.min() < 0.5:
                continue
            upper_mask = z >= 0.5
            lower_mask = ~upper_mask
            if upper_mask.sum() >= lower_mask.sum():
                fracs[idx[lower_mask], 2] = (fracs[idx[lower_mask], 2] % 1.0) + 1.0
            else:
                fracs[idx[upper_mask], 2] = (fracs[idx[upper_mask], 2] % 1.0) - 1.0
            changed = True
        if not changed:
            return s
        return Structure(
            s.lattice,
            s.species,
            fracs,
            coords_are_cartesian=False,
            to_unit_cell=False,
            site_properties=s.site_properties,
        )

    def _shift_layer_to_top(s: Structure, layer_indices: list[int]) -> Structure:
        """Trim the slab so the given layer becomes the topmost atomic plane.

        Removes all atoms strictly above the target layer. Also removes any
        partial molecule whose atoms straddle that cut (to avoid dangling
        bonds). The resulting slab is thinner; callers should ensure their
        starting `slab_thickness` accounts for this.
        """
        n = layering.interface_normal_unit(s)
        heights = np.dot(np.asarray(s.cart_coords, dtype=float), n)
        if len(layer_indices) == 0:
            return s
        target_max = float(max(heights[i] for i in layer_indices))
        cut_z = target_max + 0.05  # small tolerance for numerical jitter
        keep = heights <= cut_z

        # Don't truncate molecules: if any atom of a cluster is kept and
        # another would be cut, keep both (otherwise cut both).
        labels = molecular_clusters(s)
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
            if len(idx) < 2:
                continue
            keeps_any = bool(np.any(keep[idx]))
            cuts_any = bool(np.any(~keep[idx]))
            if keeps_any and cuts_any:
                # Decide: keep whole cluster only if its COM is at or below cut
                com_z = float(np.mean(heights[idx]))
                if com_z <= cut_z:
                    keep[idx] = True
                else:
                    keep[idx] = False

        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            return s
        new_species = [s.species[i] for i in keep_idx]
        new_fracs = np.array(s.frac_coords, dtype=float)[keep_idx]
        return Structure(
            s.lattice,
            new_species,
            new_fracs,
            coords_are_cartesian=False,
            to_unit_cell=False,
            site_properties={
                name: [values[i] for i in keep_idx] for name, values in s.site_properties.items()
            },
        )

    def _flip_c(s: Structure) -> Structure:
        fracs = np.array(s.frac_coords, dtype=float)
        fracs[:, 2] = 1.0 - fracs[:, 2]
        return Structure(
            s.lattice,
            s.species,
            fracs,
            coords_are_cartesian=False,
            to_unit_cell=False,
            site_properties=s.site_properties,
        )

    def _heavy_set(layer_idx_list: list[int], s: Structure) -> set:
        return {_site_element(s[i]) for i in layer_idx_list if _site_element(s[i]) != "H"}

    def _f1_for_target(top_heavy: set, target_heavy: set) -> float:
        if not top_heavy or not target_heavy:
            return 0.0
        inter = top_heavy & target_heavy
        if not inter:
            return 0.0
        p = len(inter) / len(top_heavy)
        r = len(inter) / len(target_heavy)
        return 2 * p * r / (p + r)

    # For each slab + flip: keep the original, and trim above the HIGHEST
    # layer that matches target_atoms (so target becomes top while preserving
    # as much slab thickness as possible). If no target, only originals.
    candidates: list[Structure] = []
    for s in slabs:
        for base in (molecular_unwrap(s), molecular_unwrap(_flip_c(s))):
            candidates.append(base)
            if not target_atoms:
                continue
            try:
                layers_idx, _ = layering.split_layers_by_z(base, gap_cut=True)
            except Exception:
                continue
            target_heavy = {e for e in target_atoms if e != "H"} or set(target_atoms)
            best_layer_idx = None
            best_layer_pos = -1
            for li, layer in enumerate(layers_idx):
                if len(layer) < 2:
                    continue
                if li == len(layers_idx) - 1:
                    continue  # already on top
                heavy = _heavy_set(layer, base)
                if _f1_for_target(heavy, target_heavy) >= 0.66 and li > best_layer_pos:
                    best_layer_pos = li
                    best_layer_idx = layer
            if best_layer_idx is not None:
                with suppress(Exception):
                    candidates.append(_shift_layer_to_top(base, best_layer_idx))

    def _terminal_elements(s: Structure) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Top/bottom chemical layers via z-gap clustering.

        Falls back to height-window method if clustering yields <2 layers.
        """
        try:
            layers_idx, _tol = layering.split_layers_by_z(s, gap_cut=True)
        except Exception:
            layers_idx = []
        if len(layers_idx) >= 2:
            top_e = tuple(sorted({_site_element(s[i]) for i in layers_idx[-1]}))
            bot_e = tuple(sorted({_site_element(s[i]) for i in layers_idx[0]}))
            return top_e, bot_e
        return (
            _layer_elements(s, which="top", layer_tol=layer_tol),
            _layer_elements(s, which="bottom", layer_tol=layer_tol),
        )

    for slab in candidates:
        top_elems, bottom_elems = _terminal_elements(slab)
        top_label = _classify_termination(top_elems)

        top_set = set(top_elems)
        bottom_set = set(bottom_elems)

        if target_atoms:
            # Compare heavy-atom sets — H is structural noise (molecule edges)
            top_heavy = {e for e in top_set if e != "H"}
            target_heavy = {e for e in target_atoms if e != "H"} or target_atoms
            inter = top_heavy & target_heavy
            recall = len(inter) / len(target_heavy) if target_heavy else 0.0
            precision = len(inter) / len(top_heavy) if top_heavy else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            score = 1.0 - f1
            # Tiebreak: penalize symmetric slabs (unrealistic surface)
            sym_union = top_set | bottom_set
            sym_inter = top_set & bottom_set
            sym = 0.0 if not sym_union else (len(sym_inter) / len(sym_union))
            score = score + 0.05 * sym
        else:
            union = top_set | bottom_set
            inter = top_set & bottom_set
            score = 1.0 if not union else 1.0 - (len(inter) / len(union))
            if target_norm and top_label != target_norm:
                score += 1.0

        if best is None or score < best_score:
            best = (slab, top_label, top_elems, bottom_elems)
            best_score = score

        if target_atoms and best_score is not None and best_score < 0.01:
            break

    if best is None:
        raise RuntimeError("Could not select a termination slab.")

    slab, top_label, top_elems, bottom_elems = best
    # Display the ACTUAL top layer composition, not the user's request
    actual_top = "+".join(sorted(set(top_elems))) if top_elems else top_label
    info = SupercellChoice(
        nx=1,
        ny=1,
        target_xy=(0.0, 0.0),
        diameter=0.0,
        termination=actual_top,
        termination_top=tuple(top_elems),
        termination_bottom=tuple(bottom_elems),
    )
    return slab, info


def stack_structures(
    bottom: Structure,
    top: Structure,
    *,
    separation: float = 3.2,
    vacuum: float = 20.0,
) -> tuple[Structure, Structure, Structure]:
    """
    Stack two slabs/structures along the interface normal.

    Returns (bottom_aligned, top_aligned, combined).
    """
    bottom_aligned = bottom.copy()
    top_aligned = top.copy()

    n = layering.interface_normal_unit(bottom_aligned)
    a_vec = np.asarray(bottom_aligned.lattice.matrix[0], dtype=float)
    b_vec = np.asarray(bottom_aligned.lattice.matrix[1], dtype=float)

    if len(bottom_aligned) > 0:
        bottom_proj = np.dot(np.asarray(bottom_aligned.cart_coords, dtype=float), n)
        min_bottom = float(bottom_proj.min())
        bottom_aligned.translate_sites(
            list(range(len(bottom_aligned))),
            -n * min_bottom,
            frac_coords=False,
            to_unit_cell=False,
        )
        bottom_proj = np.dot(np.asarray(bottom_aligned.cart_coords, dtype=float), n)
        max_bottom = float(bottom_proj.max())
    else:
        max_bottom = 0.0

    if len(top_aligned) > 0:
        top_proj = np.dot(np.asarray(top_aligned.cart_coords, dtype=float), n)
        min_top = float(top_proj.min())
        top_aligned.translate_sites(
            list(range(len(top_aligned))),
            -n * min_top,
            frac_coords=False,
            to_unit_cell=False,
        )
        top_aligned.translate_sites(
            list(range(len(top_aligned))),
            n * (max_bottom + float(separation)),
            frac_coords=False,
            to_unit_cell=False,
        )
        top_proj = np.dot(np.asarray(top_aligned.cart_coords, dtype=float), n)
        max_top = float(top_proj.max())
    else:
        max_top = max_bottom + float(separation)

    total_z = max_top + float(vacuum)
    c_vec = n * total_z

    combined_lat = Lattice(np.vstack([a_vec, b_vec, c_vec]))

    interface_z_cart = max_bottom + float(separation) / 2.0
    interface_z_frac = interface_z_cart / total_z if total_z > 1e-8 else 0.5

    def _wrap_layer(coords: np.ndarray, keep_above: bool | None) -> list[np.ndarray]:
        fracs: list[np.ndarray] = []
        for coord in coords:
            fc = combined_lat.get_fractional_coords(coord)
            fc_x = fc[0] % 1.0
            fc_y = fc[1] % 1.0
            # For the adsorbate (top) layer we do NOT wrap z — the atoms
            # have been explicitly placed above the slab by translate_sites
            # and should keep their Cartesian z.  Wrapping can push them
            # inside the slab or below it, causing the "missing fullerene"
            # visual artefact.
            if keep_above is None:
                fc_z = fc[2]
            elif keep_above:
                # Keep atoms above the interface — no modulo wrapping
                fc_z = fc[2]
                if fc_z < 0:
                    fc_z = fc_z + 1.0
            else:
                fc_z = fc[2] % 1.0
                if fc_z > interface_z_frac + 0.05:
                    fc_z = fc_z - 1.0
            fracs.append(np.array([fc_x, fc_y, fc_z], dtype=float))
        return fracs

    bottom_fracs = _wrap_layer(
        np.asarray(bottom_aligned.cart_coords, dtype=float), keep_above=False
    )
    top_fracs = _wrap_layer(np.asarray(top_aligned.cart_coords, dtype=float), keep_above=True)

    bottom_out = Structure(
        combined_lat, bottom_aligned.species, bottom_fracs, coords_are_cartesian=False
    )
    top_out = Structure(combined_lat, top_aligned.species, top_fracs, coords_are_cartesian=False)

    combined = Structure(combined_lat, [], [])
    for site in bottom_out:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)
    for site in top_out:
        combined.append(site.species_string, site.frac_coords, coords_are_cartesian=False)

    return bottom_out, top_out, combined


def build_adsorbate_interface(
    base_structure: Structure,
    adsorbate_structure: Structure,
    *,
    miller: tuple[int, int, int] = (0, 0, 1),
    slab_thickness: float = 18.0,
    vacuum: float = 20.0,
    separation: float = 3.2,
    supercell_xy: tuple[int, int] | None = None,
    buffer: float = 10.0,
    xy_frac: tuple[float, float] = (0.5, 0.5),
    termination: str | None = None,
    termination_atoms: Iterable[str] | None = None,
    layer_tol: float = 1.5,
    rotation: np.ndarray | None = None,
) -> tuple[Structure, Structure, Structure, SupercellChoice]:
    """
    Build a perovskite/adsorbate interface from bulk inputs.

    Returns bottom slab, top adsorbate layer, combined interface, and supercell choice.
    """
    target_atoms = _parse_target_atoms(termination_atoms) if termination_atoms else None
    if target_atoms is None and termination is not None:
        target_atoms = _parse_target_atoms(termination)

    slab, term_info = _select_slab_by_termination(
        SlabGenerator(
            base_structure,
            miller_index=tuple(int(x) for x in miller),
            min_slab_size=float(slab_thickness),
            min_vacuum_size=0.0,
            center_slab=True,
        ).get_slabs(),
        target=termination,
        target_atoms=target_atoms,
        layer_tol=layer_tol,
    )

    if supercell_xy is None:
        choice = auto_supercell_xy(slab, adsorbate_structure, buffer=buffer)
        nx, ny = choice.nx, choice.ny
    else:
        nx, ny = int(supercell_xy[0]), int(supercell_xy[1])
        choice = SupercellChoice(nx=nx, ny=ny, target_xy=(0.0, 0.0), diameter=0.0)

    slab_super = slab.copy()
    slab_super.make_supercell([nx, ny, 1])

    top_lattice = Lattice(np.array(slab_super.lattice.matrix))
    adsorbate_layer = prepare_adsorbate_layer(
        adsorbate_structure,
        top_lattice,
        xy_frac=xy_frac,
        rotation=rotation,
    )

    bottom_out, top_out, combined = stack_structures(
        slab_super,
        adsorbate_layer,
        separation=separation,
        vacuum=vacuum,
    )

    # Repair any molecular cations / adsorbates that got split across the
    # c-boundary by stack_structures' fc[2] % 1.0 wrapping step.
    bottom_out = molecular_unwrap(bottom_out)
    top_out = molecular_unwrap(top_out)
    combined = molecular_unwrap(combined)

    choice = SupercellChoice(
        nx=choice.nx,
        ny=choice.ny,
        target_xy=choice.target_xy,
        diameter=choice.diameter,
        termination=term_info.termination,
        termination_top=term_info.termination_top,
        termination_bottom=term_info.termination_bottom,
    )

    return bottom_out, top_out, combined, choice
