# Smart Layer Splitting Algorithm

## Purpose

Multi-layer heterostructures often contain internal gaps within a material
(e.g., perovskite organic/inorganic sublayers). A gap-only split can therefore
cut a single material into multiple layers. The smart algorithm uses local
composition and contact information to detect true material boundaries.

Implementation lives in `interfaceml/core/splitting.py`.
For the broader math across modules, see `docs/MATHEMATICAL_OVERVIEW.md`.

## Algorithm Summary

1. **Project heights** along the interface normal `n` and unwrap periodic
   coordinates by cutting at the largest gap.
2. **Label carbon components** (C-C cutoff 1.85 Angstrom) to avoid splitting a
   single fullerene molecule.
3. **Score every candidate cut position** using a weighted combination of:
   - environment (composition) change
   - soft gap preference
   - inter-window contact density
4. **Select the top N cut positions** with minimum spacing, then split.

## Scoring Details

For each cut position `k`, the algorithm inspects a window of `W` atoms on each
side (default `W = 24`).

### Environment Score

- Base signal: `1 - Jaccard(elements_before, elements_after)`.
- Material-family inference by dominant composition. Criteria: **Fullerene** if
  carbon fraction >= 0.92; **Perovskite** if halide/metal fraction >= 0.05 or
  cation fraction >= 0.05; **Oxide** if oxygen fraction >= 0.15 and a known oxide
  metal is present; **Other** otherwise.
- Strong bonuses for clear family transitions (especially
  Perovskite <-> Fullerene).
- Penalties for same-family cuts, with a strict guardrail against
  Perovskite -> Perovskite splits.
- Carbon-component guard: large penalty if a cut splits the same carbon
  connected component; small bonus if it cleanly separates different fullerenes.

### Contact Score

- `contact_fraction` = fraction of inter-window atom pairs within 3.2 Angstrom
  (PBC-aware).
- `contact_score = 1 - contact_fraction`.

### Gap Score (Soft Preference)

- `gap_scaled = gap / min_gap`, clipped to `[-1, 2]`.
- Gaps below `min_gap` are allowed but penalized.

### Combined Score

`score = w_env * env + w_gap * gap + w_contact * contact`

Default weights:

- `w_env = 0.85`
- `w_gap = w_contact = 0.075`

## Cut Selection

- Candidate cuts are ranked by score.
- A minimum spacing (>= 12 positions) avoids adjacent cuts.
- If spacing removes too many candidates, the remaining cuts are filled by score.

## Parameter Guidance

- `min_gap`: 1.5 to 2.0 Angstrom for relaxed perovskite/fullerene stacks.
- `composition_weight`: default 0.85 (highly recommended for mixed materials).
- `use_smart_detection`: True for most heterostructures.

## Notes

- The algorithm is intentionally conservative about splitting perovskites.
- Gap size alone is insufficient for relaxed or compositionally layered slabs.
- The environment-aware scoring avoids false splits while remaining robust when
  vertical gaps are small.
