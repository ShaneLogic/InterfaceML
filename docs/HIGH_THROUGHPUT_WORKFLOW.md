# High-Throughput Interface Workflow

This guide describes an automated, high-throughput workflow to build
perovskite/fullerenes interface models, optimize them, and curate
training data for generative models.

---

## Overview

1. **Prepare inputs**
   - Perovskite bulk structures (CIF/POSCAR)
   - Fullerene molecules (XYZ/CIF/POSCAR)
2. **Build initial interfaces (high-throughput)**
   - Generate perovskite slabs
   - Auto-select in-plane supercells based on fullerene size
   - Stack adsorbates above the slab at fixed separation
3. **Optimize structures**
   - Fix bottom layers (selective dynamics)
   - Relax with DFT or ML potential
4. **Assemble training dataset**
   - Store input layers, interface geometry, and metadata
5. **Train models**
   - Stage 1: global interface generator (isovariant NN)
   - Stage 2: local interface refinement (GNN)

---

## Step 1: Prepare Inputs

Perovskite structures:
```
dataset/perovskite/mp_perovskite_cifs/*.cif
dataset/perovskite/cif_merge/*.cif
```

Fullerene structures (XYZ from diffusion):
```
fullerene_e3gen/generated/**/*.xyz
dataset/fullerenes/fullerene_xyz/**/C*.xyz
```

---

## Step 2: Batch Interface Generation

Use the batch adsorbate builder:

```bash
python build_heterojunctions/batch_adsorbate_builder.py \
  --perovskite_dir dataset/perovskite/mp_perovskite_cifs \
  --fullerene_dir fullerene_e3gen/generated/quick_test \
  --output_dir outputs/interfaces \
  --miller 0,0,1 \
  --slab_thickness 18 \
  --vacuum 20 \
  --distance 3.2 \
  --supercell auto \
  --buffer 10 \
  --orientation_samples 4 \
  --orientation_mode z \
  --xy_grid 2,2 \
  --termination auto \
  --min_interlayer_dist 1.8 \
  --energy_cmd "python /path/to/energy_eval.py --input {input}" \
  --energy_threshold -500.0 \
  --max_pairs 200 \
  --skip_existing
```

Outputs:
- `outputs/interfaces/*.vasp` interface structures
- `outputs/interfaces/metadata.jsonl` metadata for each structure

---

## Step 3: Add Selective Dynamics (Optional)

```bash
python build_heterojunctions/fix_interface_layers.py \
  --by_z_layers \
  --input outputs/interfaces/<interface>.vasp \
  --n_fix_layers 3 \
  --output outputs/interfaces/<interface>_fixed.vasp
```

---

## Step 4: Dataset Assembly

For each optimized interface, store:
- `perovskite_layer` (slab before stacking)
- `fullerene_layer` (adsorbate)
- `interface_relaxed`
- `metadata.jsonl` record

These constitute the training inputs for the generative model.

---

## Step 5: Model Training Strategy

**Stage 1 (global interface):**
- Train an isovariant model to predict overall slab/adsorbate placement:
  - in-plane translation
  - separation distance
  - coarse rotation (optional)

**Stage 2 (local refinement):**
- Train a GNN to refine local atomic structure near the interface.
- Keep atoms far from the interface fixed to preserve bulk structure.

---

## Recommended Metadata Fields

Each interface record should include:
- input perovskite filename
- input fullerene filename
- slab orientation (miller)
- separation distance
- supercell size
- number of atoms
- relaxation settings + final energy (if available)

---

## Next Steps

- Add sampling of multiple adsorbate orientations and XY positions
- Integrate ML potential relaxation into the batch workflow (`--relax_cmd`)
- Use `--min_interlayer_dist` to filter unphysical overlaps
- Use `--energy_cmd` + `--energy_threshold` to screen interfaces by energy
