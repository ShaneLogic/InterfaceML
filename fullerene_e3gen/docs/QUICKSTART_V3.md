# Quick Start Guide - Version 3.0

**Enhanced Geometry and Physics Constraints**

This guide helps you get started with the v3.0 improvements for generating physically realistic fullerene structures.

---

## What's New in v3.0?

✅ **Bond angle constraints** (sp² hybridization, 120°)  
✅ **Ring planarity** (hexagons flat, pentagons curved)  
✅ **Global normalization** (consistent scale across all molecules)  
✅ **Standard templates** (mathematically correct C60/C20 topology)  
✅ **Adaptive time schedules** (constraints activate at optimal stages)

---

## Step 1: Verify Installation

Ensure all dependencies are installed:

```bash
pip install torch torchvision torch-geometric
pip install pyyaml networkx matplotlib scipy tqdm
```

Check that you're in the correct directory:

```bash
cd fullerene_e3gen
ls config.yaml  # Should exist
```

---

## Step 2: Review Configuration

The new `config.yaml` includes v3 loss weights:

```yaml
loss:
  # Existing constraints
  lambda_bond: 25.0
  lambda_sphere: 10.0
  lambda_connectivity: 3.0
  lambda_topology: 1.0
  
  # NEW v3 constraints
  lambda_angle: 5.0          # Bond angle (sp2 geometry)
  target_angle: 120.0        # Ideal angle (degrees)
  lambda_planarity: 2.0      # Ring flatness
```

**Optional**: Adjust weights if needed, but defaults are tuned for optimal performance.

---

## Step 3: Clean Old Checkpoints (Important!)

v3.0 introduces architectural changes. Old checkpoints are not compatible:

```bash
# Backup if needed
mv checkpoints checkpoints_v2_backup

# Create fresh checkpoint directory
mkdir -p checkpoints
```

---

## Step 4: Train with v3 Improvements

Start training from scratch to leverage all improvements:

```bash
python train.py --config config.yaml --epochs 150
```

**What to monitor**:
- `angle` loss: Should decrease after epoch 20, reach < 0.1 by epoch 100
- `bond` loss: Should be < 0.5 by epoch 50
- `planarity` loss: Should be < 1.0 by epoch 50

**Expected training time**:
- GPU (RTX 3070): ~4-6 hours for 150 epochs
- GPU (A100): ~2-3 hours

**Checkpoints saved**:
- `checkpoints/best_model.pt`: Best validation loss
- `checkpoints/checkpoint_epoch_N.pt`: Every 10 epochs

---

## Step 5: Generate Structures

### Generate C60 (uses icosahedral template automatically):

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 60 \
  --num_samples 10 \
  --output_dir generated_c60/
```

### Generate C20 (uses dodecahedral template):

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 20 \
  --num_samples 10 \
  --output_dir generated_c20/
```

### Enable physics guidance (optional, improves quality):

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 60 \
  --num_samples 10 \
  --guidance_scale 0.2 \
  --guidance_end 200 \
  --output_dir generated_guided/
```

### Enable post-sampling refinement (optional, further improves geometry):

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 60 \
  --num_samples 10 \
  --refine_steps 50 \
  --refine_lr 0.01 \
  --output_dir generated_refined/
```

---

## Step 6: Evaluate Results

Run comprehensive evaluation:

```bash
python evaluate.py \
  --generated_dir generated_c60/ \
  --reference_dir ../dataset/fullerenes/fullerene_xyz/ \
  --output_dir evaluation_results/
```

**Key metrics to check**:

| Metric | Target (v3.0) | What it means |
|--------|---------------|---------------|
| Bond length mean | 1.42 ± 0.05 Å | Carbon-carbon bonds |
| Bond length std | < 0.10 Å | Consistency |
| Bond angle mean | 120 ± 5° | sp² hybridization |
| Asphericity | < 0.10 | Spherical shape |
| Hexagon planarity | < 0.05 Å² | Flat hexagons |
| Pentagon count | 12 | Exact (fullerene rule) |

**Visualize** (if evaluation produces plots):
```bash
open evaluation_results/evaluation_comprehensive.png
```

---

## Step 7: Compare with v2.0 (Optional)

If you backed up v2.0 checkpoints:

```bash
# Generate with v2.0 model
python generate.py \
  --checkpoint checkpoints_v2_backup/best_model.pt \
  --C 60 \
  --num_samples 10 \
  --output_dir generated_v2/

# Evaluate both
python evaluate.py --generated_dir generated_v2/ --output_dir eval_v2/
python evaluate.py --generated_dir generated_c60/ --output_dir eval_v3/

# Compare metrics in evaluation_report.txt files
```

---

## Troubleshooting

### Issue: Training loss is NaN

**Cause**: Learning rate too high or gradient explosion  
**Solution**:
```yaml
training:
  learning_rate: 2.0e-5  # Reduce from 5e-5

loss:
  grad_clip: 10.0  # Increase from 5.0
```

### Issue: Bond angle loss not decreasing

**Cause**: Weight too low or template connectivity incorrect  
**Solution**:
```yaml
loss:
  lambda_angle: 10.0  # Increase from 5.0
```

Also check generated structures have degree-3 connectivity:
```bash
python -c "
from generate import create_template_graph
template = create_template_graph(60, 'cpu')
degrees = template.edge_index[0].bincount()
print(f'Degree distribution: {degrees.unique()}')  # Should be all 3
"
```

### Issue: Out of memory during training

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 4  # From 8
```

**Solution 2**: Reduce model size
```yaml
model:
  hidden_dim: 48  # From 64
  num_layers: 2   # From 3
```

### Issue: Generated structures are fragmented

**Cause**: Topology loss too low or template edges incorrect  
**Solution**:
```yaml
loss:
  lambda_topology: 2.0  # Increase from 1.0
  lambda_connectivity: 5.0  # Increase from 3.0
```

---

## Advanced Usage

### Custom Molecule Size

For sizes other than C20/C60 (uses k-NN fallback template):

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 70 \
  --num_samples 5 \
  --output_dir generated_c70/
```

**Note**: C70 doesn't have a standard template yet, quality may vary.

### Hyperparameter Tuning

To find optimal loss weights:

1. Start with defaults
2. Monitor individual loss components during training
3. If one component dominates (e.g., angle loss >> bond loss), reduce its weight
4. If one component doesn't decrease, increase its weight
5. Retrain and compare evaluation metrics

### Visualize in VESTA

Generated `.xyz` files can be opened in VESTA for 3D visualization:

1. Download VESTA: https://jp-minerals.org/vesta/en/
2. Open VESTA
3. File → Open → select `C60_sample_000.xyz`
4. View → Ball & Stick
5. Check geometry visually

---

## Expected Results

After successful training with v3.0, you should see:

✅ **Compact spherical shapes** (not flattened or elongated)  
✅ **Uniform bond lengths** (~1.42 Å, std < 0.10 Å)  
✅ **Correct bond angles** (~120°, sp² trigonal planar)  
✅ **Flat hexagons** and **curved pentagons**  
✅ **No fragmentation** (all atoms connected)  
✅ **Proper topology** (12 pentagons, correct hexagon count)

---

## Next Steps

1. **Experiment with guidance**: Try different `guidance_scale` values (0.1-0.5)
2. **Test refinement**: Compare with/without `refine_steps`
3. **Generate dataset**: Create large batches for downstream tasks
4. **Analyze statistics**: Use `evaluate.py` comprehensively

---

## Support

For issues or questions:
- Check `docs/improvements_v3.md` for technical details
- Review `CHANGELOG.md` for all changes
- Examine `docs/architecture.md` for system design

---

**Happy generating! 🎉**
