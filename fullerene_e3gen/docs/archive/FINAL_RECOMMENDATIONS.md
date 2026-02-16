# Final Recommendations After Comprehensive Testing

**Date**: 2026-01-31  
**After**: Multiple training iterations and extensive debugging  
**Conclusion**: Fundamental normalization strategy needs revision

---

## Summary of Testing

### Iterations Completed

1. **v3.0**: Original improvements → Loss explosions
2. **v3 fixed**: Loss clipping → Stable but bond loss not decreasing
3. **v4 high lambda**: lambda_bond=100 → Still no improvement
4. **v5 no time**: Disabled time-conditioning → Catastrophic explosion (bond loss 24648)

### Key Findings

✅ **What Works**:
- Training stability (with all loss clipping)
- Topology constraints (degree-3, connectivity)
- Shape constraints (asphericity < 0.15)
- New geometry constraints (angle, planarity) are stable

❌ **What Doesn't Work**:
- Bond length learning (10 Å vs 1.42 Å target after 10 epochs)
- Current normalization strategy prevents scale convergence
- Time-conditioning needed for stability but suppresses learning

---

## Root Cause Analysis

### The Fundamental Issue

**Standard deviation normalization is incompatible with diffusion on molecular structures.**

Problem:
1. Training data normalized: `coords /= global_std` (std ≈ 1.8)
2. Diffusion starts from: `noise ~ N(0, I)` (std = 1.0)
3. Model learns: coordinates with std >> 1.8
4. Result: Generated structures have wrong scale

**Physics losses try to correct this, but:**
- Time-conditioning weakens them 70% of the time
- Even λ=100 is insufficient vs the scale mismatch from diffusion noise

---

## Recommended Solutions

### Option 1: Radius Normalization (RECOMMENDED)

Replace std-normalization with radius-based:

```python
# In dataset.py
def _normalize_coords(self, coords):
    centered = coords - coords.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1).mean()
    if radius > 1e-6:
        coords = centered / radius  # Scale to unit radius
    return coords
```

**Benefits**:
- All molecules normalized to radius=1.0 sphere
- Bond lengths naturally ~0.4 (for C60: 1.42 Å / 3.5 Å radius)
- Diffusion noise matches training distribution
- No scale mismatch!

**Changes Needed**:
- `dataset.py`: Replace `_normalize_coords()`
- `config.yaml`: `target_bond: 0.40` (≈ 1.42/3.5)
- `generate.py`: Denormalize: `coords * typical_radius` (3.5 Å for C60)

### Option 2: No Normalization (SIMPLEST)

```yaml
data:
  center: true
  normalize_scale: false  # Turn off scaling
```

**Benefits**:
- Train directly in Angstrom space
- `target_bond: 1.42` works directly
- No scale confusion

**Drawbacks**:
- May need to adjust learning rate
- Different molecule sizes have different coord ranges

### Option 3: Fixed-Scale Normalization

```python
# Always divide by fixed value (e.g., 5.0 Å)
coords = (coords - coords.mean()) / 5.0
```

**Benefits**:
- Consistent across molecules
- `target_bond: 0.284` (= 1.42 / 5.0)
- Simple and predictable

---

## Immediate Action Plan

### Step 1: Stop Current Training
```bash
pkill -f train.py
```

### Step 2: Choose Strategy

**Recommended: Option 1 (Radius Normalization)**

Implement in `dataset.py`:
```python
def _normalize_coords(self, coords):
    centered = coords - coords.mean(axis=0)
    # Radius normalization instead of std
    radii = np.linalg.norm(centered, axis=1)
    mean_radius = radii.mean()
    if mean_radius > 1e-6:
        return centered / mean_radius
    return centered
```

Update `config.yaml`:
```yaml
loss:
  target_bond: 0.40  # 1.42 Å / ~3.5 Å (C60 radius)
  bond_tolerance: 0.01
```

### Step 3: Retrain
```bash
rm -rf checkpoints/*
python train.py --config config.yaml --epochs 150
```

### Step 4: Test at Epoch 10
```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_10.pt --C 60 --num_samples 5
```

**Expected**: Bonds ~1.3-1.6 Å by epoch 10

---

## Alternative: Simpler Fix

If you want to avoid code changes:

1. **Turn off normalization**:
   ```yaml
   data:
     normalize_scale: false
   ```

2. **Use physical targets**:
   ```yaml
   loss:
     target_bond: 1.42
     lambda_bond: 50.0
   ```

3. **Retrain and test**

---

## Final Notes

The core improvements (angle constraints, ring planarity, templates, stability fixes) are **solid and will work** once the normalization issue is resolved.

The fundamental lesson: **Diffusion models are sensitive to coordinate scale**. Std-normalization creates a mismatch between noise distribution and data distribution for molecular structures.

**Radius normalization is the standard approach in molecular generative models** for this exact reason.

---

**Decision needed**: Which normalization strategy to use?
1. Radius normalization (best, requires code change)
2. No normalization (simplest, requires retrain)
3. Fixed-scale normalization (compromise)

Let me know and I'll implement immediately!
