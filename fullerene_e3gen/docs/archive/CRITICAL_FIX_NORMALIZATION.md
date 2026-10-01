# CRITICAL FIX: Normalization Scale Mismatch

**Date**: 2026-01-30  
**Severity**: CRITICAL  
**Status**: FIXED

---

## Problem Summary

**Root Cause**: Physics loss targets (bond length, tolerances) were specified in **physical space** (Angstroms), but training coordinates were in **normalized space** (divided by global_std = 1.8292).

**Impact**: Model learned to generate structures with bonds **7× too long** (10.27 Å vs 1.42 Å target).

---

## Detailed Analysis

### Normalization Process

During training:
```python
# In dataset.py
coords_normalized = coords_physical / global_std
# global_std ≈ 1.8292 (computed from all structures)
```

### The Mismatch

**Old config (WRONG)**:
```yaml
loss:
  target_bond: 1.42        # Physical space value ❌
  bond_tolerance: 0.03     # Physical space value ❌
```

**Problem**: Loss function compared:
- Predicted bonds in normalized space (e.g., 5.62)
- Target in physical space (1.42)
- Result: Model "learned" that bonds should be ~5-6 units (in normalized space)

---

## Solution

### Updated Configuration

**New config (CORRECT)**:
```yaml
data:
  global_std: 1.8292              # Explicit normalization factor
  physical_bond_length: 1.42      # Reference (Å)
  physical_bond_tolerance: 0.03   # Reference (Å)

loss:
  target_bond: 0.776        # Normalized: 1.42 / 1.8292
  bond_tolerance: 0.016     # Normalized: 0.03 / 1.8292
```

### Calculation

```
target_bond_normalized = physical_bond / global_std
                       = 1.42 Å / 1.8292
                       = 0.776 (dimensionless, normalized space)

tolerance_normalized = physical_tolerance / global_std
                     = 0.03 Å / 1.8292
                     = 0.016 (dimensionless)
```

---

## Verification

### Expected Behavior (After Retraining)

**In normalized space** (during training):
- Bond lengths: 0.75-0.80 (≈ 0.776 target)
- Sphere radius: ~1.9-2.0 (C60 in normalized space)

**In physical space** (after denormalization):
- Bond lengths: 1.37-1.46 Å (≈ 1.42 Å target) ✅
- Sphere radius: ~3.5 Å (realistic C60) ✅

---

## Files Modified

1. **config.yaml**:
   - Added `global_std`, `physical_bond_length`, `physical_bond_tolerance`
   - Changed `target_bond: 1.42 → 0.776`
   - Changed `bond_tolerance: 0.03 → 0.016`

2. **generate.py**:
   - `save_xyz()`: Added denormalization with `global_std`
   - Template generation: Use normalized-space coordinates
   - C60 template radius: 3.5 Å → 1.9 (normalized)
   - C20 template radius: 2.0 Å → 1.1 (normalized)

3. **Training templates**: Now match normalized coordinate space

---

## Action Required

**MUST RETRAIN** with corrected configuration:

```bash
# Clean old checkpoints (wrong scale)
rm -rf checkpoints/*
mv checkpoints_backup_v2 checkpoints_wrong_scale_archive

# Retrain with correct normalization
python train.py --config config.yaml --epochs 150
```

**DO NOT** use old checkpoints - they learned the wrong scale!

---

## Testing After Retrain

```bash
# Generate test structures
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --C 60 \
  --num_samples 5 \
  --output_dir test/

# Expected results:
# - Bond length: 1.42 ± 0.15 Å (was 10.27 ± 3.76 Å)
# - Bond std: < 0.30 Å (was 3.76 Å)
# - Asphericity: < 0.12 (was 0.13, acceptable)
```

---

## Lesson Learned

**Always ensure loss targets match the coordinate space used in training!**

When using normalization:
1. Compute global statistics (global_std)
2. Convert ALL physical targets to normalized space
3. Document both physical and normalized values
4. Denormalize only when saving/visualizing final structures

---

**Status**: Configuration fixed, awaiting retrain to validate.
