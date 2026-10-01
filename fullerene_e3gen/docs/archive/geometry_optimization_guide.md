# Geometry Issues Diagnosis & Optimization Guide

## Summary of Observed Issues

- **Atomic dispersion**: structures are not compact spheres.
- **Bond length instability**: wide distribution, many too short or too long.
- **Local clustering**: multiple dense clusters connected by long bridges.
- **Shape flattening**: structures look quasi‑2D rather than 3D spherical.

## Root Causes

1. **Connectivity template mismatch** during sampling (random 3‑regular graph).
2. **Loss parameter mismatch** (bond tolerance and target bond not used consistently).
3. **Scale mismatch in sphericity loss** under normalized coordinates.
4. **Weak geometric guidance** in late diffusion steps.

## Fixes Implemented in This Version

- **Template‑aware generation** using dataset connectivity when available.
- **Per‑node attention normalization** in EGNN for stable aggregation.
- **Config‑driven bond target/tolerance** wired into physics losses.
- **Scale‑invariant sphericity loss** when `sphere_target_radius` is null.
- **Optional physics guidance** and **post‑sampling refinement** for geometry.

## Recommended Settings

```yaml
loss:
  lambda_bond: 25.0
  lambda_sphere: 10.0
  lambda_connectivity: 3.0
  lambda_topology: 1.0
  target_bond: 1.42
  bond_tolerance: 0.03
  sphere_target_radius: null
  sphere_radius_weight: 0.5
```

## Suggested Evaluation Checklist

- Bond length mean and std close to 1.42 ± 0.03.
- Degree distribution near 3 for all atoms.
- Asphericity below 0.10 (normalized).
- Visual inspection in VESTA for compact sphere.

