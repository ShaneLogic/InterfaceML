# Training Report (2026‑01‑30)

## Run Overview

- **Model**: E(3)‑equivariant diffusion with EGNN backbone
- **Dataset**: fullerene XYZ (C50–C70)
- **Normalization**: centered + unit variance per structure
- **Scheduler**: cosine beta schedule, T=1000

## Configuration Snapshot

```yaml
model:
  hidden_dim: 64
  num_layers: 3
  time_embed_dim: 64
  C_embed_dim: 32

diffusion:
  num_steps: 1000
  beta_schedule: cosine

training:
  epochs: 150
  batch_size: 8
  learning_rate: 5.0e-5

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

## Observations

- **Training stability** improved with time‑conditioned physics losses.
- **Topology constraints** became reliable after robust ring detection.
- **Geometry quality** remained sensitive to sampling connectivity and scale.

## Actions Taken

- Template‑aware generation using dataset connectivity.
- Scale‑invariant sphericity loss for normalized coordinates.
- Optional physics guidance and post‑sampling refinement.

## Next Steps

- Retrain with updated losses and template‑aware sampling.
- Evaluate on C60 with guidance + refinement enabled.
- Track metrics: bond length std, asphericity, degree distribution.

