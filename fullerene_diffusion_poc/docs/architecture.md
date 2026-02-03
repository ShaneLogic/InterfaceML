# Architecture Overview

## High-Level Pipeline

1. **Data Loading** (`dataset.py`)
   - Parse extended XYZ with 3-neighbor connectivity.
   - Center and normalize coordinates per structure.

2. **Diffusion Scheduler** (`diffusion_utils.py`)
   - Cosine beta schedule with precomputed coefficients.
   - Forward noise and reverse sampling utilities.

3. **Model** (`model.py`)
   - E(3)-equivariant EGNN stack.
   - Time and carbon-count conditioning.
   - Attention-weighted message passing with residual updates.

4. **Physics Losses** (`topology_loss_v2.py` + `model.py`)
   - Bond length, connectivity, and topology constraints.
   - Time-conditioned weighting for stability.
   - Scale-invariant sphericity loss for normalized data.

5. **Training** (`train.py`)
   - Noise prediction objective + weighted physics terms.
   - Gradient clipping and cosine LR schedule.

6. **Generation** (`generate.py`)
   - Template-aware connectivity from dataset.
   - Optional physics guidance during late denoising steps.
   - Optional post-sampling geometry refinement.

## Key Design Choices

- **Template connectivity** avoids mismatch between training and sampling graphs.
- **Per-node attention normalization** stabilizes aggregation.
- **Scale-invariant geometry constraints** match normalized coordinates.
- **Modular loss configuration** via `config.yaml`.

