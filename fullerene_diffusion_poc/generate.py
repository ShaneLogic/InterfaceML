"""
Generation script for Fullerene Diffusion Model.

Samples new fullerene structures from trained diffusion model.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --C 60 --num_samples 10

Author: InterfaceML Project
Date: 2026-01-28
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch_geometric.data import Data, Batch

from diffusion_utils import DiffusionScheduler
from model import FullereneDiffusionModel
from template_bank import TemplateBank
from topology_gnn import TopologyGNN
from units import denormalize_positions, bond_target_normalized


def create_template_graph(
    num_carbon: int,
    device: str,
    template_bank: TemplateBank | None,
    topology_model: TopologyGNN | None,
) -> Data:
    """Create a template graph for generation.

    High-probability fix: reuse real fullerene adjacency from the dataset,
    instead of inventing a random 3-regular graph.
    """

    if topology_model is not None:
        try:
            edge_index = topology_model.generate_edge_index(C_value=num_carbon, device=torch.device(device))
            pos = torch.randn(num_carbon, 3, device=torch.device(device))
            return Data(pos=pos, edge_index=edge_index)
        except Exception as exc:
            if template_bank is None:
                raise
            print(f"[WARN] Topology GNN failed for C{num_carbon}: {exc}. Falling back to dataset template.")

    if template_bank is None:
        raise ValueError("Template bank is required when topology GNN is not used.")

    return template_bank.make_template_data(num_carbon, device=torch.device(device))


def generate_samples(
    model: FullereneDiffusionModel,
    scheduler: DiffusionScheduler,
    num_samples: int,
    C_value: int,
    device: str,
    template_bank: TemplateBank | None,
    topology_model: TopologyGNN | None = None,
    use_ddim: bool = False,
    ddim_steps: int = 100,
    guidance_scale: float = 0.0,
    sampling_project_radius: bool = False,
    sampling_nonbonded_min_dist: float | None = None,
    sampling_nonbonded_strength: float = 0.5,
    sampling_nonbonded_iters: int = 1,
    sampling_bond_strength: float = 0.5,
    sampling_bond_iters: int = 1,
    sampling_max_step: float | None = None,
    sampling_rescale_each_step: bool = False,
) -> list:
    """
    Generate fullerene structures using the trained model.
    
    Args:
        model: Trained diffusion model
        scheduler: Diffusion scheduler
        num_samples: Number of samples to generate
        C_value: Number of carbon atoms
        device: Device for computation
        use_ddim: Whether to use DDIM sampling (faster)
        ddim_steps: Number of steps for DDIM (if use_ddim=True)
        guidance_scale: Conditional guidance strength (0=no guidance, 2-5=strong guidance)
                        Uses physics constraint gradients to guide generation process
    
    Returns:
        List of generated positions (each is [C, 3] tensor)
    """
    model.eval()
    
    generated_structures = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Generating C{C_value}"):
            # Create template graph from dataset
            try:
                template = create_template_graph(C_value, device, template_bank, topology_model)
            except Exception as exc:
                print(f"[WARN] Failed to create template for C{C_value}: {exc}. Skipping sample {i}.")
                continue
            
            # Start from pure noise
            pos_t = torch.randn_like(template.pos)
            
            # Prepare conditioning
            C_cond = torch.tensor([C_value], dtype=torch.long, device=device)
            batch = torch.zeros(C_value, dtype=torch.long, device=device)
            
            # Denoise step by step with self-conditioning
            if use_ddim:
                # Use fewer steps with DDIM
                timesteps = torch.linspace(
                    scheduler.num_steps - 1, 0, ddim_steps,
                    dtype=torch.long, device=device
                )
            else:
                # Use all steps with DDPM
                timesteps = torch.arange(
                    scheduler.num_steps - 1, -1, -1,
                    dtype=torch.long, device=device
                )
            
            x_0_pred_prev = None  # For self-conditioning
            
            for step_idx, t in enumerate(timesteps):
                # Predict noise (with optional self-conditioning)
                t_batch = t.unsqueeze(0)  # [1]
                noise_pred = model(
                    pos_t,
                    template.edge_index,
                    t_batch,
                    C_cond,
                    batch,
                )
                
                # Predict x_0 from noisy x_t
                sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
                x_0_pred = (pos_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                
                # NEW: 条件引导生成 (Guided Sampling)
                # 使用物理约束梯度修正预测，引导生成更符合富勒烯特征
                if guidance_scale > 0 and t > 50:  # 只在早期步骤应用引导
                    # NOTE: Guidance is intentionally disabled by default.
                    # The previous implementation attempted to use gradients inside torch.no_grad(),
                    # which cannot work. Keeping this hook here for future work.
                    pass
                
                # REMOVED HARD CLIPPING - Let model learn natural coordinate ranges
                # The model should output reasonable coordinates without artificial bounds
                # Natural C60 structure has radius ~3.5Å, coordinates in [-4, 4] range
                # Previous hard clip at ±3.0 was preventing proper structure formation
                
                # Only apply SOFT limiting at very extreme values to prevent numerical issues
                # Use tanh-based soft clipping instead of hard clamp
                # This preserves gradient flow and allows natural coordinate distributions
                max_reasonable = 8.0  # Much larger than physical C60 (~7Å diameter)
                x_0_pred = max_reasonable * torch.tanh(x_0_pred / max_reasonable)

                
                # Compute posterior mean using precomputed coefficients
                coef1 = scheduler.posterior_mean_coef1[t]
                coef2 = scheduler.posterior_mean_coef2[t]
                mean = coef1 * x_0_pred + coef2 * pos_t
                
                # Add noise (except at t=0) with momentum
                if t > 0:
                    variance = scheduler.posterior_variance[t]
                    noise = torch.randn_like(pos_t)
                    pos_t = mean + torch.sqrt(variance) * noise
                    
                    # Apply momentum if we have previous prediction (smoother transitions)
                    if x_0_pred_prev is not None:
                        momentum = 0.1
                        pos_t = pos_t + momentum * (x_0_pred - x_0_pred_prev)
                else:
                    pos_t = mean

                # Sampling-time projection / repulsion (no post-relaxation)
                if sampling_project_radius:
                    pos_t = rescale_to_unit_radius(pos_t)
                if sampling_bond_iters > 0:
                    target_bond = bond_target_normalized(C_value).to(pos_t.device)
                    pos_t = apply_bond_projection(
                        pos_t,
                        template.edge_index,
                        target_length=target_bond.item(),
                        strength=sampling_bond_strength,
                        iters=sampling_bond_iters,
                        max_step=sampling_max_step,
                    )
                if sampling_nonbonded_min_dist is not None:
                    pos_t = apply_nonbonded_repulsion(
                        pos_t,
                        template.edge_index,
                        min_dist=sampling_nonbonded_min_dist,
                        strength=sampling_nonbonded_strength,
                        iters=sampling_nonbonded_iters,
                        max_step=sampling_max_step,
                    )
                if sampling_rescale_each_step:
                    pos_t = rescale_to_unit_radius(pos_t)
                
                # Store for next iteration
                x_0_pred_prev = x_0_pred.detach()
            
            # Check for NaN
            if torch.isnan(pos_t).any():
                print(f"Warning: NaN detected in sample {i}, skipping")
                continue
            
            # Note: Coordinates are in normalized space (std≈1)
            # Dataset normalization: coords centered and scaled to unit variance
            # Physical bond lengths: C-C bonds are ~1.39-1.46 Å in real space
            # In normalized space: bonds appear as ~1.39/std ≈ various scales depending on molecule size
            # For proper evaluation, structures should be rescaled based on reference statistics
            # The evaluation script handles comparison in consistent normalized space
                
            # Store generated structure
            generated_structures.append(pos_t.cpu())
    
    return generated_structures


def save_xyz(pos: torch.Tensor, edge_index: torch.Tensor, filepath: Path):
    """
    Save generated structure to extended XYZ format.
    
    Args:
        pos: Atomic positions [N, 3]
        edge_index: Edge connectivity [2, E]
        filepath: Output file path
    """
    num_atoms = pos.size(0)
    
    # Build neighbor lists (assuming 3 neighbors per node)
    neighbors = {}
    for i in range(num_atoms):
        neighbors[i] = []
    
    for src, dst in edge_index.t().tolist():
        if len(neighbors[src]) < 3:
            neighbors[src].append(dst)
    
    # Fill missing neighbors with -1
    for i in range(num_atoms):
        while len(neighbors[i]) < 3:
            neighbors[i].append(-1)
    
    # Write to file (match dataset convention: 1-based indices)
    with open(filepath, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Generated fullerene structure\n")
        
        for i in range(num_atoms):
            x, y, z = pos[i].tolist()
            nb1, nb2, nb3 = neighbors[i][:3]
            # Convert to 1-based indexing for compatibility with dataset tooling
            idx_1 = i + 1
            nb1_1 = nb1 + 1
            nb2_1 = nb2 + 1
            nb3_1 = nb3 + 1
            f.write(f"C {x:.6f} {y:.6f} {z:.6f} {idx_1} {nb1_1} {nb2_1} {nb3_1}\n")


def compute_bond_stats(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[float, float]:
    """Return mean/std of bond lengths (unique undirected edges)."""
    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]
    if row.numel() == 0:
        return 0.0, 0.0
    lengths = (pos[row] - pos[col]).norm(dim=1)
    return lengths.mean().item(), lengths.std(unbiased=False).item()


def compute_radius_stats(pos: torch.Tensor) -> tuple[float, float]:
    """Return mean/std of atomic radii from center."""
    centered = pos - pos.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1)
    return radii.mean().item(), radii.std(unbiased=False).item()


def rescale_to_unit_radius(pos: torch.Tensor) -> torch.Tensor:
    """Center and scale positions so mean radius == 1.0."""
    centered = pos - pos.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1)
    mean_r = radii.mean().clamp(min=1e-8)
    return centered / mean_r


def _clamp_step(delta: torch.Tensor, max_step: float) -> torch.Tensor:
    """Clamp per-atom displacement to avoid exploding updates."""
    if max_step is None or max_step <= 0:
        return delta
    norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(max_step / norm, max=1.0)
    return delta * scale


def apply_nonbonded_repulsion(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    min_dist: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: float | None = None,
) -> torch.Tensor:
    """Simple nonbonded repulsion in normalized space (no gradients)."""
    out = pos
    row, col = edge_index
    for _ in range(iters):
        dists = torch.cdist(out, out)
        n = dists.size(0)
        inf = torch.tensor(float('inf'), device=dists.device, dtype=dists.dtype)
        diag_mask = torch.eye(n, device=dists.device, dtype=torch.bool)
        bond_mask = torch.zeros((n, n), device=dists.device, dtype=torch.bool)
        bond_mask[row, col] = True
        bond_mask[col, row] = True
        mask = diag_mask | bond_mask
        dists = dists.masked_fill(mask, inf)

        vec = out[:, None, :] - out[None, :, :]
        dist_safe = dists.unsqueeze(-1).clamp(min=1e-6)
        overlap = (min_dist - dists).clamp(min=0.0)
        push = (overlap.unsqueeze(-1) / dist_safe) * vec
        delta = push.sum(dim=1)
        delta = _clamp_step(delta, max_step)
        out = out + strength * delta
    return out


def apply_bond_projection(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    target_length: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: float | None = None,
) -> torch.Tensor:
    """Project bonded pairs toward target length (normalized space)."""
    out = pos
    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]
    if row.numel() == 0:
        return out

    for _ in range(iters):
        vec = out[row] - out[col]
        dist = vec.norm(dim=1).clamp(min=1e-6)
        delta = ((dist - target_length) / dist).unsqueeze(-1) * vec
        pos_update = torch.zeros_like(out)
        pos_update.index_add_(0, row, delta)
        pos_update.index_add_(0, col, -delta)
        pos_update = _clamp_step(pos_update, max_step)
        out = out - strength * pos_update
    return out


def relax_structure(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    C_value: int,
    *,
    steps: int = 200,
    lr: float = 5e-2,
    bond_weight: float = 1.0,
    radius_weight: float = 0.2,
    radius_var_weight: float = 0.2,
    nonbonded_min_dist: float = 0.45,
    nonbonded_weight: float = 0.2,
) -> torch.Tensor:
    """Lightweight geometric relaxation in normalized space.

    Minimizes:
    - bond length MSE to normalized target
    - mean radius deviation to 1.0
    - radius variance (compact shell)
    """
    pos_opt = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos_opt], lr=lr)

    target_bond = bond_target_normalized(C_value).to(pos_opt.device)

    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]

    for _ in range(steps):
        optimizer.zero_grad()
        centered = pos_opt - pos_opt.mean(dim=0, keepdim=True)
        radii = centered.norm(dim=1)
        mean_r = radii.mean()
        var_r = radii.var(unbiased=False)

        if row.numel() > 0:
            bond_len = (pos_opt[row] - pos_opt[col]).norm(dim=1)
            bond_loss = ((bond_len - target_bond) ** 2).mean()
        else:
            bond_loss = torch.tensor(0.0, device=pos_opt.device)

        radius_loss = (mean_r - 1.0) ** 2
        var_loss = var_r

        # Non-bonded repulsion to avoid atom overlap/clumping
        dists = torch.cdist(pos_opt, pos_opt)
        n = dists.size(0)
        inf = torch.tensor(float('inf'), device=dists.device, dtype=dists.dtype)
        diag_mask = torch.eye(n, device=dists.device, dtype=torch.bool)
        bond_mask = torch.zeros((n, n), device=dists.device, dtype=torch.bool)
        bond_mask[row, col] = True
        bond_mask[col, row] = True
        mask = diag_mask | bond_mask
        dists = dists.masked_fill(mask, inf)
        overlap = torch.clamp(nonbonded_min_dist - dists, min=0)
        nonbonded_loss = (overlap ** 2).mean()

        loss = (
            bond_weight * bond_loss
            + radius_weight * radius_loss
            + radius_var_weight * var_loss
            + nonbonded_weight * nonbonded_loss
        )
        loss.backward()
        optimizer.step()

    return pos_opt.detach()


def is_valid_structure(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    C_value: int,
    *,
    bond_mean_tol: float,
    bond_std_max: float,
    radius_mean_tol: float,
    radius_std_max: float,
    nonbonded_min_dist: float | None = None,
    max_nonbonded_violations: int = 0,
    radius_coeff: float = 0.45,
) -> tuple[bool, dict]:
    """Check basic geometric validity in normalized space.

    Criteria:
    - mean bond length close to normalized target
    - bond length dispersion below threshold
    - mean radius close to 1.0 (normalized)
    - radius dispersion below threshold
    """
    bond_mean, bond_std = compute_bond_stats(pos, edge_index)
    radius_mean, radius_std = compute_radius_stats(pos)

    target_bond = bond_target_normalized(C_value, radius_coeff=radius_coeff).item()

    ok_bond_mean = abs(bond_mean - target_bond) <= bond_mean_tol
    ok_bond_std = bond_std <= bond_std_max
    ok_radius_mean = abs(radius_mean - 1.0) <= radius_mean_tol
    ok_radius_std = radius_std <= radius_std_max

    nonbonded_violations = 0
    ok_nonbonded = True
    if nonbonded_min_dist is not None:
        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(float('inf'))
        row, col = edge_index
        dists[row, col] = float('inf')
        dists[col, row] = float('inf')
        nonbonded_violations = int((dists < nonbonded_min_dist).sum().item())
        ok_nonbonded = nonbonded_violations <= max_nonbonded_violations

    ok = ok_bond_mean and ok_bond_std and ok_radius_mean and ok_radius_std and ok_nonbonded

    stats = {
        "bond_mean": bond_mean,
        "bond_std": bond_std,
        "bond_target": target_bond,
        "radius_mean": radius_mean,
        "radius_std": radius_std,
        "nonbonded_violations": nonbonded_violations,
        "ok_bond_mean": ok_bond_mean,
        "ok_bond_std": ok_bond_std,
        "ok_radius_mean": ok_radius_mean,
        "ok_radius_std": ok_radius_std,
        "ok_nonbonded": ok_nonbonded,
    }
    return ok, stats


def main():
    parser = argparse.ArgumentParser(description='Generate Fullerene Structures')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--C', type=int, required=True, help='Number of carbon atoms (20-720)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of structures to generate')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output directory')
    parser.add_argument('--ddim', action='store_true', help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=100, help='DDIM sampling steps')
    parser.add_argument('--output_scale', type=str, default='normalized', choices=['normalized', 'angstrom'],
                        help='Save coordinates in normalized training space or approximate Å space')
    parser.add_argument('--filter_invalid', action='store_true',
                        help='Filter invalid structures using geometric heuristics')
    parser.add_argument('--bond_mean_tol', type=float, default=0.08,
                        help='Allowed deviation of mean bond length (normalized units)')
    parser.add_argument('--bond_std_max', type=float, default=0.08,
                        help='Max allowed bond length std (normalized units)')
    parser.add_argument('--radius_mean_tol', type=float, default=0.12,
                        help='Allowed deviation of mean radius from 1.0 (normalized units)')
    parser.add_argument('--radius_std_max', type=float, default=0.12,
                        help='Max allowed radius std (normalized units)')
    parser.add_argument('--save_rejects', action='store_true',
                        help='Also save rejected samples to <output_dir>/rejected')
    parser.add_argument('--rescale_to_unit_radius', action='store_true',
                        help='Repair: center+scale each sample to mean radius 1.0 before filtering/saving')
    parser.add_argument('--relax_steps', type=int, default=0,
                        help='Optional post-generation relaxation steps (0 disables)')
    parser.add_argument('--relax_lr', type=float, default=5e-2,
                        help='Learning rate for relaxation')
    parser.add_argument('--relax_bond_weight', type=float, default=1.0,
                        help='Bond length weight during relaxation')
    parser.add_argument('--relax_radius_weight', type=float, default=0.2,
                        help='Mean radius weight during relaxation')
    parser.add_argument('--relax_radius_var_weight', type=float, default=0.2,
                        help='Radius variance weight during relaxation')
    parser.add_argument('--relax_nonbonded_min_dist', type=float, default=0.45,
                        help='Minimum non-bonded distance in normalized space')
    parser.add_argument('--relax_nonbonded_weight', type=float, default=0.2,
                        help='Non-bonded repulsion weight during relaxation')
    parser.add_argument('--filter_nonbonded_min_dist', type=float, default=None,
                        help='Filter: minimum non-bonded distance (normalized). None disables')
    parser.add_argument('--filter_max_nonbonded_violations', type=int, default=0,
                        help='Max allowed non-bonded violations')
    parser.add_argument('--sampling_project_radius', action='store_true',
                        help='Project to unit radius at each sampling step')
    parser.add_argument('--sampling_nonbonded_min_dist', type=float, default=None,
                        help='Sampling-time nonbonded min distance (normalized). None disables')
    parser.add_argument('--sampling_nonbonded_strength', type=float, default=0.5,
                        help='Sampling-time repulsion strength')
    parser.add_argument('--sampling_nonbonded_iters', type=int, default=1,
                        help='Sampling-time repulsion iterations per step')
    parser.add_argument('--sampling_bond_strength', type=float, default=0.5,
                        help='Sampling-time bond projection strength')
    parser.add_argument('--sampling_bond_iters', type=int, default=1,
                        help='Sampling-time bond projection iterations per step')
    parser.add_argument('--sampling_max_step', type=float, default=0.05,
                        help='Clamp per-step displacement in normalized units (<=0 disables)')
    parser.add_argument('--sampling_rescale_each_step', action='store_true',
                        help='Rescale to unit radius after each sampling step')
    parser.add_argument('--use_topology_gnn', action='store_true',
                        help='Use topology GNN to generate adjacency instead of dataset templates')
    parser.add_argument('--topology_gnn_ckpt', type=str, default='checkpoints/topology_gnn.pt',
                        help='Topology GNN checkpoint path')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    print(f"  Checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    # Create model
    print("Creating model...")
    model_config = config['model']
    model = FullereneDiffusionModel(
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        edge_dim=model_config['edge_dim'],
        C_embed_dim=model_config['C_embed_dim'],
        time_embed_dim=model_config['time_embed_dim'],
        # max_C defaults to 100 in model definition
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model loaded successfully")
    
    # Create scheduler
    diff_config = config['diffusion']
    scheduler = DiffusionScheduler(
        num_steps=diff_config['num_steps'],
        beta_schedule=diff_config['beta_schedule'],
        beta_start=diff_config.get('beta_start', 0.0001),
        beta_end=diff_config.get('beta_end', 0.02),
        device=device,
    )
    
    # Generate structures
    print(f"\nGenerating {args.num_samples} samples of C{args.C}...")

    # Template bank (dataset adjacency) and/or topology GNN
    template_bank = None
    topology_model = None
    data_cfg = config.get('data', {})
    xyz_dir = data_cfg.get('xyz_dir')
    if xyz_dir:
        xyz_path = Path(xyz_dir)
        if not xyz_path.is_absolute():
            xyz_path = (Path(__file__).resolve().parent / xyz_path).resolve()
        template_bank = TemplateBank(xyz_dir=xyz_path)

    if args.use_topology_gnn:
        topo_ckpt = torch.load(args.topology_gnn_ckpt, map_location=device)
        topo_cfg = topo_ckpt.get('config', {})
        max_nodes = int(topo_cfg.get('max_C', 720))
        hidden_dim = int(topo_cfg.get('hidden_dim', 64))
        latent_dim = int(topo_cfg.get('latent_dim', 32))
        topology_model = TopologyGNN(max_nodes=max_nodes, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        topology_model.load_state_dict(topo_ckpt['model_state_dict'])
        topology_model.eval()
    elif template_bank is None:
        raise ValueError("Checkpoint config missing data.xyz_dir (needed for template graphs)")

    structures = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        C_value=args.C,
        device=device,
        template_bank=template_bank,
        topology_model=topology_model,
        use_ddim=args.ddim,
        ddim_steps=args.ddim_steps,
        sampling_project_radius=args.sampling_project_radius,
        sampling_nonbonded_min_dist=args.sampling_nonbonded_min_dist,
        sampling_nonbonded_strength=args.sampling_nonbonded_strength,
        sampling_nonbonded_iters=args.sampling_nonbonded_iters,
        sampling_bond_strength=args.sampling_bond_strength,
        sampling_bond_iters=args.sampling_bond_iters,
        sampling_max_step=args.sampling_max_step,
        sampling_rescale_each_step=args.sampling_rescale_each_step,
    )
    
    # Save to files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reject_dir = output_dir / "rejected"
    if args.save_rejects:
        reject_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving structures to {output_dir}/...")
    valid_count = 0
    stats_list = []
    for i, pos in enumerate(structures):
        # Match the same template family for edges
        template = template_bank.make_template_data(args.C, device=torch.device('cpu'))

        pos_work = pos
        if args.rescale_to_unit_radius:
            pos_work = rescale_to_unit_radius(pos_work)

        if args.relax_steps > 0:
            pos_work = relax_structure(
                pos_work,
                template.edge_index,
                args.C,
                steps=args.relax_steps,
                lr=args.relax_lr,
                bond_weight=args.relax_bond_weight,
                radius_weight=args.relax_radius_weight,
                radius_var_weight=args.relax_radius_var_weight,
                nonbonded_min_dist=args.relax_nonbonded_min_dist,
                nonbonded_weight=args.relax_nonbonded_weight,
            )

        pos_to_save = pos_work
        if args.output_scale == 'angstrom':
            pos_to_save = denormalize_positions(pos_to_save, args.C).cpu()

        is_valid, stats = is_valid_structure(
            pos_work,
            template.edge_index,
            args.C,
            bond_mean_tol=args.bond_mean_tol,
            bond_std_max=args.bond_std_max,
            radius_mean_tol=args.radius_mean_tol,
            radius_std_max=args.radius_std_max,
            nonbonded_min_dist=args.filter_nonbonded_min_dist,
            max_nonbonded_violations=args.filter_max_nonbonded_violations,
        )
        stats_list.append(stats)

        if args.filter_invalid and not is_valid:
            if args.save_rejects:
                filepath = reject_dir / f"C{args.C}_sample_{i:03d}.xyz"
                save_xyz(pos_to_save, template.edge_index, filepath)
            continue

        filepath = output_dir / f"C{args.C}_sample_{i:03d}.xyz"
        save_xyz(pos_to_save, template.edge_index, filepath)
        valid_count += 1

    if args.filter_invalid:
        total = len(structures)
        valid_rate = valid_count / total if total > 0 else 0.0
        bond_mean = sum(s["bond_mean"] for s in stats_list) / max(1, total)
        bond_std = sum(s["bond_std"] for s in stats_list) / max(1, total)
        radius_mean = sum(s["radius_mean"] for s in stats_list) / max(1, total)
        radius_std = sum(s["radius_std"] for s in stats_list) / max(1, total)
        print(f"\nValidity summary:")
        print(f"  valid: {valid_count}/{total} ({valid_rate*100:.1f}%)")
        print(f"  avg bond_mean={bond_mean:.4f} bond_std={bond_std:.4f}")
        print(f"  avg radius_mean={radius_mean:.4f} radius_std={radius_std:.4f}")
    
    print("✓ Generation complete!")


if __name__ == '__main__':
    main()
