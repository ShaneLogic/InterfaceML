"""
Generation script for Fullerene Diffusion Model.

Samples new fullerene structures from trained diffusion model.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --C 60 --num_samples 10

Author: InterfaceML Project
Date: 2026-01-28
"""

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch_geometric.data import Data, Batch

from diffusion_utils import DiffusionScheduler
from model import FullereneDiffusionModel


def create_template_graph(num_carbon: int, device: str) -> Data:
    """
    Create a template graph structure for fullerene generation.
    
    Uses a simple heuristic: start from random positions and 
    construct 3-regular graph (each node has 3 neighbors).
    
    Args:
        num_carbon: Number of carbon atoms (C value)
        device: Device to create tensors on
    
    Returns:
        PyG Data object with random positions and edge_index
    """
    # Random initial positions (will be replaced by diffusion)
    pos = torch.randn(num_carbon, 3, device=device)
    
    # Create 3-regular graph using greedy approach
    # (In practice, you might load a template or use graph generation algorithm)
    edge_index = []
    degrees = [0] * num_carbon
    
    # Simple heuristic: connect nearby nodes until each has degree 3
    for i in range(num_carbon):
        if degrees[i] >= 3:
            continue
        
        # Find nearest unconnected nodes
        candidates = []
        for j in range(num_carbon):
            if i != j and degrees[j] < 3:
                # Check if not already connected
                if not any((e[0] == i and e[1] == j) or (e[0] == j and e[1] == i) for e in edge_index):
                    candidates.append(j)
        
        # Connect to first available candidates
        for j in candidates:
            if degrees[i] >= 3:
                break
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected
            degrees[i] += 1
            degrees[j] += 1
    
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
    
    # Create Data object
    data = Data(pos=pos, edge_index=edge_index)
    
    return data


def generate_samples(
    model: FullereneDiffusionModel,
    scheduler: DiffusionScheduler,
    num_samples: int,
    C_value: int,
    device: str,
    use_ddim: bool = False,
    ddim_steps: int = 100,
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
    
    Returns:
        List of generated positions (each is [C, 3] tensor)
    """
    model.eval()
    
    generated_structures = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Generating C{C_value}"):
            # Create template graph
            template = create_template_graph(C_value, device)
            
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
                
                # DDPM reverse step using posterior q(x_{t-1} | x_t, x_0)
                # First, predict x_0 from x_t
                sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t]
                x_0_pred = (pos_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                
                # Adaptive clipping based on timestep (stricter at later steps)
                # Training data has std≈1, use dynamic range based on remaining noise
                # At t=999: more noise remains, use wider range
                # At t=0: almost clean, use tighter range
                clip_factor = 3.0 + 2.0 * (t.float() / scheduler.num_steps)
                x_0_pred = torch.clamp(x_0_pred, -clip_factor, clip_factor)
                
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
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Generated fullerene structure\n")
        
        for i in range(num_atoms):
            x, y, z = pos[i].tolist()
            nb1, nb2, nb3 = neighbors[i][:3]
            f.write(f"C {x:.6f} {y:.6f} {z:.6f} {i} {nb1} {nb2} {nb3}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Fullerene Structures')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--C', type=int, required=True, help='Number of carbon atoms (20-720)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of structures to generate')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output directory')
    parser.add_argument('--ddim', action='store_true', help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=100, help='DDIM sampling steps')
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
    structures = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        C_value=args.C,
        device=device,
        use_ddim=args.ddim,
        ddim_steps=args.ddim_steps,
    )
    
    # Save to files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving structures to {output_dir}/...")
    for i, pos in enumerate(structures):
        # Create template graph for edge_index
        template = create_template_graph(args.C, 'cpu')
        
        filepath = output_dir / f"C{args.C}_sample_{i:03d}.xyz"
        save_xyz(pos, template.edge_index, filepath)
    
    print("✓ Generation complete!")


if __name__ == '__main__':
    main()
