"""
Complete training, generation, and evaluation pipeline for fullerene diffusion model.

This script provides an end-to-end workflow:
1. Train the model with optimized hyperparameters
2. Generate structures during training for monitoring
3. Evaluate quality metrics
4. Save best models

Usage:
    python run_complete_pipeline.py --mode train
    python run_complete_pipeline.py --mode generate --checkpoint checkpoints/best_model.pt
    python run_complete_pipeline.py --mode evaluate --generated_dir generated/

Author: InterfaceML Project
Date: 2026-01-30
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.distance import pdist


def check_structure_quality(coords):
    """
    Evaluate the quality of generated structure.
    
    Args:
        coords: [N, 3] numpy array of atomic coordinates
    
    Returns:
        metrics: dict with quality metrics
    """
    # Center coordinates
    coords_centered = coords - coords.mean(axis=0)
    
    # Compute pairwise distances
    dists = pdist(coords_centered)
    
    # Find nearest neighbors for each atom (approximate connectivity)
    from scipy.spatial import distance_matrix
    dist_matrix = distance_matrix(coords_centered, coords_centered)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_dists = np.sort(dist_matrix, axis=1)[:, :3]  # 3 nearest neighbors
    
    # Compute metrics
    metrics = {
        # Basic statistics
        'coord_range': coords_centered.max() - coords_centered.min(),
        'coord_std': coords_centered.std(),
        
        # Distance statistics
        'min_distance': dists.min(),
        'mean_distance': dists.mean(),
        'bond_distances_mean': nearest_dists.mean(),
        'bond_distances_std': nearest_dists.std(),
        
        # Bond quality (C-C should be 1.39-1.45 Å)
        'ideal_bonds': ((dists > 1.35) & (dists < 1.50)).sum(),
        'close_contacts': (dists < 1.0).sum(),  # Too close
        
        # Sphericity (radius of gyration)
        'radius_gyration': np.sqrt((coords_centered**2).sum(axis=1).mean()),
        
        # Check for collapsed structure
        'is_collapsed': coords_centered.std() < 1.0,
        'is_exploded': coords_centered.std() > 10.0,
    }
    
    # Quality score (0-1, higher is better)
    score = 0.0
    if 1.35 < metrics['bond_distances_mean'] < 1.50:
        score += 0.3
    if metrics['bond_distances_std'] < 0.15:
        score += 0.2
    if 3.0 < metrics['radius_gyration'] < 5.0:
        score += 0.3
    if not metrics['is_collapsed'] and not metrics['is_exploded']:
        score += 0.2
    
    metrics['quality_score'] = score
    
    return metrics


def train_with_monitoring(config_path='config.yaml', epochs=50, checkpoint_interval=10):
    """
    Train model with periodic generation for quality monitoring.
    """
    print("="*80)
    print("FULLERENE DIFFUSION MODEL - COMPLETE TRAINING PIPELINE")
    print("="*80)
    print()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs
    config['training']['epochs'] = epochs
    
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Hidden dim: {config['model']['hidden_dim']}")
    print(f"  Num layers: {config['model']['num_layers']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Loss weights: bond={config['loss']['lambda_bond']}, sphere={config['loss']['lambda_sphere']}")
    print()
    
    # Import training module
    from train import Trainer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Create trainer
    trainer = Trainer(config, device)
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    - Noise: {train_metrics['noise_loss']:.4f}")
        print(f"    - Bond: {train_metrics['bond_loss']:.4f}")
        print(f"    - Sphere: {train_metrics['sphere_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Generate samples periodically
        if (epoch + 1) % checkpoint_interval == 0:
            print(f"\n  Generating test structures...")
            generate_and_evaluate(
                trainer.model,
                trainer.scheduler,
                device,
                num_samples=5,
                output_dir=f"generated/epoch_{epoch+1}",
                verbose=False
            )
        
        # Update learning rate
        trainer.lr_scheduler.step()
        
        # Increment epoch
        trainer.epoch += 1
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best model saved to: {trainer.ckpt_dir / 'best_model.pt'}")
    print(f"{'='*80}\n")
    
    return trainer


def generate_and_evaluate(model, scheduler, device, num_samples=10, num_atoms=60, 
                          output_dir='generated', verbose=True):
    """
    Generate structures and evaluate quality.
    """
    from generate import generate_samples
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GENERATING {num_samples} C{num_atoms} STRUCTURES")
        print(f"{'='*80}\n")
    
    # Generate using correct parameters
    structures = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        C_value=num_atoms,
        device=device,
        use_ddim=False,
        ddim_steps=100,
    )
    
    # Save structures to XYZ files
    for i, coords in enumerate(structures):
        xyz_path = output_dir / f'C{num_atoms}_sample_{i:03d}.xyz'
        num_atoms_actual = coords.size(0)
        with open(xyz_path, 'w') as f:
            f.write(f"{num_atoms_actual}\n")
            f.write(f"Generated C{num_atoms} structure\n")
            for atom_idx in range(num_atoms_actual):
                x, y, z = coords[atom_idx].tolist()
                f.write(f"C {x:.6f} {y:.6f} {z:.6f}\n")
    
    # Evaluate each structure
    quality_scores = []
    metrics_list = []
    
    for i, coords in enumerate(structures):
        # Convert torch tensor to numpy
        coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords
        metrics = check_structure_quality(coords_np)
        metrics_list.append(metrics)
        quality_scores.append(metrics['quality_score'])
        
        if verbose:
            print(f"Structure {i+1}/{num_samples}:")
            print(f"  Quality Score: {metrics['quality_score']:.3f}/1.0")
            print(f"  Bond distances: {metrics['bond_distances_mean']:.3f} ± {metrics['bond_distances_std']:.3f} Å")
            print(f"  Radius of gyration: {metrics['radius_gyration']:.3f} Å")
            if metrics['is_collapsed']:
                print(f"  ⚠️  WARNING: Structure appears collapsed")
            if metrics['is_exploded']:
                print(f"  ⚠️  WARNING: Structure appears exploded")
            print()
    
    # Summary
    quality_scores = np.array(quality_scores)
    
    if verbose:
        print(f"{'='*80}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"  Average Quality: {quality_scores.mean():.3f} ± {quality_scores.std():.3f}")
        print(f"  Best Quality: {quality_scores.max():.3f}")
        print(f"  Worst Quality: {quality_scores.min():.3f}")
        print(f"  Structures saved to: {output_dir}")
        print(f"{'='*80}\n")
    
    # Save metrics (convert numpy types to python native for JSON)
    def convert_to_native(obj):
        """Recursively convert numpy types to python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'summary': {
                'mean_quality': float(quality_scores.mean()),
                'std_quality': float(quality_scores.std()),
                'max_quality': float(quality_scores.max()),
                'min_quality': float(quality_scores.min()),
            },
            'individual_metrics': convert_to_native(metrics_list),
        }, f, indent=2)
    
    return structures, metrics_list


def evaluate_directory(generated_dir='generated'):
    """
    Evaluate all structures in a directory.
    """
    generated_dir = Path(generated_dir)
    xyz_files = sorted(generated_dir.glob('*.xyz'))
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {len(xyz_files)} STRUCTURES")
    print(f"{'='*80}\n")
    
    if not xyz_files:
        print(f"No XYZ files found in {generated_dir}")
        return
    
    metrics_list = []
    
    for xyz_file in xyz_files:
        # Read coordinates
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            n_atoms = int(lines[0].strip())
            coords = []
            for line in lines[2:2+n_atoms]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        coords = np.array(coords)
        metrics = check_structure_quality(coords)
        metrics['filename'] = xyz_file.name
        metrics_list.append(metrics)
        
        print(f"{xyz_file.name}:")
        print(f"  Quality: {metrics['quality_score']:.3f}")
        print(f"  Bonds: {metrics['bond_distances_mean']:.3f} ± {metrics['bond_distances_std']:.3f} Å")
        print(f"  R_g: {metrics['radius_gyration']:.3f} Å")
        print()
    
    # Summary
    quality_scores = np.array([m['quality_score'] for m in metrics_list])
    
    print(f"{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Files evaluated: {len(xyz_files)}")
    print(f"  Average quality: {quality_scores.mean():.3f} ± {quality_scores.std():.3f}")
    print(f"  Best: {quality_scores.max():.3f}")
    print(f"  Worst: {quality_scores.min():.3f}")
    print(f"{'='*80}\n")
    
    # Save report (convert numpy types to python native)
    def convert_to_native(obj):
        """Recursively convert numpy types to python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    report_path = generated_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'num_files': int(len(xyz_files)),
                'mean_quality': float(quality_scores.mean()),
                'std_quality': float(quality_scores.std()),
                'max_quality': float(quality_scores.max()),
                'min_quality': float(quality_scores.min()),
            },
            'individual_metrics': convert_to_native(metrics_list),
        }, f, indent=2)
    
    print(f"Saved evaluation report to: {report_path}")
    
    print(f"Report saved to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Fullerene Diffusion Model Pipeline')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'generate', 'evaluate'],
                        help='Mode: train/generate/evaluate')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Checkpoint path for generation')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to generate')
    parser.add_argument('--num_atoms', type=int, default=60,
                        help='Number of carbon atoms')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Output directory for generated structures')
    parser.add_argument('--generated_dir', type=str, default='generated',
                        help='Directory with generated structures to evaluate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_with_monitoring(args.config, args.epochs)
    
    elif args.mode == 'generate':
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        from model import FullereneDiffusionModel
        from diffusion_utils import DiffusionScheduler
        
        model = FullereneDiffusionModel(
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            edge_dim=config['model']['edge_dim'],
            C_embed_dim=config['model']['C_embed_dim'],
            time_embed_dim=config['model']['time_embed_dim'],
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        scheduler = DiffusionScheduler(
            num_steps=config['diffusion']['num_steps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            device=device,
        )
        
        generate_and_evaluate(
            model, scheduler, device,
            num_samples=args.num_samples,
            num_atoms=args.num_atoms,
            output_dir=args.output_dir,
        )
    
    elif args.mode == 'evaluate':
        evaluate_directory(args.generated_dir)


if __name__ == '__main__':
    main()
