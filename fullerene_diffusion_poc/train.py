"""
Training script for Fullerene Diffusion Model.

Trains an E(3)-equivariant diffusion model to denoise fullerene coordinates.

Usage:
    python train.py --config config.yaml --epochs 100

Author: InterfaceML Project
Date: 2026-01-28
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from dataset import get_dataloaders
from diffusion_utils import DiffusionScheduler
from model import FullereneDiffusionModel, bond_length_loss, sphericity_loss
# IMPROVEMENT: Use v2 time-conditioned losses only
from topology_loss_v2 import combined_physics_loss_v2
from train_topology_gnn import train_topology_gnn


class Trainer:
    """Handles training loop, validation, and checkpointing."""
    
    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device
        
        # Create dataloaders
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            config, num_workers=config.get('num_workers', 0)
        )
        
        # Create model
        print("Creating model...")
        model_config = config['model']
        self.model = FullereneDiffusionModel(
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            edge_dim=model_config['edge_dim'],
            C_embed_dim=model_config['C_embed_dim'],
            time_embed_dim=model_config['time_embed_dim'],
        ).to(device)
        
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create diffusion scheduler
        diff_config = config['diffusion']
        self.scheduler = DiffusionScheduler(
            num_steps=diff_config['num_steps'],
            beta_schedule=diff_config['beta_schedule'],
            beta_start=diff_config.get('beta_start', 0.0001),
            beta_end=diff_config.get('beta_end', 0.02),
            device=device,
        )
        
        # Optimizer
        train_config = config['training']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )
        
        # Learning rate scheduler (cosine annealing with warmup)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config['epochs'],
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_config = train_config
        self.loss_config = config.get('loss', {})  # Add loss config
        
        # Create checkpoint directory
        ckpt_dir = self.train_config.get('checkpoint_dir', 'checkpoints')
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_noise_loss = 0.0
        total_bond_loss = 0.0
        total_sphere_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Sample timesteps (bias toward low-noise for geometry constraints)
            batch_size = batch.C.size(0)
            low_t_fraction = self.loss_config.get('low_t_fraction', 0.5)
            low_t_max = int(self.loss_config.get('low_t_max', 200))
            low_t_max = max(1, min(low_t_max, self.scheduler.num_steps - 1))

            if low_t_fraction > 0:
                t_uniform = torch.randint(
                    0, self.scheduler.num_steps, (batch_size,),
                    device=self.device, dtype=torch.long
                )
                t_low = torch.randint(
                    0, low_t_max + 1, (batch_size,),
                    device=self.device, dtype=torch.long
                )
                choose_low = torch.rand(batch_size, device=self.device) < low_t_fraction
                t = torch.where(choose_low, t_low, t_uniform)
            else:
                t = torch.randint(
                    0, self.scheduler.num_steps, (batch_size,),
                    device=self.device, dtype=torch.long
                )
            
            # Forward diffusion: add noise manually for PyG batch format
            noise = torch.randn_like(batch.pos)
            t_expanded = t[batch.batch]  # [N] - timestep for each node
            sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_expanded].view(-1, 1)
            sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t_expanded].view(-1, 1)
            pos_noisy = sqrt_alpha * batch.pos + sqrt_one_minus_alpha * noise
            
            # Predict noise
            noise_pred = self.model(
                pos_noisy,
                batch.edge_index,
                t,
                batch.C.squeeze(),
                batch.batch,
            )
            
            # Compute losses
            noise_loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Physics-informed losses with TIME-CONDITIONING (v2)
            lambda_bond = self.loss_config.get('lambda_bond', 0.0)
            lambda_sphere = self.loss_config.get('lambda_sphere', 0.0)
            lambda_topo = self.loss_config.get('lambda_topology', 0.0)
            lambda_conn = self.loss_config.get('lambda_connectivity', 0.0)
            lambda_repulsion = self.loss_config.get('lambda_repulsion', 0.0)

            target_bond_angstrom = self.loss_config.get('target_bond', 1.42)
            bond_tolerance_angstrom = self.loss_config.get('bond_tolerance', 0.30)
            radius_coeff = self.loss_config.get('radius_coeff', 0.45)
            repulsion_min_dist_angstrom = self.loss_config.get('repulsion_min_dist', 1.60)
            
            b_loss = torch.tensor(0.0, device=self.device)
            s_loss = torch.tensor(0.0, device=self.device)
            t_loss = torch.tensor(0.0, device=self.device)
            c_loss = torch.tensor(0.0, device=self.device)
            
            if lambda_bond > 0 or lambda_sphere > 0 or lambda_topo > 0 or lambda_conn > 0:
                # Predict clean coordinates
                alpha_bar = self.scheduler.alphas_cumprod[t[batch.batch]].view(-1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
                pos_pred = (pos_noisy - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
                
                # Use v2 time-conditioned physics losses
                # This automatically reduces constraint strength at high noise
                physics_losses = combined_physics_loss_v2(
                    pos_pred,
                    batch.edge_index,
                    batch.batch,
                    batch.C,
                    t,  # Pass timesteps for time-weighting
                    lambda_bond=lambda_bond,
                    lambda_conn=lambda_conn,
                    lambda_topo=lambda_topo,
                    lambda_repulsion=lambda_repulsion,
                    target_bond_angstrom=target_bond_angstrom,
                    bond_tolerance_angstrom=bond_tolerance_angstrom,
                    radius_coeff=radius_coeff,
                    repulsion_min_dist_angstrom=repulsion_min_dist_angstrom,
                )
                
                b_loss = physics_losses['bond']
                c_loss = physics_losses['connectivity']
                t_loss = physics_losses['topology']
                
                # Sphere loss (not time-conditioned, works at all noise levels)
                if lambda_sphere > 0:
                    # Training coordinates are normalized to mean radius ~ 1.0.
                    s_loss = sphericity_loss(pos_pred, batch.batch, target_radius=1.0)
            
            # Total loss: noise reconstruction + weighted physics constraints
            loss = noise_loss + lambda_sphere * s_loss + physics_losses.get('total_physics', 0.0)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.loss_config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.loss_config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_noise_loss += noise_loss.item()
            total_bond_loss += b_loss.item()
            total_sphere_loss += s_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'noise': f"{noise_loss.item():.4f}",
                'bond': f"{b_loss.item():.4f}",
                'sphere': f"{s_loss.item():.4f}",
                'topo': f"{t_loss.item():.4f}",
                'conn': f"{c_loss.item():.4f}",
            })
        
        return {
            'loss': total_loss / num_batches,
            'noise_loss': total_noise_loss / num_batches,
            'bond_loss': total_bond_loss / num_batches,
            'sphere_loss': total_sphere_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = batch.to(self.device)
            
            batch_size = batch.C.size(0)
            t = torch.randint(
                0, self.scheduler.num_steps, (batch_size,),
                device=self.device, dtype=torch.long
            )
            
            noise = torch.randn_like(batch.pos)
            t_expanded = t[batch.batch]
            sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_expanded].view(-1, 1)
            sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t_expanded].view(-1, 1)
            pos_noisy = sqrt_alpha * batch.pos + sqrt_one_minus_alpha * noise
            
            noise_pred = self.model(
                pos_noisy,
                batch.edge_index,
                t,
                batch.C.squeeze(),
                batch.batch,
            )
            
            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': total_loss / num_batches}
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save model checkpoint."""
        ckpt_path = self.ckpt_dir / name
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs...\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if self.epoch % self.train_config.get('eval_interval', 1) == 0:
                val_metrics = self.validate()
                
                # Print metrics
                print(f"\nEpoch {self.epoch}/{num_epochs}")
                print(f"  Train loss: {train_metrics['loss']:.6f}")
                print(f"  Val loss:   {val_metrics['val_loss']:.6f}")
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
                    print("  ✓ New best model!")
            
            # Save periodic checkpoint
            if self.epoch % self.train_config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pt')
            
            # Update learning rate
            self.lr_scheduler.step()
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train Fullerene Diffusion Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve data paths relative to config file
    config_dir = Path(args.config).resolve().parent
    data_cfg = config.get('data', {})
    for key in ['xyz_dir', 'analysis_dir', 'split_csv']:
        path_val = data_cfg.get(key)
        if path_val and not Path(path_val).is_absolute():
            data_cfg[key] = str((config_dir / path_val).resolve())
    config['data'] = data_cfg

    # Resolve checkpoint paths relative to config file
    training_cfg = config.get('training', {})
    ckpt_dir = training_cfg.get('checkpoint_dir')
    if not ckpt_dir:
        ckpt_dir = str((config_dir / 'checkpoints').resolve())
        training_cfg['checkpoint_dir'] = ckpt_dir
    elif not Path(ckpt_dir).is_absolute():
        training_cfg['checkpoint_dir'] = str((config_dir / ckpt_dir).resolve())
    config['training'] = training_cfg

    topo_cfg = config.get('topology_gnn', {})
    topo_ckpt = topo_cfg.get('checkpoint_path')
    if topo_ckpt:
        topo_cfg['checkpoint_path'] = str((config_dir / topo_ckpt).resolve()) if not Path(topo_ckpt).is_absolute() else topo_ckpt
        config['topology_gnn'] = topo_cfg
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('device') == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Optional: train topology GNN before diffusion
    topo_cfg = config.get('topology_gnn', {})
    if topo_cfg.get('enabled', False) and topo_cfg.get('train_before_diffusion', True):
        print("Training topology GNN before diffusion...")
        train_topology_gnn(config, device)

    # Create trainer
    trainer = Trainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(config['training']['epochs'])


if __name__ == '__main__':
    main()
