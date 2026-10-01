"""
Training script for Fullerene Diffusion Model.

Trains an E(3)-equivariant diffusion model to denoise fullerene coordinates.

Usage:
    python train.py --config config.yaml --epochs 100

Author: InterfaceML Project
Date: 2026-01-28
"""

import argparse
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

import numpy as np

from dataset import get_dataloaders
from diffusion_utils import DiffusionScheduler
from model import FullereneDiffusionModel, bond_length_loss, sphericity_loss
# IMPROVEMENT: Use v2 time-conditioned losses only
from topology_loss_v2 import combined_physics_loss_v2
from train_topology_gnn import train_topology_gnn
from flow_matching import FlowMatchingScheduler


class Trainer:
    """Handles training loop, validation, and checkpointing."""
    
    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device
        
        # Create dataloaders
        logger.info("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            config, num_workers=config.get('num_workers', 0)
        )
        
        # Create model — dispatch on architecture type
        logger.info("Creating model...")
        model_config = config['model']
        self.architecture = model_config.get('architecture', 'egnn')

        if self.architecture == 'painn':
            from model_painn import FullerenePaiNNModel
            self.model = FullerenePaiNNModel(
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 8),
                num_rbf=model_config.get('num_rbf', 20),
                cutoff=model_config.get('cutoff', 5.0),
                time_embed_dim=model_config.get('time_embed_dim', 128),
                C_embed_dim=model_config.get('C_embed_dim', 64),
                C_fourier_features=model_config.get('C_fourier_features', 16),
                max_C=model_config.get('max_C', 720),
                cfg_drop_prob=model_config.get('cfg_drop_prob', 0.1),
                num_atom_types=model_config.get('num_atom_types', 1),
                use_pbc=model_config.get('use_pbc', False),
            ).to(device)
        elif self.architecture == 'egnn':
            self.model = FullereneDiffusionModel(
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                edge_dim=model_config.get('edge_dim', 0),
                C_embed_dim=model_config.get('C_embed_dim', 64),
                time_embed_dim=model_config.get('time_embed_dim', 128),
                max_C=model_config.get('max_C', config.get('data', {}).get('C_max', 720)),
                continuous_C_embed=model_config.get('continuous_C_embed', False),
                C_fourier_features=model_config.get('C_fourier_features', 16),
                use_hierarchical=model_config.get('use_hierarchical', False),
                hierarchical_threshold=model_config.get('hierarchical_threshold', 80),
                hierarchical_layers=model_config.get('hierarchical_layers', 2),
                use_global_attention=model_config.get('use_global_attention', False),
                global_attention_heads=model_config.get('global_attention_heads', 4),
                use_distance_weight=model_config.get('use_distance_weight', True),
                use_degree_norm=model_config.get('use_degree_norm', True),
            ).to(device)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create diffusion scheduler — dispatch on type
        diff_config = config['diffusion']
        self.diffusion_type = diff_config.get('type', 'ddpm')
        self.flow_matching = self.diffusion_type == 'flow_matching'

        if self.flow_matching:
            self.fm_scheduler = FlowMatchingScheduler()
            # Create a DDPM scheduler too for backward compat (validation, etc.)
            self.scheduler = DiffusionScheduler(
                num_steps=diff_config.get('num_steps', 1000),
                beta_schedule=diff_config.get('beta_schedule', 'cosine'),
                beta_start=diff_config.get('beta_start', 0.0001),
                beta_end=diff_config.get('beta_end', 0.02),
                device=device,
            )
            logger.info("Using Flow Matching diffusion (continuous t)")
        else:
            self.scheduler = DiffusionScheduler(
                num_steps=diff_config['num_steps'],
                beta_schedule=diff_config['beta_schedule'],
                beta_start=diff_config.get('beta_start', 0.0001),
                beta_end=diff_config.get('beta_end', 0.02),
                device=device,
            )
            self.fm_scheduler = None
            logger.info("Using DDPM diffusion (discrete t)")
        
        # Optimizer
        train_config = config['training']
        self.train_config = train_config
        self.loss_config = config.get('loss', {})
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
        )
        
        # Learning rate scheduler: cosine annealing with optional warmup
        warmup_epochs = self.loss_config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, train_config['epochs'] - warmup_epochs),
            )
            self.lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            logger.info("LR schedule: %d warmup epochs → cosine annealing", warmup_epochs)
        else:
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs'],
            )
        
        # EMA (Exponential Moving Average) for improved generation quality
        self.ema_decay = config.get('training', {}).get('ema_decay', 0.9999)
        self.ema_model = deepcopy(self.model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        logger.info("EMA enabled with decay=%.6f", self.ema_decay)
        
        # Fixed RNG for deterministic validation (CPU generator — MPS doesn't support torch.Generator)
        self._val_rng = torch.Generator(device='cpu')
        self._val_rng.manual_seed(42)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.nan_skip_count = 0  # Track NaN skips for monitoring
        
        # Create checkpoint directory
        ckpt_dir = self.train_config.get('checkpoint_dir', 'checkpoints')
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Template bank for structural validation during training
        self._template_bank = None
        if self.loss_config.get('struct_eval_interval', 0) > 0:
            try:
                from template_bank import TemplateBank
                data_cfg = config.get('data', {})
                xyz_dir = data_cfg.get('xyz_dir')
                if xyz_dir:
                    xyz_path = Path(xyz_dir)
                    if not xyz_path.is_absolute():
                        xyz_path = (Path(__file__).resolve().parent / xyz_path).resolve()
                    if xyz_path.exists():
                        self._template_bank = TemplateBank(xyz_dir=xyz_path)
                        logger.info("Template bank loaded for structural validation")
            except Exception as exc:
                logger.warning("Could not init template bank for struct eval: %s", exc)
    
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
            
            batch_size = batch.C.size(0)

            if self.flow_matching:
                # --- Flow Matching path ---
                # Sample continuous t ~ U(0, 1)
                t_cont = self.fm_scheduler.sample_t(batch_size, device=self.device)  # [B]
                t_per_node = t_cont[batch.batch].unsqueeze(-1)  # [N, 1]

                noise = torch.randn_like(batch.pos)
                pos_noisy, target_v = self.fm_scheduler.add_noise(batch.pos, t_per_node, noise)

                # Convert continuous t to integer scale for model's time embedding
                t_int = (t_cont * 999).long()

                # Predict velocity field
                v_pred = self.model(
                    pos_noisy,
                    batch.edge_index,
                    t_int,
                    batch.C.view(-1),
                    batch.batch,
                )

                noise_loss = nn.functional.mse_loss(v_pred, target_v)
                noise_pred = v_pred  # for physics loss: use v_pred as proxy
                t = t_int  # for physics loss time conditioning

                # For physics loss: approximate clean pos from flow matching
                # x_0 ≈ x_t - t * v_pred (rearranging x_t = (1-t)*x_0 + t*noise)
                sqrt_alpha = (1.0 - t_per_node)  # for compat with physics loss below
                sqrt_one_minus_alpha = t_per_node

            else:
                # --- DDPM path (unchanged) ---
                # Sample timesteps (bias toward low-noise for geometry constraints)
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
                    batch.C.view(-1),
                    batch.batch,
                )

                noise_loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Physics-informed losses with TIME-CONDITIONING (v2)
            lambda_bond = self.loss_config.get('lambda_bond', 0.0)
            lambda_sphere = self.loss_config.get('lambda_sphere', 0.0)
            lambda_topo = self.loss_config.get('lambda_topology', 0.0)
            lambda_conn = self.loss_config.get('lambda_connectivity', 0.0)
            lambda_repulsion = self.loss_config.get('lambda_repulsion', 0.0)
            lambda_angle = self.loss_config.get('lambda_angle', 0.0)

            target_bond_angstrom = self.loss_config.get('target_bond', 1.42)
            bond_tolerance_angstrom = self.loss_config.get('bond_tolerance', 0.30)
            radius_coeff = self.loss_config.get('radius_coeff', 0.45)
            repulsion_min_dist_angstrom = self.loss_config.get('repulsion_min_dist', 1.60)
            
            b_loss = torch.tensor(0.0, device=self.device)
            s_loss = torch.tensor(0.0, device=self.device)
            t_loss = torch.tensor(0.0, device=self.device)
            c_loss = torch.tensor(0.0, device=self.device)
            total_physics_loss = torch.tensor(0.0, device=self.device)
            
            use_physics = (lambda_bond > 0 or lambda_sphere > 0
                           or lambda_topo > 0 or lambda_conn > 0
                           or lambda_angle > 0)
            if use_physics:
                # Predict clean coordinates from noisy x_t and predicted noise
                # Reuse precomputed sqrt_alpha and sqrt_one_minus_alpha
                pos_pred = (pos_noisy - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
                
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
                    lambda_angle=lambda_angle,
                    target_bond_angstrom=target_bond_angstrom,
                    bond_tolerance_angstrom=bond_tolerance_angstrom,
                    radius_coeff=radius_coeff,
                    repulsion_min_dist_angstrom=repulsion_min_dist_angstrom,
                )
                
                b_loss = physics_losses['bond']
                c_loss = physics_losses['connectivity']
                t_loss = physics_losses['topology']
                total_physics_loss = physics_losses.get('total_physics', torch.tensor(0.0, device=self.device))
                
                # Sphere loss — optionally time-conditioned (v5)
                if lambda_sphere > 0:
                    # Training coordinates are normalized to mean radius ~ 1.0.
                    s_loss = sphericity_loss(pos_pred, batch.batch, target_radius=1.0)
                    # Apply time-conditioning if enabled
                    if self.loss_config.get('sphere_time_conditioned', False):
                        from topology_loss_v2 import compute_time_weight
                        sphere_tw = compute_time_weight(t, T=self.scheduler.num_steps, mode='sigmoid').mean()
                        s_loss = s_loss * sphere_tw
            
            # Total loss: noise reconstruction + weighted physics constraints
            loss = noise_loss + lambda_sphere * s_loss + total_physics_loss

            # NaN guard: skip this batch if loss is NaN/Inf
            if not torch.isfinite(loss):
                logger.warning(
                    "NaN/Inf loss detected at epoch %d batch %d — skipping",
                    self.epoch, num_batches,
                )
                self.optimizer.zero_grad()
                num_batches += 1
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # NaN gradient guard: skip if any gradient is NaN
            has_nan_grad = False
            if self.loss_config.get('nan_guard', True):
                for p in self.model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan_grad = True
                        break
            if has_nan_grad:
                logger.warning(
                    "NaN gradient detected at epoch %d batch %d — skipping",
                    self.epoch, num_batches,
                )
                self.optimizer.zero_grad()
                num_batches += 1
                continue
            
            # Gradient clipping
            if self.loss_config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.loss_config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update EMA weights
            self._update_ema()
            
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
    def _update_ema(self):
        """Update EMA model parameters after each optimizer step."""
        for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p_model.data, alpha=1.0 - self.ema_decay)
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = batch.to(self.device)
            
            batch_size = batch.C.size(0)
            noise = torch.randn_like(batch.pos)

            if self.flow_matching:
                # Flow Matching validation: use fixed t values for reproducibility
                t_cont = torch.linspace(0.1, 0.9, batch_size, device=self.device)
                t_per_node = t_cont[batch.batch].unsqueeze(-1)
                pos_noisy, target_v = self.fm_scheduler.add_noise(batch.pos, t_per_node, noise)
                t_int = (t_cont * 999).long()

                v_pred = self.model(
                    pos_noisy,
                    batch.edge_index,
                    t_int,
                    batch.C.view(-1),
                    batch.batch,
                )
                loss = nn.functional.mse_loss(v_pred, target_v)
            else:
                # DDPM validation
                t = torch.randint(
                    0, self.scheduler.num_steps, (batch_size,),
                    device='cpu', dtype=torch.long,
                    generator=self._val_rng,
                ).to(self.device)

                t_expanded = t[batch.batch]
                sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_expanded].view(-1, 1)
                sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t_expanded].view(-1, 1)
                pos_noisy = sqrt_alpha * batch.pos + sqrt_one_minus_alpha * noise

                noise_pred = self.model(
                    pos_noisy,
                    batch.edge_index,
                    t,
                    batch.C.view(-1),
                    batch.batch,
                )
                loss = nn.functional.mse_loss(noise_pred, noise)

            total_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    @torch.no_grad()
    def validate_structures(self) -> dict:
        """Generate sample structures with EMA model and compute quality metrics.

        This provides structural quality feedback during training without
        waiting for a full evaluation run.

        Returns dict with per-C metrics: bond_mean, bond_std, validity, degree3_frac.
        """
        from generate import (
            create_template_graph,
            compute_bond_stats,
            is_valid_structure,
            sample_structure,
        )

        if self._template_bank is None:
            return {}

        C_values = self.loss_config.get('struct_eval_C_values', [60])
        num_samples = self.loss_config.get('struct_eval_samples', 3)
        results = {}

        self.ema_model.eval()

        for C_val in C_values:
            bond_means = []
            bond_stds = []
            valid_count = 0
            degree3_count = 0
            total = 0

            for _ in range(num_samples):
                try:
                    template = create_template_graph(
                        C_val, self.device, self._template_bank, None
                    )
                    pos, _ = sample_structure(
                        self.ema_model,
                        self.scheduler,
                        template,
                        C_value=C_val,
                        use_ddim=False,
                        loss_config={'guidance_scale': 0},
                    )
                    bm, bs = compute_bond_stats(pos, template.edge_index)
                    bond_means.append(bm)
                    bond_stds.append(bs)

                    ok, stats = is_valid_structure(
                        pos, template.edge_index, C_val,
                        bond_mean_tol=0.15,
                        bond_std_max=0.15,
                        radius_mean_tol=0.2,
                        radius_std_max=0.2,
                    )
                    if ok:
                        valid_count += 1

                    # Check degree-3 fraction from edge_index
                    row = template.edge_index[0]
                    degrees = torch.zeros(C_val, device=row.device)
                    degrees.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
                    if (degrees == 3).all():
                        degree3_count += 1

                    total += 1
                except Exception as exc:
                    logger.warning("Struct eval failed for C%d: %s", C_val, exc)

            if total > 0:
                results[f"C{C_val}"] = {
                    "bond_mean": float(np.mean(bond_means)) if bond_means else 0.0,
                    "bond_std": float(np.mean(bond_stds)) if bond_stds else 0.0,
                    "validity": valid_count / total,
                    "degree3_frac": degree3_count / total,
                }

        return results

    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save model checkpoint."""
        ckpt_path = self.ckpt_dir / name
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)
    
    def load_checkpoint(self, ckpt_path: str):
        """Load model checkpoint with automatic v2→v3 migration.

        Handles the key renaming from the old flat ``egnn_layers.{i}`` layout
        to the new split ``egnn_layers_lower.{i}`` / ``egnn_layers_upper.{j}``
        layout introduced by Plan D (hierarchical message passing).

        New modules (C_embed_continuous, coarsen, uncoarsen, egnn_layers_coarse)
        are left at their random initialisation — they will be learned during
        fine-tuning.
        """
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        # Detect whether the current model uses the old EGNN hierarchical layout
        needs_egnn_migration = hasattr(self.model, 'egnn_layers_lower')

        if needs_egnn_migration:
            # --- Migrate egnn_layers.{i} → egnn_layers_lower/upper ---
            migrated = {}
            num_lower = len(self.model.egnn_layers_lower)
            for key, value in state_dict.items():
                if key.startswith('egnn_layers.'):
                    parts = key.split('.', 2)
                    layer_idx = int(parts[1])
                    rest = parts[2]
                    if layer_idx < num_lower:
                        new_key = 'egnn_layers_lower.{}.{}'.format(layer_idx, rest)
                    else:
                        new_key = 'egnn_layers_upper.{}.{}'.format(layer_idx - num_lower, rest)
                    migrated[new_key] = value
                    logger.info("  Migrated %s → %s", key, new_key)
                else:
                    migrated[key] = value

            # --- Pre-fit ContinuousCEmbedding to match old discrete C_embed ---
            old_C_weight = migrated.pop('C_embed.weight', None)
            if old_C_weight is not None and hasattr(self.model, 'C_embed_continuous'):
                logger.info("Pre-fitting ContinuousCEmbedding to old discrete C_embed...")
                self._prefit_continuous_C_embed(old_C_weight)
        else:
            # PaiNN or other architectures — load state dict directly
            migrated = state_dict

        missing, unexpected = self.model.load_state_dict(migrated, strict=False)
        if missing:
            logger.info("New params (randomly initialised, will be fine-tuned): %s",
                        [k for k in missing if not k.startswith('_')])
        if unexpected:
            logger.info("Skipped old params (not in current model): %s", unexpected)

        # EMA: try to load, fall back to copying model
        if 'ema_state_dict' in checkpoint:
            ema_dict = checkpoint['ema_state_dict']
            if needs_egnn_migration:
                ema_migrated = {}
                for key, value in ema_dict.items():
                    if key.startswith('egnn_layers.'):
                        parts = key.split('.', 2)
                        layer_idx = int(parts[1])
                        rest = parts[2]
                        if layer_idx < num_lower:
                            new_key = 'egnn_layers_lower.{}.{}'.format(layer_idx, rest)
                        else:
                            new_key = 'egnn_layers_upper.{}.{}'.format(layer_idx - num_lower, rest)
                        ema_migrated[new_key] = value
                    else:
                        ema_migrated[key] = value
                ema_migrated.pop('C_embed.weight', None)
            else:
                ema_migrated = ema_dict
            self.ema_model.load_state_dict(ema_migrated, strict=False)
            # Copy pre-fitted C_embed_continuous weights to EMA if applicable
            if needs_egnn_migration and hasattr(self.model, 'C_embed_continuous'):
                ema_c_state = self.model.C_embed_continuous.state_dict()
                for k, v in ema_c_state.items():
                    full_key = 'C_embed_continuous.' + k
                    self.ema_model.state_dict()[full_key].copy_(v)
        else:
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()

        # Don't restore old optimizer / LR scheduler state when architecture
        # has changed — start fresh with the fine-tuning LR.
        old_model_cfg = checkpoint.get('config', {}).get('model', {})
        arch_changed = (
            old_model_cfg.get('continuous_C_embed') != self.config.get('model', {}).get('continuous_C_embed')
            or old_model_cfg.get('use_hierarchical') != self.config.get('model', {}).get('use_hierarchical')
            or old_model_cfg.get('architecture') != self.config.get('model', {}).get('architecture')
        )
        if arch_changed:
            logger.info("Architecture changed — using fresh optimizer & LR schedule")
            self.best_val_loss = float('inf')
        else:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as exc:
                logger.warning("Could not restore optimizer state: %s — using fresh", exc)

        if arch_changed:
            self.epoch = 0
            logger.info("Loaded weights from epoch %d — starting fine-tuning from epoch 0",
                        checkpoint['epoch'])
        else:
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info("Loaded checkpoint from epoch %d (val_loss=%.6f)",
                        self.epoch, self.best_val_loss)

    def _prefit_continuous_C_embed(self, old_C_weight: torch.Tensor,
                                   num_steps: int = 2000, lr: float = 1e-3):
        """Pre-train ContinuousCEmbedding to approximate old discrete C_embed.

        Fits the MLP using MSE regression so that
        ``C_embed_continuous(c) ≈ old_C_embed[c]`` for every C value that
        had a non-zero embedding in the old lookup table.  This gives the
        downstream EGNN layers (which were trained with the old embeddings)
        compatible input features from day-1.

        Args:
            old_C_weight: [max_C+1, embed_dim] discrete embedding table.
            num_steps: Number of optimisation steps.
            lr: Learning rate for the pre-fitting.
        """
        import torch.optim as optim

        c_module = self.model.C_embed_continuous
        c_module.train()

        # Find C values with significant embeddings (trained, not random init)
        norms = old_C_weight.norm(dim=-1)  # [max_C+1]
        # Heuristic: most untrained rows have near-zero or random-init norms.
        # Keep C values where the embedding norm is above a small threshold.
        # For the old model, trained C values (20-100) should have large norms.
        median_norm = norms[norms > 0].median()
        valid_mask = norms > median_norm * 0.3
        valid_C = valid_mask.nonzero(as_tuple=True)[0]  # [K]
        targets = old_C_weight[valid_C].to(self.device)  # [K, embed_dim]
        valid_C = valid_C.to(self.device)

        logger.info("  Pre-fitting on %d C values (C=%d..%d), %d steps",
                    len(valid_C), valid_C.min().item(), valid_C.max().item(),
                    num_steps)

        optimizer = optim.Adam(c_module.parameters(), lr=lr)
        best_loss = float('inf')
        best_state = None

        for step in range(num_steps):
            pred = c_module(valid_C)  # [K, embed_dim]
            loss = torch.nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in c_module.state_dict().items()}

            if (step + 1) % 500 == 0:
                # Check cosine similarity for C=60
                with torch.no_grad():
                    c60_pred = c_module(torch.tensor([60], device=self.device))[0]
                    c60_target = old_C_weight[60].to(self.device)
                    cos_sim = torch.nn.functional.cosine_similarity(
                        c60_pred.unsqueeze(0), c60_target.unsqueeze(0)
                    ).item()
                logger.info("  Step %d/%d: MSE=%.6f, cos_sim(C=60)=%.4f",
                            step + 1, num_steps, loss.item(), cos_sim)

        # Restore best state
        if best_state is not None:
            c_module.load_state_dict(best_state)

        c_module.eval()
        # Final check
        with torch.no_grad():
            c60_pred = c_module(torch.tensor([60], device=self.device))[0]
            c60_target = old_C_weight[60].to(self.device)
            cos_sim = torch.nn.functional.cosine_similarity(
                c60_pred.unsqueeze(0), c60_target.unsqueeze(0)
            ).item()
            norm_ratio = c60_pred.norm().item() / c60_target.norm().item()
        logger.info("  Pre-fit done: cos_sim(C=60)=%.4f, norm_ratio=%.3f, best_MSE=%.6f",
                    cos_sim, norm_ratio, best_loss)
    
    def train(self, num_epochs: int):
        """Main training loop with NaN monitoring."""
        start_epoch = self.epoch  # 0 for fresh training, or loaded from checkpoint
        remaining = num_epochs - start_epoch
        logger.info("=" * 60)
        logger.info("Starting training for %d epochs (epoch %d → %d)",
                     remaining, start_epoch + 1, num_epochs)
        logger.info("  Model params: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("  Train samples: %d", len(self.train_loader.dataset))
        logger.info("  LR: %.2e → cosine decay", self.train_config['learning_rate'])
        logger.info("  NaN guard: %s", self.loss_config.get('nan_guard', True))
        logger.info("=" * 60)
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            eval_interval = self.train_config.get(
                'eval_interval', self.loss_config.get('eval_interval', 1)
            )
            if self.epoch % eval_interval == 0:
                val_metrics = self.validate()
                
                lr_now = self.optimizer.param_groups[0]['lr']
                logger.info(
                    "Epoch %d/%d — loss=%.6f noise=%.6f bond=%.6f "
                    "sphere=%.6f | val=%.6f | lr=%.2e | nan_skips=%d",
                    self.epoch, num_epochs,
                    train_metrics['loss'],
                    train_metrics['noise_loss'],
                    train_metrics['bond_loss'],
                    train_metrics['sphere_loss'],
                    val_metrics['val_loss'],
                    lr_now,
                    self.nan_skip_count,
                )
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
                    logger.info("★ New best model! val_loss=%.6f", val_metrics['val_loss'])
            
            # Structural validation
            struct_eval_interval = self.loss_config.get('struct_eval_interval', 0)
            if struct_eval_interval > 0 and self.epoch % struct_eval_interval == 0:
                struct_metrics = self.validate_structures()
                if struct_metrics:
                    parts = []
                    for key, vals in struct_metrics.items():
                        parts.append(
                            f"{key}: bond={vals['bond_mean']:.4f}±{vals['bond_std']:.4f} "
                            f"valid={vals['validity']:.0%} deg3={vals['degree3_frac']:.0%}"
                        )
                    logger.info("Structural eval @ epoch %d: %s", self.epoch, " | ".join(parts))

            # Save periodic checkpoint
            save_interval = self.train_config.get(
                'save_interval', self.loss_config.get('save_interval', 10)
            )
            if self.epoch % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pt')
            
            # Update learning rate
            self.lr_scheduler.step()
        
        logger.info("=" * 60)
        logger.info("Training complete! Best val_loss=%.6f", self.best_val_loss)
        logger.info("Total NaN skips: %d", self.nan_skip_count)
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train Fullerene Diffusion Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
    if topo_ckpt := topo_cfg.get('checkpoint_path'):
        topo_cfg['checkpoint_path'] = topo_ckpt if Path(topo_ckpt).is_absolute() else str((config_dir / topo_ckpt).resolve())
        config['topology_gnn'] = topo_cfg
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set device — prefer CUDA > MPS > CPU
    device_cfg = config.get('device', 'auto')
    if device_cfg == 'auto' or device_cfg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    elif device_cfg == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device(device_cfg if device_cfg in ('cpu',) else 'cpu')
    logger.info("Using device: %s", device)
    
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Optional: train topology GNN before diffusion
    topo_cfg = config.get('topology_gnn', {})
    if topo_cfg.get('enabled', False) and topo_cfg.get('train_before_diffusion', True):
        logger.info("Training topology GNN before diffusion...")
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
