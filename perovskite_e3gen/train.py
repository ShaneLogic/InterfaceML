"""Training script for Perovskite E3Gen model.

Trains a Periodic PaiNN flow matching model to jointly generate:
  - Fractional coordinates (torus flow)
  - Lattice parameters (Euclidean flow)
  - Atom types (continuous relaxation)

Usage:
    python train.py --config config.yaml --data_dir ../dataset/perovskite/mp_perovskite_cifs

Author: InterfaceML Project
Date: 2026-02
"""

import argparse
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

from dataset import get_dataloaders, PerovskiteDataset
from model import PerovskitePaiNNModel
from flow_matching import CrystalFlowMatcher
from physics_loss import combined_physics_loss
from units import LatticeScaler, lattice_params_to_matrix

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training loop, validation, and checkpointing."""

    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device

        # Create dataloaders
        logger.info("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            config, num_workers=config.get("num_workers", 0)
        )

        # Store lattice scaler from dataset
        # (Accessed via the underlying dataset's scaler)
        self.lattice_scaler = None
        if hasattr(self.train_loader.dataset, 'lattice_scaler'):
            self.lattice_scaler = self.train_loader.dataset.lattice_scaler
        elif isinstance(self.train_loader.dataset, list) and len(self.train_loader.dataset) > 0:
            # DataLoader wraps a list; scaler stored at class level
            pass

        # Create model
        model_cfg = config["model"]
        self.model = PerovskitePaiNNModel(
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 6),
            num_rbf=model_cfg.get("num_rbf", 20),
            cutoff=model_cfg.get("cutoff", 5.0),
            time_embed_dim=model_cfg.get("time_embed_dim", 128),
            num_elements=model_cfg.get("num_elements", 100),
            element_embed_dim=model_cfg.get("element_embed_dim", 64),
            composition_embed_dim=model_cfg.get("composition_embed_dim", 128),
        ).to(device)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model parameters: %s", f"{num_params:,}")

        # Flow matching scheduler
        diff_cfg = config.get("diffusion", {})
        self.flow_matcher = CrystalFlowMatcher(
            sigma_min=diff_cfg.get("sigma_min", 1e-4),
            coord_noise_type=diff_cfg.get("coord_noise_type", "wrapped_normal"),
            lattice_noise_scale=diff_cfg.get("lattice_noise_scale", 1.0),
        )

        # Optimizer
        train_cfg = config.get("training", {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 5e-4),
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            eps=train_cfg.get("eps", 1e-8),
        )

        # Cosine annealing with warmup
        self.epochs = train_cfg.get("epochs", 300)
        warmup_epochs = train_cfg.get("warmup_epochs", 10)
        self.warmup_steps = warmup_epochs * len(self.train_loader)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs * len(self.train_loader)
        )

        # EMA
        self.ema_decay = train_cfg.get("ema_decay", 0.9999)
        self.ema_model = deepcopy(self.model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        # Gradient clipping
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.nan_guard = train_cfg.get("nan_guard", True)

        # Loss weights
        loss_cfg = config.get("loss", {})
        self.lambda_coord = loss_cfg.get("lambda_coord", 1.0)
        self.lambda_lattice = loss_cfg.get("lambda_lattice", 1.0)
        self.lambda_type = loss_cfg.get("lambda_type", 0.5)

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def _update_ema(self):
        """Update EMA model weights."""
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)

    def _warmup_lr(self):
        """Linear warmup for learning rate."""
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, self.global_step / max(1, self.warmup_steps))
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config["training"]["learning_rate"] * lr_scale

    def train_step(self, batch) -> dict:
        """Single training step.

        Returns:
            dict of loss components.
        """
        self.model.train()

        frac_coords = batch.frac_coords.to(self.device)
        atom_types = batch.atom_types.to(self.device)
        lattice_params_norm = batch.lattice_params_norm.to(self.device)
        lattice_matrix = batch.lattice_matrix.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        edge_shift = batch.edge_shift.to(self.device)
        batch_idx = batch.batch.to(self.device)

        batch_size = batch_idx.max().item() + 1
        num_elements = self.model.num_elements

        # Reshape lattice params: [total_6] -> [B, 6]
        lattice_params_batch = lattice_params_norm.view(batch_size, 6)
        lattice_matrix_batch = lattice_matrix.view(batch_size, 3, 3)

        # Sample time
        t = self.flow_matcher.sample_t(batch_size, device=self.device)
        t_per_node = t[batch_idx]

        # Add noise to coordinates (torus flow)
        frac_t, target_v_coord = self.flow_matcher.add_coord_noise(frac_coords, t_per_node)

        # Add noise to lattice (Euclidean flow)
        lattice_t, target_v_lattice = self.flow_matcher.add_lattice_noise(lattice_params_batch, t)

        # Build noisy lattice matrix for forward pass
        lattice_t_lengths = lattice_t[:, :3]
        lattice_t_angles = lattice_t[:, 3:]

        # For the model, we need a valid lattice matrix from noisy params
        # During early training, noisy params may be unreasonable, so we clamp
        lattice_t_matrix = lattice_matrix_batch  # Use ground truth lattice for graph structure

        # Forward pass
        coord_vel, lattice_vel, type_logits = self.model(
            frac_t, atom_types, lattice_t_matrix, t,
            edge_index, edge_shift, batch_idx,
        )

        # Losses
        # 1. Coordinate velocity MSE
        coord_loss = F.mse_loss(coord_vel, target_v_coord)

        # 2. Lattice velocity MSE
        lattice_loss = F.mse_loss(lattice_vel, target_v_lattice)

        # 3. Atom type cross-entropy
        type_loss = F.cross_entropy(type_logits, atom_types)

        # Combined primary loss
        total_loss = (
            self.lambda_coord * coord_loss
            + self.lambda_lattice * lattice_loss
            + self.lambda_type * type_loss
        )

        # Physics losses (time-conditioned)
        physics_loss, physics_dict = combined_physics_loss(
            frac_t, atom_types, batch.lattice_params.to(self.device).view(batch_size, 6),
            lattice_matrix_batch, batch_idx, t, self.config,
        )
        total_loss = total_loss + physics_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # NaN guard
        if self.nan_guard:
            has_nan = False
            for p in self.model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                logger.warning("NaN gradient detected, skipping step %d", self.global_step)
                self.optimizer.zero_grad()
                return {"total": float("nan"), "coord": float("nan"),
                        "lattice": float("nan"), "type": float("nan")}

        # Gradient clipping
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self._warmup_lr()
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        self._update_ema()
        self.global_step += 1

        return {
            "total": total_loss.item(),
            "coord": coord_loss.item(),
            "lattice": lattice_loss.item(),
            "type": type_loss.item(),
            **physics_dict,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation epoch."""
        self.ema_model.eval()
        total_losses = {"coord": 0.0, "lattice": 0.0, "type": 0.0, "total": 0.0}
        num_batches = 0

        for batch in self.val_loader:
            frac_coords = batch.frac_coords.to(self.device)
            atom_types = batch.atom_types.to(self.device)
            lattice_params_norm = batch.lattice_params_norm.to(self.device)
            lattice_matrix = batch.lattice_matrix.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_shift = batch.edge_shift.to(self.device)
            batch_idx = batch.batch.to(self.device)
            batch_size = batch_idx.max().item() + 1

            lattice_params_batch = lattice_params_norm.view(batch_size, 6)
            lattice_matrix_batch = lattice_matrix.view(batch_size, 3, 3)

            t = self.flow_matcher.sample_t(batch_size, device=self.device)
            t_per_node = t[batch_idx]

            frac_t, target_v_coord = self.flow_matcher.add_coord_noise(frac_coords, t_per_node)
            _, target_v_lattice = self.flow_matcher.add_lattice_noise(lattice_params_batch, t)

            coord_vel, lattice_vel, type_logits = self.ema_model(
                frac_t, atom_types, lattice_matrix_batch, t,
                edge_index, edge_shift, batch_idx,
            )

            coord_loss = F.mse_loss(coord_vel, target_v_coord)
            lattice_loss = F.mse_loss(lattice_vel, target_v_lattice)
            type_loss = F.cross_entropy(type_logits, atom_types)

            total = (self.lambda_coord * coord_loss
                     + self.lambda_lattice * lattice_loss
                     + self.lambda_type * type_loss)

            total_losses["coord"] += coord_loss.item()
            total_losses["lattice"] += lattice_loss.item()
            total_losses["type"] += type_loss.item()
            total_losses["total"] += total.item()
            num_batches += 1

        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches

        return total_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        if self.lattice_scaler is not None:
            state["lattice_scaler"] = self.lattice_scaler.state_dict()

        # Save periodic checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info("Saved best model: %s (val_loss=%.4f)", best_path, val_loss)

    def load_checkpoint(self, path: str):
        """Load checkpoint and resume training."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.ema_model.load_state_dict(state["ema_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.global_step = state.get("global_step", 0)
        if "lattice_scaler" in state:
            self.lattice_scaler = LatticeScaler.from_state_dict(state["lattice_scaler"])
        logger.info("Resumed from epoch %d (step %d)", state["epoch"], self.global_step)
        return state["epoch"]

    def train(self, start_epoch: int = 0):
        """Full training loop."""
        train_cfg = self.config.get("training", {})
        eval_interval = train_cfg.get("eval_interval", 25)
        save_interval = train_cfg.get("save_interval", 50)
        log_interval = train_cfg.get("log_interval", 10)

        for epoch in range(start_epoch, self.epochs):
            epoch_losses = {}
            epoch_start = time.time()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch in pbar:
                losses = self.train_step(batch)

                # Accumulate
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v)

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{losses['total']:.4f}",
                    "coord": f"{losses['coord']:.4f}",
                    "lat": f"{losses['lattice']:.4f}",
                    "type": f"{losses['type']:.4f}",
                })

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items() if v}
            lr = self.optimizer.param_groups[0]["lr"]

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d [%.1fs] lr=%.2e | loss=%.4f coord=%.4f lat=%.4f type=%.4f",
                    epoch + 1, self.epochs, epoch_time, lr,
                    avg_losses.get("total", 0),
                    avg_losses.get("coord", 0),
                    avg_losses.get("lattice", 0),
                    avg_losses.get("type", 0),
                )

            # Validation
            if (epoch + 1) % eval_interval == 0:
                val_losses = self.validate()
                logger.info(
                    "  VAL | loss=%.4f coord=%.4f lat=%.4f type=%.4f",
                    val_losses["total"], val_losses["coord"],
                    val_losses["lattice"], val_losses["type"],
                )

                is_best = val_losses["total"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses["total"]

                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch + 1, val_losses["total"], is_best=is_best)

        # Final save
        val_losses = self.validate()
        self.save_checkpoint(self.epochs, val_losses["total"],
                             is_best=val_losses["total"] < self.best_val_loss)
        logger.info("Training complete! Best val loss: %.4f", self.best_val_loss)


def get_device(device_str: str = "auto") -> str:
    """Auto-detect best available device."""
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train Perovskite E3Gen model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Override MP CIF directory")
    parser.add_argument("--hoip_dir", type=str, default=None, help="Override HOIP CIF directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device: auto/cuda/mps/cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--cutoff", type=float, default=None, help="Radius cutoff (Angstrom)")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of PaiNN layers")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    base = config_path.parent
    for key in ["mp_cif_dir", "hoip_cif_dir"]:
        if key in config["data"] and config["data"][key]:
            p = Path(config["data"][key])
            if not p.is_absolute():
                config["data"][key] = str(base / p)

    # Apply CLI overrides
    if args.data_dir:
        config["data"]["mp_cif_dir"] = args.data_dir
    if args.hoip_dir:
        config["data"]["hoip_cif_dir"] = args.hoip_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.cutoff:
        config["data"]["cutoff"] = args.cutoff
        config["model"]["cutoff"] = args.cutoff
    if args.hidden_dim:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.num_layers:
        config["model"]["num_layers"] = args.num_layers

    # Device
    device_str = args.device or config.get("device", "auto")
    device = get_device(device_str)
    logger.info("Using device: %s", device)

    # Seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Train
    trainer = Trainer(config, device)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(args.checkpoint)

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
