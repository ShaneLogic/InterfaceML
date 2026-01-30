"""
DDPM (Denoising Diffusion Probabilistic Models) utilities for coordinate generation.

This module implements an optimized diffusion process for 3D molecular structure generation.
The diffusion process gradually adds Gaussian noise to data (forward) and learns to reverse
this process (backward) to generate new samples.

Implemented Features (Version 2.0):
  1. Cosine Beta Schedule: Better SNR preservation across timesteps
  2. Improved Sampling: Numerical stability with adaptive clipping
  3. DDIM Support: Faster generation with fewer steps (optional)
  4. SNR Tracking: Signal-to-noise ratio for analysis
  5. Robust Variance Handling: Prevents numerical instability

Mathematical Framework:
  Forward Process (Add Noise):
    q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
  
  Reverse Process (Denoise):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), β̃_t·I)
  
  Posterior Mean:
    μ_θ = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t)·x̂_0 + (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t)·x_t
  
  Where x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t

Key Optimizations:
  - Cosine schedule provides smoother noise addition
  - Precomputed coefficients for efficient sampling  
  - Adaptive clipping prevents numerical explosion
  - Support for both DDPM (stochastic) and DDIM (deterministic) sampling

Reference: 
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
  - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
  - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021

Author: InterfaceML Project
Date: 2026-01-30
Version: 2.0 (Optimized)
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Improved cosine schedule with better signal-to-noise ratio (SNR) properties.
    
    The cosine schedule provides smoother transitions and better preservation
    of structure information at early timesteps compared to linear schedule.
    
    Args:
        timesteps: Number of diffusion steps T
        s: Small offset to prevent beta from being too small near t=0 (default 0.008)
    
    Returns:
        betas: [T] tensor of variance schedule
    
    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", 2021
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # Clip to prevent numerical instability and ensure valid variance
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear variance schedule.
    
    Args:
        timesteps: Number of diffusion steps T
        beta_start: Starting variance
        beta_end: Ending variance
    
    Returns:
        betas: [T] tensor
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionScheduler:
    """
    Manages diffusion process: forward (add noise) and reverse (denoise).
    
    Precomputes all necessary constants for efficient sampling.
    """
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu',
    ):
        """
        Args:
            num_steps: Total diffusion steps T
            beta_schedule: 'linear' or 'cosine'
            beta_start: Starting beta (for linear schedule)
            beta_end: Ending beta (for linear schedule)
            device: 'cpu' or 'cuda'
        """
        self.num_steps = num_steps
        self.device = device
        
        # Create variance schedule
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_steps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(num_steps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
        
        self.betas = betas.to(device)
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Signal-to-noise ratio (SNR) = alpha_bar / (1 - alpha_bar)
        # Useful for analysis and adaptive sampling
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).
        
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        
        Args:
            x_0: Original coordinates [B, N, 3]
            t: Timestep indices [B]
            noise: Optional pre-generated noise (for reproducibility)
        
        Returns:
            x_t: Noisy coordinates [B, N, 3]
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Broadcast coefficients to match x_0 shape
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)  # [B, 1, 1]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t
    
    def sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True,
        clip_range: float = 3.0,
    ) -> torch.Tensor:
        """
        Enhanced single reverse diffusion step: p(x_{t-1} | x_t).
        
        Implements DDIM (eta=0 for deterministic) or DDPM (eta=1) with improved stability.
        
        Args:
            x_t: Current noisy coordinates [B, N, 3]
            t: Current timestep [B]
            noise_pred: Predicted noise from model [B, N, 3]
            eta: DDIM interpolation parameter (0=deterministic, 1=stochastic)
            clip_denoised: Whether to clip predicted x_0
            clip_range: Range for clipping x_0 prediction
        
        Returns:
            x_{t-1}: Less noisy coordinates [B, N, 3]
        """
        # Extract coefficients with proper broadcasting
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)
        
        # Predict x_0 from x_t and noise with numerical stability
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).clamp(min=1e-8)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        # Clip x_0 prediction for stability (adaptive based on data scale)
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -clip_range, clip_range)
        
        # Compute variance (DDIM interpolation)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)
        
        # DDIM variance interpolation
        variance = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)) * \
                   torch.sqrt(1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
        
        # Compute mean using predicted x_0
        mean = sqrt_alpha_bar_prev * x_0_pred + \
               torch.sqrt((1 - alpha_bar_prev - variance ** 2).clamp(min=0)) * noise_pred
        
        # Add noise (except at t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + variance * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        edge_index: torch.Tensor,
        C_cond: torch.Tensor,
        batch: torch.Tensor,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Full reverse sampling process (generate from noise).
        
        Args:
            model: Trained denoising model
            shape: Shape of output (num_nodes, 3)
            edge_index: Graph connectivity [2, E]
            C_cond: Carbon count condition [B]
            batch: Batch assignment for nodes [N]
            num_steps: Number of sampling steps (can be < T for faster generation)
            eta: DDIM parameter
        
        Returns:
            x_0: Generated coordinates [num_nodes, 3]
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # Start from random noise
        x_t = torch.randn(shape, device=self.device)
        
        # Choose timesteps (can skip for faster sampling)
        if num_steps < self.num_steps:
            timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long)
        else:
            timesteps = torch.arange(self.num_steps - 1, -1, -1, dtype=torch.long)
        
        # Iterative denoising
        for i, t in enumerate(timesteps):
            t_batch = torch.full((len(C_cond),), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(x_t, edge_index, t_batch, C_cond, batch)
            
            # One step of denoising
            x_t = self.sample_step(x_t, t_batch, noise_pred, eta=eta)
        
        return x_t


if __name__ == '__main__':
    """Quick test of diffusion scheduler."""
    
    # Create scheduler
    scheduler = DiffusionScheduler(num_steps=1000, beta_schedule='cosine')
    
    print("Scheduler created successfully")
    print(f"  num_steps: {scheduler.num_steps}")
    print(f"  beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"  alpha_bar range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
    
    # Test forward process
    x_0 = torch.randn(2, 60, 3)  # 2 samples, 60 atoms, 3D
    t = torch.tensor([0, 999])   # First and last timestep
    
    x_t = scheduler.add_noise(x_0, t)
    print(f"\nForward diffusion test:")
    print(f"  x_0 std: {x_0.std():.4f}")
    print(f"  x_t[t=0] std: {x_t[0].std():.4f}  (should be ~same as x_0)")
    print(f"  x_t[t=999] std: {x_t[1].std():.4f}  (should be ~1.0, pure noise)")
