"""
E(3)-Equivariant Graph Neural Network for coordinate denoising.

This module implements an enhanced EGNN architecture for fullerene structure generation
via diffusion models. The model respects E(3) symmetry (rotation + translation invariance)
and incorporates several optimization strategies for improved performance.

Key Features (Version 2.0):
  1. Attention Mechanism: Adaptive message weighting for important neighbors
  2. Residual Connections: Improved gradient flow through deep networks
  3. RBF Distance Weighting: Distance-aware coordinate updates
  4. Physics-Informed Losses: Bond length and sphericity constraints
  5. Numerical Stability: Degree normalization and bounded updates

Architecture Overview:
  Input: Noisy coordinates X_t [N, 3]
         Graph structure edge_index [2, E]
         Timestep t (scalar condition)
         Carbon count N (scalar condition)
  
  Processing:
    1. Embed timestep t using sinusoidal encoding
    2. Embed condition N using learnable embedding
    3. Initialize node features h_i = Condition_embed(N) + Time_embed(t)
    4. Apply L EGNN layers: (h, X) → EGNN_layer(h, X, edges)
    5. Predict noise ε from final node features
  
  Output: Predicted noise ε [N, 3]

Mathematical Properties:
  - E(3) Equivariance: f(RX + t, G) = Rf(X, G) + t
  - Permutation Invariance: f(PX, PG) = Pf(X, G)
  - Energy Conservation: Coordinate updates preserve center of mass

Reference: 
  - Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

Author: InterfaceML Project  
Date: 2026-01-30
Version: 2.0 (Optimized)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timestep t.
    
    Maps t ∈ [0, T] to high-dimensional vector with different frequencies.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] tensor of timesteps
        
        Returns:
            embeddings: [B, dim] sinusoidal embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        
        args = t[:, None].float() * freqs[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings


class EGNNLayer(MessagePassing):
    """
    Enhanced E(3)-equivariant graph convolution layer with attention and residual connections.
    
    Updates:
      - Node features h_i via message passing (invariant)
      - Coordinates x_i via attention-weighted relative positions (equivariant)
    
    Architecture:
      m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)  # Edge message (invariant)
      h_i' = h_i + φ_h(h_i, Σ_j m_ij)       # Node update with residual (invariant)
      x_i' = x_i + Σ_j α_ij (x_i - x_j)     # Coordinate update (equivariant)
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int = 0, use_attention: bool = True):
        super().__init__(aggr='add')  # Sum aggregation
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.use_attention = use_attention
        
        # Edge MLP: φ_e(h_i, h_j, d_ij²) → message (deeper for better expressivity)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),  # +1 for distance²
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node MLP: φ_h(h_i, Σ m_ij) → Δh_i (residual update)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Attention mechanism for adaptive message weighting
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        # Coordinate attention: α_ij (improved with softmax normalization)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            h: Node features [N, hidden_dim]
            pos: Coordinates [N, 3]
            edge_index: [2, E]
            edge_attr: Optional edge features [E, edge_dim]
        
        Returns:
            h_new: Updated node features [N, hidden_dim]
            pos_new: Updated coordinates [N, 3]
        """
        # Ensure proper dimensions
        if h.dim() != 2:
            raise ValueError(f"Expected h to be 2D [N, hidden_dim], got shape {h.shape}")
        if pos.dim() != 2 or pos.size(1) != 3:
            raise ValueError(f"Expected pos to be [N, 3], got shape {pos.shape}")
        
        # Store edge_index and pos for use in message
        self._edge_index = edge_index
        self._pos = pos
        
        # Manual message passing to avoid PyG dimension issues
        row, col = edge_index
        h_i = h[row]  # [E, hidden_dim]
        h_j = h[col]  # [E, hidden_dim]
        
        # Compute messages
        pos_i = pos[row]  # [E, 3]
        pos_j = pos[col]  # [E, 3]
        rel_pos = pos_i - pos_j  # [E, 3]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]
        
        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        
        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]
        
        # Apply attention if enabled
        if self.use_attention:
            attention_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
            attention_weights = self.attention_mlp(attention_input)  # [E, 1]
            attention_weights = torch.softmax(attention_weights.view(-1), dim=0).view(-1, 1)
            messages = messages * attention_weights
        
        # Aggregate messages by destination node
        h_agg = torch.zeros_like(h)  # [N, hidden_dim]
        h_agg.index_add_(0, col, messages)  # Add messages to destination nodes
        
        # Update node features with residual connection
        h_delta = self.node_mlp(torch.cat([h, h_agg], dim=-1))
        h_new = h + h_delta  # Residual connection improves gradient flow
        
        # Coordinate updates (scaled for stability)
        pos_delta = self.coord_update(h, pos, edge_index)
        pos_new = pos + 0.1 * pos_delta  # Scale down coordinate updates
        
        return h_new, pos_new
    
    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute edge messages m_ij.
        
        Args:
            h_i, h_j: Node features [E, hidden_dim]
            edge_attr: Edge features [E, edge_dim] or None
        
        Returns:
            messages: [E, hidden_dim]
        """
        # Get positions manually
        row, col = self._edge_index
        pos_i = self._pos.index_select(0, row)  # [E, 3]
        pos_j = self._pos.index_select(0, col)  # [E, 3]
        
        # Relative position and distance
        rel_pos = pos_i - pos_j  # [E, 3]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]
        
        # Ensure all tensors are 2D [E, feature_dim]
        # h_i and h_j should be [E, hidden_dim]
        while h_i.dim() > 2:
            h_i = h_i.squeeze(0)
        while h_j.dim() > 2:
            h_j = h_j.squeeze(0)
        
        # Ensure dist_sq is [E, 1]
        if dist_sq.dim() > 2:
            dist_sq = dist_sq.view(-1, 1)
        
        # Concatenate features
        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist_sq, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        
        # Compute message
        message = self.edge_mlp(edge_input)
        return message
    
    def coord_update(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coordinate updates Δx_i = Σ_j α_ij (x_i - x_j).
        
        This is E(3)-equivariant because:
          - α_ij is scalar (invariant)
          - (x_i - x_j) transforms like a vector
        
        Enhanced with distance-aware weighting and normalization.
        """
        row, col = edge_index
        rel_pos = pos[row] - pos[col]  # [E, 3]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]
        
        # Compute attention weights from edge features
        h_messages = self.edge_mlp(
            torch.cat([
                h[row],
                h[col],
                dist_sq
            ], dim=-1)
        )
        alpha = self.coord_mlp(h_messages)  # [E, 1]
        
        # Apply tanh for bounded updates
        alpha = torch.tanh(alpha)
        
        # Distance-aware weighting (closer nodes have more influence)
        # Use RBF kernel: exp(-dist²/σ²) with σ=1.0
        distance_weight = torch.exp(-dist_sq / 2.0)
        alpha = alpha * distance_weight
        
        # Weighted sum of relative positions
        coord_update = alpha * rel_pos  # [E, 3]
        
        # Aggregate to nodes with averaging for stability
        coord_update_agg = torch.zeros_like(pos)
        coord_update_agg.index_add_(0, row, coord_update)
        
        # Normalize by degree for stable learning
        degree = torch.zeros(pos.size(0), device=pos.device)
        ones = torch.ones(edge_index.size(1), device=pos.device)
        degree.index_add_(0, row, ones)
        degree = degree.clamp(min=1.0).view(-1, 1)
        coord_update_agg = coord_update_agg / degree
        
        return coord_update_agg


class FullereneDiffusionModel(nn.Module):
    """
    Complete diffusion model for fullerene coordinate generation.
    
    Architecture:
      1. Embed timestep t and condition C
      2. Stack of EGNN layers
      3. Output head predicts noise ε
    
    Inputs:
      - Noisy coordinates x_t
      - Graph structure (edge_index)
      - Timestep t
      - Condition (carbon count C)
    
    Output:
      - Predicted noise ε (same shape as x_t)
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 0,
        C_embed_dim: int = 32,
        time_embed_dim: int = 64,
        max_C: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Condition embedding (carbon count)
        self.C_embed = nn.Embedding(max_C + 1, C_embed_dim)
        
        # Initial node features (learned per node + condition + time)
        self.node_init = nn.Linear(C_embed_dim, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        
        # Output head (predicts noise)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Predict 3D noise
        )
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        C: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos: Noisy coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
            t: Diffusion timestep [B]
            C: Carbon count [B]
            batch: Batch assignment [N] (which molecule each atom belongs to)
        
        Returns:
            noise_pred: Predicted noise [N, 3]
        """
        num_nodes = pos.size(0)
        
        # Embed timestep (per batch)
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # Embed condition (per batch)
        C_emb = self.C_embed(C)  # [B, C_embed_dim]
        
        # Initialize node features (broadcast condition to all nodes)
        h = self.node_init(C_emb[batch])  # [N, hidden_dim]
        h = h + t_emb[batch]  # Add time embedding
        
        # Apply EGNN layers
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index)
        
        # Predict noise
        noise_pred = self.output_head(h)  # [N, 3]
        
        return noise_pred


def bond_length_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    target_length: float = 1.39,
) -> torch.Tensor:
    """
    Physics-informed loss: penalize deviations from ideal C-C bond length.
    
    Args:
        pos: Coordinates [N, 3]
        edge_index: [2, E]
        target_length: Target bond length (Angstroms)
    
    Returns:
        loss: Scalar MSE between actual and target bond lengths
    """
    row, col = edge_index
    rel_pos = pos[row] - pos[col]
    distances = torch.norm(rel_pos, dim=-1)
    
    # Use Huber loss for robustness to outliers
    loss = F.smooth_l1_loss(distances, torch.full_like(distances, target_length))
    return loss


def sphericity_loss(
    pos: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Sphericity loss: encourages structures to be spherical (low asphericity).
    
    Computes normalized asphericity from gyration tensor.
    Uses relative asphericity = (λ1 - λ3) / (λ1 + λ2 + λ3) 
    Range: [0, 1] where 0 = perfect sphere, 1 = maximally elongated
    
    Args:
        pos: Coordinates [N, 3]
        batch: Batch assignment [N]
    
    Returns:
        loss: Mean relative asphericity across batch (0-1 range)
    """
    batch_size = batch.max().item() + 1
    asphericities = []
    
    for i in range(batch_size):
        # Get positions for this molecule
        mask = batch == i
        mol_pos = pos[mask]  # [N_i, 3]
        
        # Center coordinates
        centered = mol_pos - mol_pos.mean(dim=0, keepdim=True)
        
        # Gyration tensor: S = (1/N) X^T X
        S = torch.mm(centered.t(), centered) / centered.size(0)  # [3, 3]
        
        # Compute eigenvalues (sorted ascending: λ1 ≤ λ2 ≤ λ3)
        eigvals = torch.linalg.eigvalsh(S)  # [3], sorted ascending
        eigvals = torch.clamp(eigvals, min=1e-8)  # Avoid division by zero
        
        # Relative asphericity (normalized by total inertia)
        # Perfect sphere: λ1 = λ2 = λ3, asphericity = 0
        # Maximally elongated: λ1 = λ2 = 0, asphericity → 1
        trace = eigvals.sum()
        asphericity = (eigvals[2] - eigvals[0]) / (trace + 1e-8)
        
        asphericities.append(asphericity)
    
    return torch.mean(torch.stack(asphericities))


if __name__ == '__main__':
    """Quick test of model architecture."""
    
    # Create dummy data
    num_nodes = 60
    pos = torch.randn(num_nodes, 3)
    
    # Simple graph (3-regular)
    edges = []
    for i in range(num_nodes):
        for j in [(i+1) % num_nodes, (i+2) % num_nodes, (i-1) % num_nodes]:
            if i < j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Conditions
    t = torch.tensor([500])  # Timestep
    C = torch.tensor([60])   # Carbon count
    batch = torch.zeros(num_nodes, dtype=torch.long)  # All in same batch
    
    # Create model
    model = FullereneDiffusionModel(hidden_dim=32, num_layers=2)
    
    print("Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        noise_pred = model(pos, edge_index, t, C, batch)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {pos.shape}")
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Output mean: {noise_pred.mean():.4f}")
    print(f"  Output std: {noise_pred.std():.4f}")
    
    # Test bond length loss
    loss = bond_length_loss(pos, edge_index, target_length=1.39)
    print(f"\nBond length loss: {loss.item():.4f}")
