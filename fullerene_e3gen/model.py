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

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

logger = logging.getLogger(__name__)


def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Per-node softmax over edges: normalize logits for each destination node.

    This is the correct way to apply attention in GNNs — each node's incoming
    edge weights sum to 1.  A naive global softmax would wash out all weights
    in large graphs.

    Parameters
    ----------
    logits : [E] raw attention scores.
    index  : [E] destination node indices.
    num_nodes : total number of nodes.

    Returns
    -------
    weights : [E] attention weights, summing to 1 per destination node.
    """
    # Numerical stability: subtract per-node max
    max_vals = torch.full((num_nodes,), -1e9, device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(0, index, logits, reduce="amax", include_self=False)
    logits = logits - max_vals[index]

    exp_logits = torch.exp(logits)
    sum_exp = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    sum_exp.index_add_(0, index, exp_logits)
    return exp_logits / (sum_exp[index] + 1e-12)


class ContinuousCEmbedding(nn.Module):
    """Continuous condition embedding for carbon count C.

    **Plan B improvement**: replaces ``nn.Embedding(max_C, dim)`` with a
    continuous Fourier feature mapping followed by an MLP.  This allows the
    model to generalise to C values **never seen during training** because
    the representation varies smoothly with C rather than being a discrete
    lookup.

    The mapping is:
        C → [sin(ω₁·C/C_ref), cos(ω₁·C/C_ref), …, sin(ωK·C/C_ref), cos(ωK·C/C_ref)]
        → MLP → embedding  ∈ ℝ^{embed_dim}

    where C_ref is a reference scale (default 100) that keeps the Fourier
    arguments in a numerically pleasant range.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        fourier_features: int = 16,
        C_ref: float = 100.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.C_ref = C_ref

        half = fourier_features
        # Logarithmically spaced frequencies (like NeRF positional encoding)
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half)
        )
        self.register_buffer('freqs', freqs)  # [half]

        input_dim = 2 * half + 1  # sin + cos + raw C/C_ref
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Embed carbon count.

        Args:
            C: [B] integer tensor of carbon counts.

        Returns:
            [B, embed_dim] continuous embedding.
        """
        c = C.float() / self.C_ref  # [B]
        args = c[:, None] * self.freqs[None, :]  # [B, half]
        fourier = torch.cat([torch.sin(args), torch.cos(args), c[:, None]], dim=-1)  # [B, 2*half+1]
        return self.mlp(fourier)  # [B, embed_dim]


class GraphCoarsenLayer(nn.Module):
    """Coarsen a graph by merging pairs of bonded nodes.

    **Plan D improvement**: for large graphs (C > ~100) the 6-layer EGNN
    has a limited receptive field (≤ 6-hop).  Hierarchical message passing
    first coarsens the graph, runs EGNN layers on the smaller graph (where
    each node represents a cluster), then maps information back.

    Coarsening strategy:
    - Greedy edge-contraction: traverse edges in random order, merge
      unmatched endpoints.  This roughly halves the node count.
    - Cluster features are computed by a learned aggregation (mean + linear).
    - Cluster positions are the mean of member positions.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.merge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coarsen the graph.

        Returns:
            h_coarse   [M, hidden_dim]
            pos_coarse [M, 3]
            edge_index_coarse [2, E']
            assignment [N] mapping from fine node → coarse node
        """
        num_nodes = h.size(0)
        device = h.device

        # Greedy matching
        assignment = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        cluster_id = 0
        row, col = edge_index

        # Shuffle edges deterministically per forward pass (use positions hash)
        perm = torch.randperm(row.size(0), device=device)
        row_p, col_p = row[perm], col[perm]

        for i in range(row_p.size(0)):
            u, v = row_p[i].item(), col_p[i].item()
            if assignment[u] == -1 and assignment[v] == -1:
                assignment[u] = cluster_id
                assignment[v] = cluster_id
                cluster_id += 1

        # Assign remaining unmatched nodes to their own cluster
        for i in range(num_nodes):
            if assignment[i] == -1:
                assignment[i] = cluster_id
                cluster_id += 1

        num_clusters = cluster_id

        # Aggregate features and positions
        h_coarse = torch.zeros(num_clusters, self.hidden_dim, device=device)
        pos_coarse = torch.zeros(num_clusters, 3, device=device)
        counts = torch.zeros(num_clusters, 1, device=device)

        h_coarse.index_add_(0, assignment, h)
        pos_coarse.index_add_(0, assignment, pos)
        counts.index_add_(0, assignment, torch.ones(num_nodes, 1, device=device))
        counts = counts.clamp(min=1)

        h_coarse = h_coarse / counts
        pos_coarse = pos_coarse / counts

        h_coarse = self.merge_mlp(h_coarse)

        # Build coarse edge index (merge fine edges, remove self-loops & dups)
        coarse_src = assignment[row]
        coarse_dst = assignment[col]
        mask = coarse_src != coarse_dst
        coarse_edges = torch.stack([coarse_src[mask], coarse_dst[mask]], dim=0)
        # Remove duplicates via unique edge hashing
        if coarse_edges.size(1) > 0:
            edge_hash = coarse_edges[0] * num_clusters + coarse_edges[1]
            unique_hash = torch.unique(edge_hash)
            coarse_src_u = unique_hash // num_clusters
            coarse_dst_u = unique_hash % num_clusters
            edge_index_coarse = torch.stack([coarse_src_u, coarse_dst_u], dim=0)
        else:
            edge_index_coarse = coarse_edges

        return h_coarse, pos_coarse, edge_index_coarse, assignment


class GraphUncoarsenLayer(nn.Module):
    """Map information from coarse graph back to fine graph.

    Each fine node reads its cluster's coarse representation and combines
    it with its own (stored) fine representation via a gating mechanism.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h_fine: torch.Tensor,
        pos_fine: torch.Tensor,
        h_coarse: torch.Tensor,
        pos_coarse: torch.Tensor,
        assignment: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uncoarsen: map coarse info back to fine nodes.

        Returns:
            h_updated [N, hidden_dim]
            pos_updated [N, 3]
        """
        h_upsampled = h_coarse[assignment]        # [N, hidden_dim]
        pos_upsampled = pos_coarse[assignment]  # [N, 3]

        # Gated combination
        gate = self.gate(torch.cat([h_fine, h_upsampled], dim=-1))  # [N, hidden_dim]
        h_out = h_fine + gate * self.proj(h_upsampled)

        # Position: weighted blend of fine and coarse-upsampled positions
        # Fine positions are more geometrically precise; coarse positions
        # carry global context.  Use a small blend factor.
        pos_delta = 0.1 * (pos_upsampled - pos_fine)
        pos_out = pos_fine + pos_delta

        return h_out, pos_out


class GlobalAttentionPool(nn.Module):
    """O(N) pool-broadcast-gate module on invariant features h.

    Provides each node with a global graph summary via a gated residual
    connection.  Only operates on invariant features (not coordinates), so
    SE(3) equivariance of subsequent EGNN layers is preserved.

    Architecture:
        1. h_global = mean_pool(h, batch)          [B, D]
        2. h_context = MLP(h_global)               [B, D]
        3. h_context_i = h_context[batch]           [N, D]
        4. gate = sigmoid(W_gate[h; h_context_i])   [N, D]
        5. return h + gate * proj(h_context_i)       gated residual
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [N, D].
            batch: Batch assignment [N].

        Returns:
            h_out: Updated node features [N, D] with global context.
        """
        h_global = global_mean_pool(h, batch)          # [B, D]
        h_context = self.context_mlp(h_global)         # [B, D]
        h_context_i = h_context[batch]                 # [N, D]
        gate = self.gate(torch.cat([h, h_context_i], dim=-1))  # [N, D]
        return h + gate * self.proj(h_context_i)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timestep t.
    
    Maps t ∈ [0, T] to high-dimensional vector with different frequencies.
    Frequencies are registered as a buffer to avoid repeated CPU→GPU transfers.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] tensor of timesteps
        
        Returns:
            embeddings: [B, dim] sinusoidal embeddings
        """
        args = t[:, None].float() * self.freqs[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings


class EGNNLayer(MessagePassing):
    """
    Enhanced E(3)-equivariant graph convolution layer (v2).

    Key improvements over v1:
      - Pre-LayerNorm on node features for gradient stability
      - Bounded coordinate updates via tanh + distance weighting
      - RBF distance encoding (replaces raw d²)
      - NaN-safe aggregation (clamps intermediate values)

    Architecture:
      m_ij = φ_e(h_i, h_j, RBF(d_ij))        # Edge message (invariant)
      h_i' = h_i + φ_h(h_i, Σ_j m_ij)         # Node update with residual
      x_i' = x_i + 0.1 * Σ_j α_ij (x_i - x_j) # Coordinate update (equivariant)
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0, use_attention: bool = True, num_rbf: int = 16,
                 use_distance_weight: bool = True, use_degree_norm: bool = True):
        super().__init__(aggr='add')

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.use_attention = use_attention
        self.num_rbf = num_rbf
        self.use_distance_weight = use_distance_weight
        self.use_degree_norm = use_degree_norm

        # Pre-LayerNorm for stable gradient flow
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Distance features: RBF encoding (num_rbf > 1) or raw d² (num_rbf == 1)
        if self.num_rbf > 1:
            rbf_centers = torch.linspace(0.0, 5.0, self.num_rbf)
            self.register_buffer('rbf_centers', rbf_centers)
            self.rbf_width = 0.5
        dist_feat_dim = self.num_rbf  # 1 for raw d², num_rbf for RBF

        # Edge MLP: φ_e(h_i, h_j, dist_feat) → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + dist_feat_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node MLP: φ_h(h_i, Σ m_ij) → Δh_i
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention for adaptive message weighting
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + dist_feat_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Coordinate attention: α_ij
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
        if h.dim() != 2:
            raise ValueError(f"Expected h to be 2D [N, hidden_dim], got shape {h.shape}")
        if pos.dim() != 2 or pos.size(1) != 3:
            raise ValueError(f"Expected pos to be [N, 3], got shape {pos.shape}")

        # Pre-LayerNorm for gradient stability
        h_normed = self.layer_norm(h)

        self._edge_index = edge_index
        self._pos = pos

        # --- Manual message passing ---
        row, col = edge_index
        h_i = h_normed[row]  # [E, hidden_dim]
        h_j = h_normed[col]  # [E, hidden_dim]

        pos_i = pos[row]  # [E, 3]
        pos_j = pos[col]  # [E, 3]
        rel_pos = pos_i - pos_j  # [E, 3]
        dist = rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]

        # Distance features: RBF encoding or raw d²
        if self.num_rbf > 1:
            dist_feat = torch.exp(-((dist - self.rbf_centers) ** 2) / (2 * self.rbf_width ** 2))  # [E, num_rbf]
        else:
            dist_feat = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1] raw d²

        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist_feat, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist_feat], dim=-1)

        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Attention (per-destination softmax)
        if self.use_attention:
            attention_input = torch.cat([h_i, h_j, dist_feat], dim=-1)
            attention_logits = self.attention_mlp(attention_input).squeeze(-1)
            attention_weights = _scatter_softmax(attention_logits, col, num_nodes=h.size(0))
            messages = messages * attention_weights.unsqueeze(-1)

        # Aggregate messages by destination node
        h_agg = torch.zeros_like(h)
        h_agg.index_add_(0, col, messages)

        # Node update with residual
        h_delta = self.node_mlp(torch.cat([h_normed, h_agg], dim=-1))
        h_new = h + h_delta  # Residual on UN-normed h

        # Coordinate updates (full-scale for equivariant noise prediction)
        pos_delta = self.coord_update(h_normed, pos, edge_index, messages=messages)
        pos_new = pos + pos_delta

        # NaN guard: if any output is NaN, fall back to input
        if torch.isnan(h_new).any() or torch.isnan(pos_new).any():
            logger.warning("NaN detected in EGNNLayer output, falling back to input")
            return h, pos

        return h_new, pos_new
    
    def coord_update(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        messages: Optional[torch.Tensor] = None,
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
        
        # Reuse messages if provided; otherwise recompute
        if messages is None:
            dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]
            messages = self.edge_mlp(
                torch.cat([h[row], h[col], dist_sq], dim=-1)
            )
        
        alpha = self.coord_mlp(messages)  # [E, 1]
        
        # Apply tanh for bounded updates
        alpha = torch.tanh(alpha)

        # Distance-aware weighting (closer nodes have more influence)
        # Use RBF kernel: exp(-dist²/σ²) with σ=1.0
        if self.use_distance_weight:
            dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
            distance_weight = torch.exp(-dist_sq / 2.0)
            alpha = alpha * distance_weight

        # Weighted sum of relative positions
        coord_update = alpha * rel_pos  # [E, 3]

        # Aggregate to nodes with averaging for stability
        coord_update_agg = torch.zeros_like(pos)
        coord_update_agg.index_add_(0, row, coord_update)

        # Normalize by degree for stable learning
        if self.use_degree_norm:
            degree = torch.zeros(pos.size(0), device=pos.device)
            ones = torch.ones(edge_index.size(1), device=pos.device)
            degree.index_add_(0, row, ones)
            degree = degree.clamp(min=1.0).view(-1, 1)
            coord_update_agg = coord_update_agg / degree

        return coord_update_agg


class FullereneDiffusionModel(nn.Module):
    """
    Complete diffusion model for fullerene coordinate generation (v3).

    Architecture:
      1. Embed timestep t + condition C with LayerNorm
      2. Stack of EGNN layers (each with pre-LayerNorm + RBF distances)
      3. Optional hierarchical coarsen → EGNN → uncoarsen for large graphs
      4. Final LayerNorm + output head → predicted noise ε

    v3 improvements (Plan B + Plan D):
      - **Plan B**: ContinuousCEmbedding replaces nn.Embedding for carbon
        count conditioning.  Uses Fourier features + MLP so the model
        generalises smoothly to C values never seen during training.
        Backward compatible: old checkpoints with discrete C_embed are
        loaded via ``strict=False`` (the continuous embedding is randomly
        initialised and fine-tuned).
      - **Plan D**: Hierarchical message passing via GraphCoarsenLayer /
        GraphUncoarsenLayer.  For graphs with N > ``hierarchical_threshold``
        nodes, 2 extra EGNN layers run on a coarsened graph (~N/2 nodes)
        between the main EGNN blocks.  This doubles the effective receptive
        field without doubling compute.
      - NaN-safe forward pass (returns zeros if NaN detected)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        edge_dim: int = 0,
        C_embed_dim: int = 64,
        time_embed_dim: int = 128,
        max_C: int = 720,
        num_rbf: int = 16,
        # --- Plan B ---
        continuous_C_embed: bool = True,
        C_fourier_features: int = 16,
        # --- Plan D ---
        use_hierarchical: bool = True,
        hierarchical_threshold: int = 80,
        hierarchical_layers: int = 2,
        # --- Global Attention ---
        use_global_attention: bool = False,
        global_attention_heads: int = 4,
        # --- Coordinate update flags ---
        use_distance_weight: bool = True,
        use_degree_norm: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.continuous_C_embed_flag = continuous_C_embed
        self.use_hierarchical = use_hierarchical
        self.hierarchical_threshold = hierarchical_threshold
        self.use_global_attention = use_global_attention

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Plan B: Condition embedding (carbon count) ---
        if continuous_C_embed:
            self.C_embed_continuous = ContinuousCEmbedding(
                embed_dim=C_embed_dim,
                fourier_features=C_fourier_features,
                C_ref=100.0,
            )
        else:
            # Legacy discrete embedding (for loading old checkpoints)
            self.C_embed = nn.Embedding(max_C + 1, C_embed_dim)

        # Initial node features
        self.node_init = nn.Linear(C_embed_dim, hidden_dim)

        # Post-init LayerNorm for stable initial features
        self.init_norm = nn.LayerNorm(hidden_dim)

        # Split EGNN layers into two halves for hierarchical insertion
        mid = num_layers // 2
        self.egnn_layers_lower = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim, num_rbf=num_rbf,
                             use_distance_weight=use_distance_weight, use_degree_norm=use_degree_norm) for _ in range(mid)
        ])
        upper_count = num_layers - mid
        if use_global_attention and upper_count >= 2:
            # Split upper into pre (1 layer) and post (rest), with global
            # attention between them so enriched h propagates to coordinates.
            self.egnn_layers_upper_pre = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, num_rbf=num_rbf,
                             use_distance_weight=use_distance_weight, use_degree_norm=use_degree_norm)
            ])
            self.egnn_layers_upper_post = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, num_rbf=num_rbf,
                             use_distance_weight=use_distance_weight, use_degree_norm=use_degree_norm) for _ in range(upper_count - 1)
            ])
            self.global_attention = GlobalAttentionPool(hidden_dim, num_heads=global_attention_heads)
            # Backward-compat: alias so egnn_layers_upper still works
            self.egnn_layers_upper = nn.ModuleList(
                list(self.egnn_layers_upper_pre) + list(self.egnn_layers_upper_post)
            )
        else:
            self.egnn_layers_upper = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, num_rbf=num_rbf,
                             use_distance_weight=use_distance_weight, use_degree_norm=use_degree_norm) for _ in range(upper_count)
            ])

        # --- Plan D: Hierarchical message passing ---
        if use_hierarchical:
            self.coarsen = GraphCoarsenLayer(hidden_dim)
            self.uncoarsen = GraphUncoarsenLayer(hidden_dim)
            self.egnn_layers_coarse = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, num_rbf=num_rbf,
                             use_distance_weight=use_distance_weight, use_degree_norm=use_degree_norm)
                for _ in range(hierarchical_layers)
            ])

        # Final LayerNorm before output
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output head (predicts noise)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Predict 3D noise
        )

    # Backward-compatible property so old code referencing model.egnn_layers
    # still works (e.g. for parameter counting).
    @property
    def egnn_layers(self):
        """Return all EGNN layers (lower + upper + coarse) for inspection."""
        layers = list(self.egnn_layers_lower) + list(self.egnn_layers_upper)
        if self.use_hierarchical:
            layers += list(self.egnn_layers_coarse)
        return nn.ModuleList(layers)
    
    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        C: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Equivariant noise prediction via coordinate displacement.

        The EGNN layers update both node features (invariant) and coordinates
        (equivariant).  The predicted noise is the total coordinate displacement
        across all layers, which is SE(3)-equivariant by construction.

        v3: Hierarchical message passing is inserted between the lower and
        upper EGNN blocks when the graph is large (N > hierarchical_threshold).

        Args:
            pos: Noisy coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
            t: Diffusion timestep [B]
            C: Carbon count [B]
            batch: Batch assignment [N]

        Returns:
            noise_pred: Predicted noise [N, 3] (equivariant)
        """
        num_nodes = pos.size(0)
        pos_input = pos.clone()  # Save input for equivariant displacement

        # Embed timestep (per batch)
        t_emb = self.time_embed(t)      # [B, hidden_dim]

        # --- Plan B: Embed condition (per batch) ---
        if self.continuous_C_embed_flag:
            C_emb = self.C_embed_continuous(C)   # [B, C_embed_dim]
        else:
            C_emb = self.C_embed(C)              # [B, C_embed_dim]

        # Initialize node features with LayerNorm
        h = self.node_init(C_emb[batch])  # [N, hidden_dim]
        h = h + t_emb[batch]              # Add time embedding
        h = self.init_norm(h)             # Normalize initial features

        # --- Lower EGNN layers ---
        for layer in self.egnn_layers_lower:
            h, pos = layer(h, pos, edge_index)

        # --- Plan D: Hierarchical message passing (for large graphs) ---
        if self.use_hierarchical and num_nodes > self.hierarchical_threshold:
            h_fine_saved = h.clone()
            pos_fine_saved = pos.clone()

            h_c, pos_c, edge_c, assignment = self.coarsen(h, pos, edge_index)

            for layer in self.egnn_layers_coarse:
                h_c, pos_c = layer(h_c, pos_c, edge_c)

            h, pos = self.uncoarsen(h_fine_saved, pos_fine_saved, h_c, pos_c, assignment)

        # --- Upper EGNN layers (with optional global attention) ---
        if self.use_global_attention and hasattr(self, 'egnn_layers_upper_pre'):
            for layer in self.egnn_layers_upper_pre:
                h, pos = layer(h, pos, edge_index)
            h = self.global_attention(h, batch)
            for layer in self.egnn_layers_upper_post:
                h, pos = layer(h, pos, edge_index)
        else:
            for layer in self.egnn_layers_upper:
                h, pos = layer(h, pos, edge_index)

        # EQUIVARIANT noise prediction: coordinate displacement
        noise_pred = pos - pos_input

        # NaN safety: return zeros rather than NaN (prevents poison gradients)
        if torch.isnan(noise_pred).any():
            logger.warning("NaN in model output, returning zeros")
            noise_pred = torch.zeros_like(noise_pred)

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
    return F.smooth_l1_loss(distances, torch.full_like(distances, target_length))


def sphericity_loss(
    pos: torch.Tensor,
    batch: torch.Tensor,
    target_radius: float = 1.0,
    radius_weight: float = 0.5,
) -> torch.Tensor:
    """
    ENHANCED Sphericity loss: encourages compact spherical structures.
    
    v3 improvements (fixes fragmented structures):
    1. Asphericity penalty (shape should be spherical)
    2. Radius penalty (atoms should cluster at target radius)
    3. Compactness penalty (atoms should not be scattered)
    
    Computes normalized asphericity from gyration tensor.
    Uses relative asphericity = (λ1 - λ3) / (λ1 + λ2 + λ3) 
    Range: [0, 1] where 0 = perfect sphere, 1 = maximally elongated
    
    Notes:
        The default training pipeline normalizes each molecule so that the mean
        atomic radius is ~1.0 (see dataset.py radius normalization). Therefore
        the *normalized-space* target radius is 1.0.

    Args:
        pos: Coordinates [N, 3] (normalized)
        batch: Batch assignment [N]
        target_radius: Expected radius in the same units as pos (default=1.0)
        radius_weight: Weight for radius penalty vs asphericity
    
    Returns:
        loss: Combined sphericity + compactness loss
    """
    batch_size = batch.max().item() + 1
    asphericities = []
    radius_penalties = []
    
    for i in range(batch_size):
        # Get positions for this molecule
        mask = batch == i
        mol_pos = pos[mask]  # [N_i, 3]
        
        # Center coordinates
        centered = mol_pos - mol_pos.mean(dim=0, keepdim=True)
        
        # ===== Asphericity penalty (shape) =====
        # Gyration tensor: S = (1/N) X^T X
        S = torch.mm(centered.t(), centered) / centered.size(0)  # [3, 3]
        
        # Compute eigenvalues (sorted ascending: λ1 ≤ λ2 ≤ λ3)
        # Note: eigvalsh not implemented on MPS — fall back to CPU
        if S.device.type == 'mps':
            eigvals = torch.linalg.eigvalsh(S.cpu()).to(S.device)
        else:
            eigvals = torch.linalg.eigvalsh(S)  # [3], sorted ascending
        eigvals = torch.clamp(eigvals, min=1e-8)  # Avoid division by zero
        
        # Relative asphericity (normalized by total inertia)
        # Perfect sphere: λ1 = λ2 = λ3, asphericity = 0
        # Maximally elongated: λ1 = λ2 = 0, asphericity → 1
        trace = eigvals.sum()
        asphericity = (eigvals[2] - eigvals[0]) / (trace + 1e-8)
        asphericities.append(asphericity)
        
        # ===== Radius penalty (compactness) =====
        # Penalize atoms being too far from target radius
        radii = torch.norm(centered, dim=1)  # [N_i]
        mean_radius = radii.mean()
        
        # Penalty for wrong average radius
        radius_deviation = (mean_radius - target_radius).abs()
        
        # CRITICAL: Also penalize variance (atoms scattered at different radii)
        # This prevents fragmentation where some atoms are far away
        radius_variance = radii.var()
        
        # Combined radius penalty
        radius_penalty = radius_deviation + 0.5 * torch.sqrt(radius_variance + 1e-8)
        radius_penalties.append(radius_penalty)
    
    asphericity_loss = torch.mean(torch.stack(asphericities)) if asphericities else torch.tensor(0.0, device=pos.device, requires_grad=True)
    radius_loss = torch.mean(torch.stack(radius_penalties)) if radius_penalties else torch.tensor(0.0, device=pos.device, requires_grad=True)
    
    # Weighted combination
    # Asphericity is already 0-1 normalized, radius is in Angstroms
    # Balance them with radius_weight
    return asphericity_loss + radius_weight * radius_loss


if __name__ == '__main__':
    """Quick test of model architecture (v3 with Plan B + Plan D)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # === Test 1: Small graph (C60, below hierarchical threshold) ===
    num_nodes = 60
    pos = torch.randn(num_nodes, 3)
    
    edges = [
        [i, j]
        for i in range(num_nodes)
        for j in [(i+1) % num_nodes, (i+2) % num_nodes, (i-1) % num_nodes]
        if i < j
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    t = torch.tensor([500])
    C = torch.tensor([60])
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    model = FullereneDiffusionModel(
        hidden_dim=32, num_layers=4,
        continuous_C_embed=True, use_hierarchical=True,
        hierarchical_threshold=80,
    )
    
    logger.info("Model v3 created successfully")
    logger.info("  Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    
    with torch.no_grad():
        noise_pred = model(pos, edge_index, t, C, batch)
    
    logger.info("Test 1 (C60, below threshold):")
    logger.info("  Input shape: %s", pos.shape)
    logger.info("  Output shape: %s", noise_pred.shape)
    logger.info("  Output mean: %.4f", noise_pred.mean())
    logger.info("  Output std: %.4f", noise_pred.std())

    # === Test 2: Large graph (C200, above hierarchical threshold) ===
    num_nodes_lg = 200
    pos_lg = torch.randn(num_nodes_lg, 3)
    edges_lg = [
        [i, j]
        for i in range(num_nodes_lg)
        for j in [(i+1) % num_nodes_lg, (i+2) % num_nodes_lg, (i-1) % num_nodes_lg]
        if i < j
    ]
    edge_index_lg = torch.tensor(edges_lg, dtype=torch.long).t()
    t_lg = torch.tensor([500])
    C_lg = torch.tensor([200])
    batch_lg = torch.zeros(num_nodes_lg, dtype=torch.long)

    with torch.no_grad():
        noise_pred_lg = model(pos_lg, edge_index_lg, t_lg, C_lg, batch_lg)

    logger.info("Test 2 (C200, above threshold — hierarchical active):")
    logger.info("  Input shape: %s", pos_lg.shape)
    logger.info("  Output shape: %s", noise_pred_lg.shape)
    logger.info("  Output mean: %.4f", noise_pred_lg.mean())
    logger.info("  Output std: %.4f", noise_pred_lg.std())

    # === Test 3: Unseen C value (C350 — Plan B generalisation) ===
    C_unseen = torch.tensor([350])
    with torch.no_grad():
        noise_pred_unseen = model(pos_lg, edge_index_lg, t_lg, C_unseen, batch_lg)

    logger.info("Test 3 (C350, unseen C — continuous embed generalisation):")
    logger.info("  Output mean: %.4f", noise_pred_unseen.mean())
    logger.info("  Output std: %.4f", noise_pred_unseen.std())
    
    # Test bond length loss
    loss = bond_length_loss(pos, edge_index, target_length=1.39)
    logger.info("Bond length loss: %.4f", loss.item())
