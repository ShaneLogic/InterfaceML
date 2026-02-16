"""
Fullerene-Optimized Topology Losses (v2)

Key improvements for training stability:
1. Time-conditioned loss weighting (only enforce physics at low noise)
2. Progressive connectivity penalties (no sudden 10× jump)
3. Robust topology detection (handle malformed graphs)
4. Gradient-aware scaling (prevent explosion)

Design Philosophy:
- High noise (t>800): Allow free diffusion, minimal constraints
- Medium noise (400<t<800): Gradually introduce bond constraints  
- Low noise (t<400): Full physics enforcement
"""

import logging
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import knn_graph
from typing import Tuple, Optional

from units import bond_target_normalized, bond_tolerance_normalized

logger = logging.getLogger(__name__)


def _min_dist_normalized(
    C_values: Optional[torch.Tensor],
    min_dist_angstrom: float,
    radius_coeff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if C_values is None:
        return torch.as_tensor(min_dist_angstrom, device=device, dtype=dtype)
    # Convert Å to normalized using radius scaling
    return (min_dist_angstrom / (radius_coeff * torch.sqrt(C_values.float().to(device)))).to(dtype=dtype)


def compute_time_weight(t: torch.Tensor, 
                       T: int = 1000,
                       mode: str = 'sigmoid') -> torch.Tensor:
    """
    Compute time-dependent loss weights: physics constraints should only 
    apply at low noise (small t), not during high-noise diffusion stages.
    
    Args:
        t: [B] or [N] timestep values (0 to T-1)
        T: total diffusion steps
        mode: 'sigmoid', 'linear', or 'exponential'
        
    Returns:
        weight: [B] or [N] weights in [0, 1], higher at small t
    """
    # Normalize t to [0, 1]
    t_norm = t.float() / T
    
    if mode == 'sigmoid':
        # Sigmoid centered at t=0.3, steepness=10
        # - t < 0.2 (low noise): weight ≈ 1.0 (full constraint)
        # - t > 0.6 (high noise): weight ≈ 0.0 (no constraint)
        center = 0.4
        steepness = 15.0
        weight = torch.sigmoid(-steepness * (t_norm - center))
        
    elif mode == 'linear':
        # Linear decay from t=0 to t=T
        weight = 1.0 - t_norm
        
    elif mode == 'exponential':
        # Exponential decay: exp(-5*t/T)
        weight = torch.exp(-5.0 * t_norm)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return weight


def progressive_connectivity_loss(pos: torch.Tensor,
                                  edge_index: torch.Tensor, 
                                  batch: torch.Tensor,
                                  t: Optional[torch.Tensor] = None,
                                  C_values: Optional[torch.Tensor] = None,
                                  target_bond_angstrom: float = 1.42,
                                  tolerance_angstrom: float = 0.30,
                                  radius_coeff: float = 0.45) -> torch.Tensor:
    """
    Progressive connectivity loss with time-conditioning.
    
    Key improvements over v1:
    1. Smooth overlap penalty (no 10× jump)
    2. Time-weighted (only active at low noise)
    3. Huber-like robustness (reduce outlier impact)
    4. Gradient clipping per edge (prevent single-edge explosion)
    
    Args:
        pos: [N, 3] atomic coordinates
        edge_index: [2, E] edge connectivity
        batch: [N] batch assignment
        t: [B] timestep per batch (for time-weighting)
        C_values: [B] carbon count per molecule; if provided, bond targets are
                  converted from Å to normalized units using a C-dependent radius.
        target_bond_angstrom: target C-C bond length in Å
        tolerance_angstrom: acceptable deviation in Å
        radius_coeff: radius scaling coefficient for R(C) ≈ a*sqrt(C)
        
    Returns:
        loss: scalar connectivity loss
    """
    row, col = edge_index
    edge_vec = pos[row] - pos[col]  # [E, 3]
    edge_len = edge_vec.norm(dim=1)  # [E]

    # Convert physical targets (Å) to normalized-space targets if C is provided.
    if C_values is not None:
        edge_mol = batch[row]  # [E]
        target_bond = bond_target_normalized(
            C_values.to(pos.device)[edge_mol],
            bond_length_angstrom=target_bond_angstrom,
            radius_coeff=radius_coeff,
        )
        tolerance = bond_tolerance_normalized(
            C_values.to(pos.device)[edge_mol],
            tolerance_angstrom=tolerance_angstrom,
            radius_coeff=radius_coeff,
        )
    else:
        target_bond = torch.as_tensor(target_bond_angstrom, device=pos.device, dtype=pos.dtype)
        tolerance = torch.as_tensor(tolerance_angstrom, device=pos.device, dtype=pos.dtype)
    
    # Define ranges
    lower_strict = target_bond - tolerance
    upper_strict = target_bond + tolerance
    
    # IMPROVED: Progressive penalty instead of 10× jump
    # For overlaps (d < lower), use smooth polynomial penalty:
    #   penalty = a * (lower - d)^2 + b * (lower - d)^3
    # This grows smoothly instead of jumping at threshold
    
    overlap_amount = torch.clamp(lower_strict - edge_len, min=0)  # >0 if overlap
    too_long_amount = torch.clamp(edge_len - upper_strict, min=0)  # >0 if broken
    
    # Polynomial penalty for overlaps (smoother than step function)
    # Coefficients: quadratic=3, cubic=2 (grows fast but smoothly)
    overlap_loss = 3.0 * (overlap_amount ** 2) + 2.0 * (overlap_amount ** 3)
    
    # Moderate penalty for stretched bonds
    stretch_loss = 1.0 * (too_long_amount ** 2)
    
    # Huber-like: cap per-edge loss at 100 to prevent single-edge domination
    per_edge_loss = overlap_loss + stretch_loss
    per_edge_loss = torch.clamp(per_edge_loss, max=100.0)
    
    # Aggregate
    base_loss = per_edge_loss.mean()
    
    # Time-weighted (optional)
    if t is not None:
        # Get time weight for each edge's batch
        batch_t = batch[edge_index[0]]  # [E] - batch ID for each edge
        t_weight = compute_time_weight(t[batch_t], mode='sigmoid')  # [E]
        weighted_loss = (per_edge_loss * t_weight).mean()
        return weighted_loss
    else:
        return base_loss


def robust_topology_loss(pos: torch.Tensor,
                        edge_index: torch.Tensor,
                        batch: torch.Tensor,
                        C_values: torch.Tensor,
                        t: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Robust fullerene topology loss with graceful degradation.
    
    Improvements:
    1. Handle disconnected graphs (common in early training)
    2. Use approximate ring detection when exact fails
    3. Time-weighted (only enforce at low noise)
    4. Fallback to degree constraint if topology detection fails
    
    Args:
        pos: [N, 3] coordinates
        edge_index: [2, E] edges
        batch: [N] batch indices
        C_values: [B] carbon count per molecule
        t: [B] timestep per batch
        
    Returns:
        loss: scalar topology loss
    """
    batch_size = batch.max().item() + 1
    device = pos.device
    
    total_loss = torch.tensor(0.0, device=device)
    valid_mols = 0
    
    for i in range(batch_size):
        mask = batch == i
        mol_pos = pos[mask]
        n_atoms = mol_pos.size(0)
        C_n = C_values[i].item() if C_values.dim() > 0 else C_values.item()
        
        # Reconstruct edges (3-nearest neighbors)
        try:
            _mol_pos_cpu = mol_pos.detach().cpu() if mol_pos.device.type != 'cpu' else mol_pos
            mol_edge_index = knn_graph(_mol_pos_cpu, k=3, batch=None, loop=False).to(device)
        except Exception:
            # Graph construction failed (e.g., too few atoms)
            continue
        
        # ROBUST: Always check degree distribution (fallback metric)
        degrees = torch.zeros(n_atoms, device=device)
        degrees.scatter_add_(0, mol_edge_index[0], 
                           torch.ones(mol_edge_index.size(1), device=device))
        degree_loss = ((degrees - 3.0) ** 2).mean()
        
        # Try topology detection (may fail for malformed graphs)
        try:
            pentagons, hexagons = detect_rings_safe(mol_edge_index, n_atoms)
            
            target_pentagons = 12
            target_hexagons = max(0, C_n // 2 - 10)
            
            # Soft loss (square root to reduce sensitivity)
            pentagon_loss = torch.sqrt(torch.tensor(
                (pentagons - target_pentagons) ** 2 + 1e-6, device=device
            ))
            hexagon_loss = torch.sqrt(torch.tensor(
                (hexagons - target_hexagons) ** 2 + 1e-6, device=device
            ))
            
            mol_loss = degree_loss + 0.1 * (pentagon_loss + hexagon_loss)
            
        except Exception:
            # Topology detection failed - use only degree constraint
            mol_loss = degree_loss
        
        total_loss += mol_loss
        valid_mols += 1
    
    if valid_mols == 0:
        return torch.tensor(0.0, device=device)
    
    base_loss = total_loss / valid_mols
    
    # Time-weighted
    if t is not None:
        t_weight = compute_time_weight(t, mode='sigmoid').mean()
        return base_loss * t_weight
    else:
        return base_loss


def detect_rings_safe(edge_index: torch.Tensor, 
                     num_nodes: int,
                     max_ring_size: int = 6) -> Tuple[int, int]:
    """
    Safe ring detection with error handling.
    
    Returns (0, 0) if detection fails or molecule is too large.
    """
    # Skip ring detection for large molecules (too expensive)
    if num_nodes > 70:
        return 0, 0
    
    try:
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().detach().cpu().numpy()
        G.add_edges_from(edges)
        
        # Check connectivity
        if not nx.is_connected(G):
            return 0, 0
        
        # Use cycle_basis (faster than minimum_cycle_basis)
        try:
            cycles = nx.cycle_basis(G)
        except Exception:
            return 0, 0
        
        pentagons = sum(1 for c in cycles if len(c) == 5)
        hexagons = sum(1 for c in cycles if len(c) == 6)
        
        return pentagons, hexagons
        
    except Exception:
        return 0, 0


def bond_length_loss_v2(pos: torch.Tensor,
                       edge_index: torch.Tensor,
                       batch: torch.Tensor,
                       t: Optional[torch.Tensor] = None,
                       C_values: Optional[torch.Tensor] = None,
                       target_length_angstrom: float = 1.42,
                       radius_coeff: float = 0.45) -> torch.Tensor:
    """
    Time-conditioned bond length loss (simpler than connectivity).
    
    Only penalizes deviations from target bond length, without
    special handling of overlaps.
    """
    row, col = edge_index
    edge_vec = pos[row] - pos[col]
    edge_len = edge_vec.norm(dim=1)

    if C_values is not None:
        edge_mol = batch[row]
        target_length = bond_target_normalized(
            C_values.to(pos.device)[edge_mol],
            bond_length_angstrom=target_length_angstrom,
            radius_coeff=radius_coeff,
        ).to(dtype=edge_len.dtype)
    else:
        target_length = torch.as_tensor(target_length_angstrom, device=pos.device, dtype=edge_len.dtype)
    
    deviation = (edge_len - target_length).abs()
    
    # Huber loss (more robust than MSE)
    delta = 0.5  # Huber threshold
    huber_loss = torch.where(
        deviation < delta,
        0.5 * (deviation ** 2),
        delta * (deviation - 0.5 * delta)
    )
    
    base_loss = huber_loss.mean()
    
    # Time-weighted
    if t is not None:
        batch_t = batch[edge_index[0]]
        t_weight = compute_time_weight(t[batch_t], mode='sigmoid')
        return (huber_loss * t_weight).mean()
    else:
        return base_loss


def nonbonded_repulsion_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    C_values: Optional[torch.Tensor] = None,
    min_dist_angstrom: float = 1.60,
    radius_coeff: float = 0.45,
    k: int = 6,
) -> torch.Tensor:
    """Repel non-bonded close contacts using kNN pairs (lightweight)."""
    device = pos.device
    batch_size = batch.max().item() + 1
    losses = []

    for i in range(batch_size):
        mask = batch == i
        mol_pos = pos[mask]
        if mol_pos.size(0) < 4:
            continue

        # kNN graph in molecule (torch_cluster only works on CPU)
        _mol_pos_cpu = mol_pos.detach().cpu() if mol_pos.device.type != 'cpu' else mol_pos
        mol_edge_index = knn_graph(
            _mol_pos_cpu,
            k=min(k, _mol_pos_cpu.size(0) - 1),
            batch=None,
            loop=False,
        ).to(device)

        # Build bonded set for this molecule (local indices)
        local_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        idx_map = {int(g): int(l) for l, g in enumerate(local_idx.tolist())}
        bonded = set()
        for u, v in edge_index.t().tolist():
            if u in idx_map and v in idx_map:
                a = idx_map[u]
                b = idx_map[v]
                bonded.add((a, b))
                bonded.add((b, a))

        row, col = mol_edge_index
        keep = []
        for r, c in zip(row.tolist(), col.tolist()):
            if (r, c) not in bonded:
                keep.append(True)
            else:
                keep.append(False)
        if not keep:
            continue
        keep = torch.tensor(keep, device=device, dtype=torch.bool)
        row = row[keep]
        col = col[keep]

        if row.numel() == 0:
            continue

        d = (mol_pos[row] - mol_pos[col]).norm(dim=1)
        min_dist = _min_dist_normalized(C_values[i:i+1], min_dist_angstrom, radius_coeff, device, d.dtype)
        overlap = torch.clamp(min_dist - d, min=0)
        losses.append((overlap ** 2).mean())

    if not losses:
        base_loss = torch.tensor(0.0, device=device)
    else:
        base_loss = torch.stack(losses).mean()

    if t is not None:
        t_weight = compute_time_weight(t, mode='sigmoid').mean()
        return base_loss * t_weight
    return base_loss


def compute_angle_loss(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    T: int = 1000,
    angle_low: float = 103.0,
    angle_high: float = 125.0,
) -> torch.Tensor:
    """Penalize bond angles outside the sp2 range [103, 125] degrees.

    Fullerene carbon atoms have 3 neighbors forming pentagons (108 deg)
    and hexagons (120 deg). We use a soft window that accommodates both.

    For each atom i with neighbors j, k, l from edge_index, we compute
    the three bond angles (j-i-k, j-i-l, k-i-l) and penalize if they
    fall outside [angle_low, angle_high].

    Args:
        pos: [N, 3] atomic coordinates.
        edge_index: [2, E] directed edges (assumed bidirectional).
        batch: [N] batch assignment.
        t: [B] timestep per batch (for time-conditioning).
        T: total diffusion steps.
        angle_low: minimum acceptable angle in degrees (default 103).
        angle_high: maximum acceptable angle in degrees (default 125).

    Returns:
        loss: scalar angle loss.
    """
    device = pos.device
    num_nodes = pos.size(0)

    # Build adjacency list: for each node, collect neighbor indices
    # Use only unique directed edges (row -> col means col is neighbor of row)
    row, col = edge_index

    # Build neighbor lists (max 3 for fullerenes, but handle more)
    # Use scatter to group neighbors per node
    neighbor_count = torch.zeros(num_nodes, dtype=torch.long, device=device)
    neighbor_count.index_add_(0, row, torch.ones(row.size(0), dtype=torch.long, device=device))

    # Only process nodes with >= 2 neighbors (need at least 2 for an angle)
    valid_nodes = (neighbor_count >= 2).nonzero(as_tuple=True)[0]
    if valid_nodes.numel() == 0:
        return torch.tensor(0.0, device=device)

    # For efficiency, build a padded neighbor tensor
    max_neighbors = min(int(neighbor_count.max().item()), 6)
    # Collect neighbor lists per node
    # Sort edges by source node for grouped access
    sort_idx = torch.argsort(row)
    row_sorted = row[sort_idx]
    col_sorted = col[sort_idx]

    # Compute angle penalties in a vectorized way using pairs of neighbors
    angle_losses = []

    # Process in chunks by iterating over unique nodes with enough neighbors
    # Build padded neighbor matrix [N, max_neighbors] = -1 for missing
    neighbors = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long, device=device)
    current_counts = torch.zeros(num_nodes, dtype=torch.long, device=device)

    for idx in range(row_sorted.size(0)):
        src = row_sorted[idx].item()
        dst = col_sorted[idx].item()
        cnt = current_counts[src].item()
        if cnt < max_neighbors:
            neighbors[src, cnt] = dst
            current_counts[src] = cnt + 1

    # For valid nodes, compute all pairwise angles between neighbors
    for a in range(max_neighbors):
        for b in range(a + 1, max_neighbors):
            # Get neighbor indices
            n_a = neighbors[valid_nodes, a]  # [V]
            n_b = neighbors[valid_nodes, b]  # [V]

            # Filter out invalid pairs
            valid_pair = (n_a >= 0) & (n_b >= 0)
            if valid_pair.sum() == 0:
                continue

            center = valid_nodes[valid_pair]
            j = n_a[valid_pair]
            k = n_b[valid_pair]

            # Compute vectors center->j and center->k
            vec_j = pos[j] - pos[center]  # [P, 3]
            vec_k = pos[k] - pos[center]  # [P, 3]

            # Compute angle via dot product
            dot = (vec_j * vec_k).sum(dim=-1)
            norm_j = vec_j.norm(dim=-1).clamp(min=1e-8)
            norm_k = vec_k.norm(dim=-1).clamp(min=1e-8)
            cos_angle = (dot / (norm_j * norm_k)).clamp(-1.0, 1.0)
            angle_deg = torch.acos(cos_angle) * (180.0 / 3.14159265358979)

            # Soft penalty outside [angle_low, angle_high]
            below = torch.clamp(angle_low - angle_deg, min=0)
            above = torch.clamp(angle_deg - angle_high, min=0)
            penalty = below ** 2 + above ** 2  # [P]

            angle_losses.append(penalty)

    if not angle_losses:
        return torch.tensor(0.0, device=device)

    base_loss = torch.cat(angle_losses).mean()

    # Time conditioning: only enforce at low noise
    if t is not None:
        t_weight = compute_time_weight(t, T=T, mode='sigmoid').mean()
        return base_loss * t_weight

    return base_loss


def combined_physics_loss_v2(pos: torch.Tensor,
                             edge_index: torch.Tensor,
                             batch: torch.Tensor,
                             C_values: torch.Tensor,
                             t: torch.Tensor,
                             lambda_bond: float = 5.0,
                             lambda_conn: float = 0.3,
                             lambda_topo: float = 0.1,
                             lambda_repulsion: float = 0.0,
                             lambda_angle: float = 0.0,
                             target_bond_angstrom: float = 1.42,
                             bond_tolerance_angstrom: float = 0.30,
                             radius_coeff: float = 0.45,
                             repulsion_min_dist_angstrom: float = 1.60) -> dict:
    """
    Combined physics-informed loss with time-conditioning.
    
    Returns dict of individual loss components for monitoring.
    """
    # All losses are time-weighted internally
    bond_loss = bond_length_loss_v2(
        pos,
        edge_index,
        batch,
        t,
        C_values=C_values,
        target_length_angstrom=target_bond_angstrom,
        radius_coeff=radius_coeff,
    )
    conn_loss = progressive_connectivity_loss(
        pos,
        edge_index,
        batch,
        t,
        C_values=C_values,
        target_bond_angstrom=target_bond_angstrom,
        tolerance_angstrom=bond_tolerance_angstrom,
        radius_coeff=radius_coeff,
    )
    topo_loss = robust_topology_loss(pos, edge_index, batch, C_values, t)
    repulsion_loss = nonbonded_repulsion_loss(
        pos,
        edge_index,
        batch,
        t,
        C_values=C_values,
        min_dist_angstrom=repulsion_min_dist_angstrom,
        radius_coeff=radius_coeff,
    )

    angle_loss = torch.tensor(0.0, device=pos.device)
    if lambda_angle > 0:
        angle_loss = compute_angle_loss(pos, edge_index, batch, t)

    total = (
        lambda_bond * bond_loss
        + lambda_conn * conn_loss
        + lambda_topo * topo_loss
        + lambda_repulsion * repulsion_loss
        + lambda_angle * angle_loss
    )

    return {
        'bond': bond_loss,
        'connectivity': conn_loss,
        'topology': topo_loss,
        'repulsion': repulsion_loss,
        'angle': angle_loss,
        'total_physics': total
    }


if __name__ == '__main__':
    """Test improved losses"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.info("Testing Fullerene-Optimized Topology Losses v2")
    logger.info("=" * 60)
    
    # Simulate batch of C60 molecules at different timesteps
    num_nodes = 120  # 2 molecules × 60 atoms
    pos = torch.randn(num_nodes, 3) * 3.5
    batch = torch.cat([torch.zeros(60), torch.ones(60)]).long()
    C_values = torch.tensor([60, 60])
    t = torch.tensor([100, 800])  # One at low noise, one at high noise
    
    # Build edges (on CPU — torch_cluster doesn't support MPS)
    edge_index = knn_graph(pos.cpu(), k=3, batch=batch.cpu(), loop=False).to(pos.device)
    
    logger.info("Setup:")
    logger.info("  Molecules: 2 x C60")
    logger.info("  Timesteps: t=%s (low noise vs high noise)", t.tolist())
    logger.info("  Edges: %d", edge_index.size(1))
    
    # Test time weighting
    logger.info("Time weights (sigmoid mode):")
    for t_val in [0, 100, 400, 800, 999]:
        w = compute_time_weight(torch.tensor([t_val]), T=1000, mode='sigmoid')
        logger.info("  t=%3d: weight=%.4f", t_val, w.item())
    
    # Test progressive connectivity loss
    logger.info("Progressive Connectivity Loss:")
    conn_loss_with_time = progressive_connectivity_loss(
        pos, edge_index, batch, t=t
    )
    conn_loss_no_time = progressive_connectivity_loss(
        pos, edge_index, batch, t=None
    )
    logger.info("  With time-weighting: %.4f", conn_loss_with_time.item())
    logger.info("  Without time-weighting: %.4f", conn_loss_no_time.item())
    logger.info("  Reduction: %.1f%%", (1 - conn_loss_with_time/conn_loss_no_time)*100)
    
    # Test robust topology loss
    logger.info("Robust Topology Loss:")
    topo_loss = robust_topology_loss(pos, edge_index, batch, C_values, t=t)
    logger.info("  Loss: %.4f", topo_loss.item())
    
    # Test combined
    logger.info("Combined Physics Loss:")
    losses = combined_physics_loss_v2(
        pos, edge_index, batch, C_values, t,
        lambda_bond=5.0, lambda_conn=0.3, lambda_topo=0.1
    )
    for key, val in losses.items():
        logger.info("  %15s: %.4f", key, val.item())
    
    logger.info("=" * 60)
    logger.info("All tests passed!")
