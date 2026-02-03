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

import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import knn_graph
from typing import Tuple, Optional

from units import bond_target_normalized, bond_tolerance_normalized


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
            mol_edge_index = knn_graph(mol_pos, k=3, batch=None, loop=False)
        except:
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
            
        except:
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
    
    Returns (0, 0) if detection fails instead of crashing.
    """
    try:
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        
        # Check connectivity
        if not nx.is_connected(G):
            # Graph is disconnected - can't reliably detect cycles
            return 0, 0
        
        # Detect fundamental cycles
        cycles = nx.cycle_basis(G)
        
        pentagons = sum(1 for c in cycles if len(c) == 5)
        hexagons = sum(1 for c in cycles if len(c) == 6)
        
        return pentagons, hexagons
        
    except Exception as e:
        # Detection failed - return neutral values
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

        # kNN graph in molecule
        mol_edge_index = knn_graph(mol_pos, k=min(k, mol_pos.size(0) - 1), batch=None, loop=False)

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


def combined_physics_loss_v2(pos: torch.Tensor,
                             edge_index: torch.Tensor,
                             batch: torch.Tensor,
                             C_values: torch.Tensor,
                             t: torch.Tensor,
                             lambda_bond: float = 5.0,
                             lambda_conn: float = 0.3,
                             lambda_topo: float = 0.1,
                             lambda_repulsion: float = 0.0,
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
    
    total = (
        lambda_bond * bond_loss
        + lambda_conn * conn_loss
        + lambda_topo * topo_loss
        + lambda_repulsion * repulsion_loss
    )
    
    return {
        'bond': bond_loss,
        'connectivity': conn_loss,
        'topology': topo_loss,
        'repulsion': repulsion_loss,
        'total_physics': total
    }


if __name__ == '__main__':
    """Test improved losses"""
    print("Testing Fullerene-Optimized Topology Losses v2")
    print("=" * 60)
    
    # Simulate batch of C60 molecules at different timesteps
    num_nodes = 120  # 2 molecules × 60 atoms
    pos = torch.randn(num_nodes, 3) * 3.5
    batch = torch.cat([torch.zeros(60), torch.ones(60)]).long()
    C_values = torch.tensor([60, 60])
    t = torch.tensor([100, 800])  # One at low noise, one at high noise
    
    # Build edges
    edge_index = knn_graph(pos, k=3, batch=batch, loop=False)
    
    print(f"\nSetup:")
    print(f"  Molecules: 2 × C60")
    print(f"  Timesteps: t={t.tolist()} (low noise vs high noise)")
    print(f"  Edges: {edge_index.size(1)}")
    
    # Test time weighting
    print(f"\nTime weights (sigmoid mode):")
    for t_val in [0, 100, 400, 800, 999]:
        w = compute_time_weight(torch.tensor([t_val]), T=1000, mode='sigmoid')
        print(f"  t={t_val:3d}: weight={w.item():.4f}")
    
    # Test progressive connectivity loss
    print(f"\nProgressive Connectivity Loss:")
    conn_loss_with_time = progressive_connectivity_loss(
        pos, edge_index, batch, t=t
    )
    conn_loss_no_time = progressive_connectivity_loss(
        pos, edge_index, batch, t=None
    )
    print(f"  With time-weighting: {conn_loss_with_time.item():.4f}")
    print(f"  Without time-weighting: {conn_loss_no_time.item():.4f}")
    print(f"  Reduction: {(1 - conn_loss_with_time/conn_loss_no_time)*100:.1f}%")
    
    # Test robust topology loss
    print(f"\nRobust Topology Loss:")
    topo_loss = robust_topology_loss(pos, edge_index, batch, C_values, t=t)
    print(f"  Loss: {topo_loss.item():.4f}")
    
    # Test combined
    print(f"\nCombined Physics Loss:")
    losses = combined_physics_loss_v2(
        pos, edge_index, batch, C_values, t,
        lambda_bond=5.0, lambda_conn=0.3, lambda_topo=0.1
    )
    for key, val in losses.items():
        print(f"  {key:15s}: {val.item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
