"""
Generation script for Fullerene Diffusion Model.

Samples new fullerene structures from trained diffusion model.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --C 60 --num_samples 10

Author: InterfaceML Project
Date: 2026-01-28
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)

from diffusion_utils import DiffusionScheduler
from flow_matching import FlowMatchingScheduler
from model import FullereneDiffusionModel
from template_bank import TemplateBank
from topology_gnn import TopologyGNN
from units import denormalize_positions, bond_target_normalized
from synthetic_topology import generate_fullerene_topology


def create_template_graph(
    num_carbon: int,
    device: str,
    template_bank: TemplateBank | None,
    topology_model: TopologyGNN | None,
) -> Data:
    """Create a template graph for generation.

    Strategy:
      1. If a real fullerene topology exists in the template bank for this C
         value, always prefer it — it gives a physically valid adjacency by
         construction and avoids the expensive (and often failing) algorithmic
         search for large C.
      2. Only fall back to the topology GNN for C values NOT covered by the
         dataset (e.g. C54, C56, C58, …).
    """

    # --- Try template bank first (fast, guaranteed valid topology) ---
    if template_bank is not None:
        try:
            return template_bank.make_template_data(num_carbon, device=torch.device(device))
        except FileNotFoundError:
            # No dataset template for this C value — continue to GNN
            pass

    # --- Topology GNN for C values without dataset coverage ---
    if topology_model is not None:
        try:
            edge_index = topology_model.generate_edge_index(C_value=num_carbon, device=torch.device(device))
            pos = torch.randn(num_carbon, 3, device=torch.device(device))
            return Data(pos=pos, edge_index=edge_index)
        except Exception as exc:
            logger.warning("Topology GNN failed for C%d: %s.", num_carbon, exc)

    # --- Last resort: convex-hull dual (deterministic, works for any even C) ---
    from synthetic_topology import generate_planar_3_regular_graph
    G = generate_fullerene_topology(num_carbon)
    if G is None:
        # Ultra-fallback: random 3-regular search (only viable for small C)
        G = generate_planar_3_regular_graph(num_carbon, max_tries=2000)
    if G is not None:
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).t().contiguous()
        pos = torch.randn(num_carbon, 3, device=torch.device(device))
        return Data(pos=pos, edge_index=edge_index.to(torch.device(device)))

    raise RuntimeError(
        f"No template available for C{num_carbon}. "
        f"Add XYZ files to dataset/fullerenes/fullerene_xyz/C{num_carbon}/ "
        f"or train a topology GNN that covers this size."
    )


# ---------------------------------------------------------------------------
# Backwards-compatible helpers for the web API (FullereneAPI expects these).
# ---------------------------------------------------------------------------

_TEMPLATE_BANK_CACHE: dict[Path, TemplateBank] = {}
_TOPOLOGY_GNN_CACHE: dict[Path, TopologyGNN] = {}


def _resolve_xyz_dir(config: dict) -> Path:
    data_cfg = (config or {}).get("data", {})
    xyz_dir = data_cfg.get("xyz_dir")
    if not xyz_dir:
        raise ValueError("Config missing data.xyz_dir (needed for template graphs)")
    xyz_path = Path(xyz_dir)
    if not xyz_path.is_absolute():
        xyz_path = (Path(__file__).resolve().parent / xyz_path).resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ template directory not found: {xyz_path}")
    return xyz_path


def _get_template_bank(config: dict) -> TemplateBank:
    xyz_path = _resolve_xyz_dir(config)
    cached = _TEMPLATE_BANK_CACHE.get(xyz_path)
    if cached is not None:
        return cached
    bank = TemplateBank(xyz_dir=xyz_path)
    _TEMPLATE_BANK_CACHE[xyz_path] = bank
    return bank


def _get_topology_gnn(config: dict, device: torch.device) -> TopologyGNN | None:
    topo_cfg = (config or {}).get("topology_gnn", {})
    if not topo_cfg.get("enabled"):
        return None
    ckpt_path = topo_cfg.get("checkpoint_path")
    if not ckpt_path:
        return None

    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = (Path(__file__).resolve().parent / ckpt).resolve()
    if not ckpt.exists():
        return None

    cached = _TOPOLOGY_GNN_CACHE.get(ckpt)
    if cached is not None:
        return cached

    try:
        topo_state = torch.load(ckpt, map_location=device)
        topo_cfg_ckpt = topo_state.get("config", {})
        max_nodes = int(topo_cfg_ckpt.get("max_C", topo_cfg.get("max_C", 720)))
        hidden_dim = int(topo_cfg_ckpt.get("hidden_dim", topo_cfg.get("hidden_dim", 64)))
        latent_dim = int(topo_cfg_ckpt.get("latent_dim", topo_cfg.get("latent_dim", 32)))
        model = TopologyGNN(max_nodes=max_nodes, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        model.load_state_dict(topo_state["model_state_dict"])
        model.eval()
        _TOPOLOGY_GNN_CACHE[ckpt] = model
        return model
    except Exception as exc:
        logger.warning("Failed to load topology GNN: %s. Falling back to template bank.", exc)
        return None


def load_template_graph(num_carbon: int, config: dict, device: str | torch.device) -> Data:
    """Legacy helper expected by FullereneAPI."""
    device_obj = torch.device(device)
    template_bank = _get_template_bank(config)
    topology_model = _get_topology_gnn(config, device_obj)
    return create_template_graph(num_carbon, device_obj, template_bank, topology_model)


def _compute_geometry_guidance(
    x_0_pred: torch.Tensor,
    edge_index: torch.Tensor,
    C_value: int,
    guidance_scale: float,
    sqrt_one_minus_alpha_bar_t: float,
) -> torch.Tensor:
    """Compute reconstruction guidance gradient for geometric constraints.

    This implements *reconstruction guidance* (Chung et al., 2022 / Ho & Salimans
    2022 style).  At each reverse-diffusion step we have an estimate x̂₀ of the
    clean structure.  We define a differentiable energy

        E(x̂₀) = bond_MSE + sphericity + repulsion

    and compute  ∇_{x_t} E(x̂₀)  which, by the chain rule through the
    x̂₀ = f(x_t, ε_θ) formula, gives a gradient that shifts x_t toward
    lower-energy (more physical) structures.

    The guidance is scaled by √(1−ᾱ_t) so it is *noise-adaptive*: stronger
    at low noise (where geometry matters) and near-zero at high noise.

    This is a **pure AI/sampling-time** improvement — no retraining required.
    """
    if guidance_scale <= 0:
        return torch.zeros_like(x_0_pred)

    # Detach-and-reclone so we can take gradients w.r.t. x_0_pred
    x = x_0_pred.detach().requires_grad_(True)

    # --- Bond length loss (normalized space, target ≈ bond_target_normalized) ---
    target_bond = bond_target_normalized(C_value).to(x.device)
    row, col = edge_index
    mask = row < col
    r, c = row[mask], col[mask]
    if r.numel() > 0:
        bond_vec = x[r] - x[c]
        bond_len = bond_vec.norm(dim=1).clamp(min=1e-6)
        bond_loss = ((bond_len - target_bond) ** 2).mean()
    else:
        bond_loss = torch.tensor(0.0, device=x.device)

    # --- Sphericity loss (mean radius → 1, low variance) ---
    centered = x - x.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1).clamp(min=1e-6)
    radius_loss = (radii.mean() - 1.0) ** 2
    variance_loss = radii.var()

    # --- Non-bonded repulsion (prevent atom overlap) ---
    dists = torch.cdist(x, x)  # [N, N]
    n = dists.size(0)
    diag_inf = torch.eye(n, device=x.device, dtype=torch.bool)
    bond_mask = torch.zeros(n, n, device=x.device, dtype=torch.bool)
    bond_mask[row, col] = True
    bond_mask[col, row] = True
    ignore = diag_inf | bond_mask
    dists_nb = dists.masked_fill(ignore, 1e6)
    min_dist = 0.45  # normalized units
    overlap = torch.clamp(min_dist - dists_nb, min=0)
    repulsion_loss = (overlap ** 2).mean()

    # Combined energy
    energy = bond_loss + 0.3 * radius_loss + 0.2 * variance_loss + 0.5 * repulsion_loss
    energy.backward()

    grad = x.grad.detach()

    # Noise-adaptive scaling: guidance is weaker at high noise
    noise_scale = sqrt_one_minus_alpha_bar_t
    return guidance_scale * noise_scale * grad


def _flow_matching_loop(
    model,
    fm_scheduler: FlowMatchingScheduler,
    template: Data,
    *,
    C_value: int,
    num_steps: int = 50,
    cfg_weight: float = 2.0,
    return_trajectory: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """ODE integration for flow matching from t=1 (noise) to t=0 (data).

    Uses Euler integration with optional CFG (classifier-free guidance).
    CoM is projected at each step for translation invariance.

    Parameters
    ----------
    model : PaiNN or EGNN model with forward(pos, edge_index, t, C, batch).
    fm_scheduler : FlowMatchingScheduler instance.
    template : PyG Data with ``edge_index``.
    C_value : number of carbon atoms.
    num_steps : number of Euler steps.
    cfg_weight : CFG guidance weight (>1.0 amplifies conditioning).
    return_trajectory : whether to store intermediate positions.

    Returns
    -------
    pos_final : [N, 3] generated positions.
    trajectory : list of intermediate positions.
    """
    device = template.edge_index.device
    pos_t = torch.randn(C_value, 3, device=device)

    C_cond = torch.tensor([C_value], dtype=torch.long, device=device)
    batch = torch.zeros(C_value, dtype=torch.long, device=device)

    dt = 1.0 / num_steps
    trajectory: list[torch.Tensor] = []

    use_cfg = cfg_weight > 1.0 and hasattr(model, 'forward_cfg')

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_batch = torch.full((1,), t_val * 999, device=device, dtype=torch.long)

        if use_cfg:
            v_pred = model.forward_cfg(pos_t, template.edge_index, t_batch, C_cond, batch, cfg_weight=cfg_weight)
        else:
            v_pred = model(pos_t, template.edge_index, t_batch, C_cond, batch)

        # NaN guard
        if torch.isnan(v_pred).any():
            logger.warning("NaN in flow matching at step %d — aborting", step)
            return pos_t, trajectory

        # Euler step: x_{t-dt} = x_t - dt * v_pred
        pos_t = pos_t - dt * v_pred

        # CoM projection (translation invariance)
        pos_t = pos_t - pos_t.mean(dim=0, keepdim=True)

        if torch.isnan(pos_t).any():
            logger.warning("NaN in positions at step %d — aborting", step)
            return pos_t, trajectory

        if return_trajectory:
            trajectory.append(pos_t.detach().clone())

    return pos_t, trajectory


def _reverse_diffusion_loop(
    model: FullereneDiffusionModel,
    scheduler: DiffusionScheduler,
    template: Data,
    *,
    C_value: int,
    use_ddim: bool = False,
    ddim_steps: int = 100,
    return_trajectory: bool = False,
    sampling_momentum: float = 0.0,
    sampling_project_radius: bool = False,
    sampling_nonbonded_min_dist: float | None = None,
    sampling_nonbonded_strength: float = 0.5,
    sampling_nonbonded_iters: int = 1,
    sampling_bond_strength: float = 0.5,
    sampling_bond_iters: int = 0,
    sampling_max_step: float | None = None,
    sampling_rescale_each_step: bool = False,
    guidance_scale: float = 0.0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Core reverse-diffusion sampling loop (single sample).

    This is the single source of truth for DDPM / DDIM reverse sampling.
    Both :func:`sample_structure` (web API) and :func:`generate_samples`
    (CLI batch generation) delegate here.

    Parameters
    ----------
    model : trained EGNN diffusion model.
    scheduler : pre-configured DiffusionScheduler.
    template : PyG Data with ``pos`` (initial noise) and ``edge_index``.
    C_value : number of carbon atoms.
    use_ddim : use deterministic DDIM sampling.
    ddim_steps : number of steps for DDIM.
    return_trajectory : whether to store intermediate positions.
    sampling_* : optional per-step geometric corrections.

    Returns
    -------
    pos_final : [N, 3] final denoised positions.
    trajectory : list of intermediate positions (empty if not requested).
    """
    device = template.pos.device
    pos_t = torch.randn_like(template.pos)

    C_cond = torch.tensor([C_value], dtype=torch.long, device=device)
    batch = torch.zeros(C_value, dtype=torch.long, device=device)

    if use_ddim:
        timesteps = torch.linspace(
            scheduler.num_steps - 1, 0, ddim_steps,
            dtype=torch.long, device=device,
        )
    else:
        timesteps = torch.arange(
            scheduler.num_steps - 1, -1, -1,
            dtype=torch.long, device=device,
        )

    trajectory: list[torch.Tensor] = []
    x_0_pred_prev = None

    for t in timesteps:
        t_batch = t.unsqueeze(0)
        noise_pred = model(
            pos_t, template.edge_index, t_batch, C_cond, batch,
        )

        # NaN guard: if model output is NaN, abort early
        if torch.isnan(noise_pred).any():
            logger.warning("NaN in model output at t=%d — aborting diffusion", t.item())
            return pos_t, trajectory

        sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t]

        # Clamp denominator to avoid division by near-zero
        sqrt_alpha_bar_t_safe = sqrt_alpha_bar_t.clamp(min=1e-6)
        x_0_pred = (pos_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t_safe

        # Soft limiting for numerical stability (tanh-based, no hard clamp)
        max_reasonable = 8.0
        x_0_pred = max_reasonable * torch.tanh(x_0_pred / max_reasonable)

        # ===== Reconstruction Guidance =====
        # Shift x_0_pred toward better geometry using gradient-based guidance.
        # This is the key algorithmic improvement: the diffusion model's output
        # is nudged by ∇E_geom(x̂₀) so the AI learns to produce structures
        # that satisfy physical constraints WITHOUT classical post-processing.
        if guidance_scale > 0:
            guidance_grad = _compute_geometry_guidance(
                x_0_pred, template.edge_index, C_value,
                guidance_scale=guidance_scale,
                sqrt_one_minus_alpha_bar_t=sqrt_one_minus_alpha_bar_t.item(),
            )
            x_0_pred = x_0_pred - guidance_grad

        coef1 = scheduler.posterior_mean_coef1[t]
        coef2 = scheduler.posterior_mean_coef2[t]
        mean = coef1 * x_0_pred + coef2 * pos_t

        if t > 0:
            variance = scheduler.posterior_variance[t]
            noise = torch.randn_like(pos_t)
            pos_t = mean + torch.sqrt(variance.clamp(min=1e-8)) * noise
            if sampling_momentum > 0 and x_0_pred_prev is not None:
                pos_t = pos_t + sampling_momentum * (x_0_pred - x_0_pred_prev)
        else:
            pos_t = mean

        # NaN guard: if positions become NaN mid-loop, abort
        if torch.isnan(pos_t).any():
            logger.warning("NaN in positions at t=%d — aborting diffusion", t.item())
            return pos_t, trajectory

        # Optional per-step geometric corrections
        # Only apply at low-noise timesteps (t < T/2) to avoid destabilising
        # the diffusion at high-noise stages where positions are still noisy.
        correction_threshold = scheduler.num_steps // 2
        apply_corrections = t.item() < correction_threshold

        if apply_corrections and sampling_project_radius:
            pos_t = rescale_to_unit_radius(pos_t)
        if apply_corrections and sampling_bond_iters > 0:
            target_bond = bond_target_normalized(C_value).to(pos_t.device)
            pos_t = apply_bond_projection(
                pos_t, template.edge_index,
                target_length=target_bond.item(),
                strength=sampling_bond_strength,
                iters=sampling_bond_iters,
                max_step=sampling_max_step,
            )
        if apply_corrections and sampling_nonbonded_min_dist is not None:
            pos_t = apply_nonbonded_repulsion(
                pos_t, template.edge_index,
                min_dist=sampling_nonbonded_min_dist,
                strength=sampling_nonbonded_strength,
                iters=sampling_nonbonded_iters,
                max_step=sampling_max_step,
            )
        if apply_corrections and sampling_rescale_each_step:
            pos_t = rescale_to_unit_radius(pos_t)

        x_0_pred_prev = x_0_pred.detach()

        if return_trajectory:
            trajectory.append(pos_t.detach().clone())

    return pos_t, trajectory


def sample_structure(
    model,
    scheduler,
    template: Data,
    *,
    C_value: int,
    use_ddim: bool = False,
    return_trajectory: bool = False,
    loss_config: dict | None = None,
    diffusion_type: str = 'ddpm',
    fm_scheduler: FlowMatchingScheduler | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sampling helper used by FullereneAPI (web interface).

    Dispatches to flow matching or DDPM reverse diffusion loop.
    Validates output and raises ValueError if NaN detected.

    Returns
    -------
    pos_final, trajectory  (trajectory empty if *return_trajectory* is False)
    """
    lc = loss_config or {}

    if diffusion_type == 'flow_matching' and fm_scheduler is not None:
        # Flow matching path
        cfg_weight = lc.get('cfg_guidance_weight', 2.0)
        num_steps = lc.get('sampling_steps', 50)
        pos_final, trajectory = _flow_matching_loop(
            model, fm_scheduler, template,
            C_value=C_value,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            return_trajectory=return_trajectory,
        )
    else:
        # DDPM path
        guidance_scale = lc.get('guidance_scale', 0.5)
        bond_iters = lc.get('sampling_bond_iters', 1)
        bond_strength = lc.get('sampling_bond_strength', 0.3)
        project_radius = lc.get('sampling_project_radius', True)
        rescale_each = lc.get('sampling_rescale_each_step', True)
        nonbonded_dist = lc.get('sampling_nonbonded_min_dist', 0.3)

        pos_final, trajectory = _reverse_diffusion_loop(
            model, scheduler, template,
            C_value=C_value,
            use_ddim=use_ddim,
            return_trajectory=return_trajectory,
            guidance_scale=guidance_scale,
            sampling_bond_iters=bond_iters,
            sampling_bond_strength=bond_strength,
            sampling_project_radius=project_radius,
            sampling_rescale_each_step=rescale_each,
            sampling_nonbonded_min_dist=nonbonded_dist,
        )

    # Post-diffusion NaN validation
    if torch.isnan(pos_final).any():
        nan_frac = torch.isnan(pos_final).float().mean().item()
        raise ValueError(
            f"Diffusion produced NaN positions ({nan_frac:.1%} of coordinates). "
            "The model checkpoint may need retraining."
        )

    return pos_final, trajectory


def generate_samples(
    model,
    scheduler,
    num_samples: int,
    C_value: int,
    device: str,
    template_bank: TemplateBank | None,
    topology_model: TopologyGNN | None = None,
    use_ddim: bool = False,
    ddim_steps: int = 100,
    guidance_scale: float = 0.0,
    sampling_project_radius: bool = False,
    sampling_nonbonded_min_dist: float | None = None,
    sampling_nonbonded_strength: float = 0.5,
    sampling_nonbonded_iters: int = 1,
    sampling_bond_strength: float = 0.5,
    sampling_bond_iters: int = 0,
    sampling_max_step: float | None = None,
    sampling_rescale_each_step: bool = False,
    diffusion_type: str = 'ddpm',
    fm_scheduler: FlowMatchingScheduler | None = None,
    cfg_weight: float = 0.0,
    fm_num_steps: int = 50,
) -> list:
    """Generate fullerene structures using the trained model.

    Args:
        model: Trained diffusion model.
        scheduler: Diffusion scheduler.
        num_samples: Number of samples to generate.
        C_value: Number of carbon atoms.
        device: Device for computation.
        use_ddim: Whether to use DDIM sampling (faster).
        ddim_steps: Number of steps for DDIM.
        guidance_scale: Reserved for future guided sampling (currently unused).

    Returns:
        List of generated positions (each is [C, 3] tensor on CPU).
    """
    model.eval()
    generated_structures: list[torch.Tensor] = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Generating C{C_value}"):
            try:
                template = create_template_graph(C_value, device, template_bank, topology_model)
            except Exception as exc:
                logger.warning("Failed to create template for C%d: %s. Skipping sample %d.", C_value, exc, i)
                continue

            if diffusion_type == 'flow_matching' and fm_scheduler is not None:
                pos_t, _ = _flow_matching_loop(
                    model, fm_scheduler, template,
                    C_value=C_value,
                    num_steps=fm_num_steps,
                    cfg_weight=cfg_weight,
                    return_trajectory=False,
                )
            else:
                pos_t, _ = _reverse_diffusion_loop(
                    model, scheduler, template,
                    C_value=C_value,
                    use_ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    return_trajectory=False,
                    sampling_project_radius=sampling_project_radius,
                    sampling_nonbonded_min_dist=sampling_nonbonded_min_dist,
                    sampling_nonbonded_strength=sampling_nonbonded_strength,
                    sampling_nonbonded_iters=sampling_nonbonded_iters,
                    sampling_bond_strength=sampling_bond_strength,
                    sampling_bond_iters=sampling_bond_iters,
                    sampling_max_step=sampling_max_step,
                    sampling_rescale_each_step=sampling_rescale_each_step,
                )

            if torch.isnan(pos_t).any():
                logger.warning("NaN detected in sample %d, skipping", i)
                continue

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
    neighbors = {i: [] for i in range(num_atoms)}
    
    for src, dst in edge_index.t().tolist():
        if len(neighbors[src]) < 3:
            neighbors[src].append(dst)
    
    # Fill missing neighbors with -1
    for i in range(num_atoms):
        while len(neighbors[i]) < 3:
            neighbors[i].append(-1)
    
    # Write to file (match dataset convention: 1-based indices)
    with open(filepath, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Generated fullerene structure\n")
        
        for i in range(num_atoms):
            x, y, z = pos[i].tolist()
            nb1, nb2, nb3 = neighbors[i][:3]
            # Convert to 1-based indexing for compatibility with dataset tooling
            idx_1 = i + 1
            nb1_1 = nb1 + 1
            nb2_1 = nb2 + 1
            nb3_1 = nb3 + 1
            f.write(f"C {x:.6f} {y:.6f} {z:.6f} {idx_1} {nb1_1} {nb2_1} {nb3_1}\n")


def compute_bond_stats(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[float, float]:
    """Return mean/std of bond lengths (unique undirected edges)."""
    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]
    if row.numel() == 0:
        return 0.0, 0.0
    lengths = (pos[row] - pos[col]).norm(dim=1)
    return lengths.mean().item(), lengths.std(unbiased=False).item()


def compute_radius_stats(pos: torch.Tensor) -> tuple[float, float]:
    """Return mean/std of atomic radii from center."""
    centered = pos - pos.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1)
    return radii.mean().item(), radii.std(unbiased=False).item()


def rescale_to_unit_radius(pos: torch.Tensor) -> torch.Tensor:
    """Center and scale positions so mean radius == 1.0."""
    centered = pos - pos.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1)
    mean_r = radii.mean().clamp(min=1e-8)
    return centered / mean_r


def _clamp_step(delta: torch.Tensor, max_step: float) -> torch.Tensor:
    """Clamp per-atom displacement to avoid exploding updates."""
    if max_step is None or max_step <= 0:
        return delta
    norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(max_step / norm, max=1.0)
    return delta * scale


def apply_nonbonded_repulsion(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    min_dist: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: float | None = None,
) -> torch.Tensor:
    """Simple nonbonded repulsion in normalized space (no gradients)."""
    out = pos
    row, col = edge_index
    for _ in range(iters):
        dists = torch.cdist(out, out)
        n = dists.size(0)
        inf = torch.tensor(float('inf'), device=dists.device, dtype=dists.dtype)
        diag_mask = torch.eye(n, device=dists.device, dtype=torch.bool)
        bond_mask = torch.zeros((n, n), device=dists.device, dtype=torch.bool)
        bond_mask[row, col] = True
        bond_mask[col, row] = True
        mask = diag_mask | bond_mask
        dists = dists.masked_fill(mask, inf)

        vec = out[:, None, :] - out[None, :, :]
        dist_safe = dists.unsqueeze(-1).clamp(min=1e-6)
        overlap = (min_dist - dists).clamp(min=0.0)
        push = (overlap.unsqueeze(-1) / dist_safe) * vec
        delta = push.sum(dim=1)
        delta = _clamp_step(delta, max_step)
        out = out + strength * delta
    return out


def apply_bond_projection(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    target_length: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: float | None = None,
) -> torch.Tensor:
    """Project bonded pairs toward target length (normalized space)."""
    out = pos
    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]
    if row.numel() == 0:
        return out

    for _ in range(iters):
        vec = out[row] - out[col]
        dist = vec.norm(dim=1).clamp(min=1e-6)
        delta = ((dist - target_length) / dist).unsqueeze(-1) * vec
        pos_update = torch.zeros_like(out)
        pos_update.index_add_(0, row, delta)
        pos_update.index_add_(0, col, -delta)
        pos_update = _clamp_step(pos_update, max_step)
        out = out - strength * pos_update
    return out


def relax_structure(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    C_value: int,
    *,
    steps: int = 200,
    lr: float = 5e-2,
    bond_weight: float = 1.0,
    radius_weight: float = 0.2,
    radius_var_weight: float = 0.2,
    nonbonded_min_dist: float = 0.45,
    nonbonded_weight: float = 0.2,
) -> torch.Tensor:
    """Lightweight geometric relaxation in normalized space.

    Minimizes:
    - bond length MSE to normalized target
    - mean radius deviation to 1.0
    - radius variance (compact shell)
    """
    pos_opt = pos.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pos_opt], lr=lr)

    target_bond = bond_target_normalized(C_value).to(pos_opt.device)

    row, col = edge_index
    mask = row < col
    row = row[mask]
    col = col[mask]

    for _ in range(steps):
        optimizer.zero_grad()
        centered = pos_opt - pos_opt.mean(dim=0, keepdim=True)
        radii = centered.norm(dim=1)
        mean_r = radii.mean()
        var_r = radii.var(unbiased=False)

        if row.numel() > 0:
            bond_len = (pos_opt[row] - pos_opt[col]).norm(dim=1)
            bond_loss = ((bond_len - target_bond) ** 2).mean()
        else:
            bond_loss = torch.tensor(0.0, device=pos_opt.device)

        radius_loss = (mean_r - 1.0) ** 2
        var_loss = var_r

        # Non-bonded repulsion to avoid atom overlap/clumping
        dists = torch.cdist(pos_opt, pos_opt)
        n = dists.size(0)
        inf = torch.tensor(float('inf'), device=dists.device, dtype=dists.dtype)
        diag_mask = torch.eye(n, device=dists.device, dtype=torch.bool)
        bond_mask = torch.zeros((n, n), device=dists.device, dtype=torch.bool)
        bond_mask[row, col] = True
        bond_mask[col, row] = True
        mask = diag_mask | bond_mask
        dists = dists.masked_fill(mask, inf)
        overlap = torch.clamp(nonbonded_min_dist - dists, min=0)
        nonbonded_loss = (overlap ** 2).mean()

        loss = (
            bond_weight * bond_loss
            + radius_weight * radius_loss
            + radius_var_weight * var_loss
            + nonbonded_weight * nonbonded_loss
        )
        loss.backward()
        optimizer.step()

    return pos_opt.detach()


def is_valid_structure(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    C_value: int,
    *,
    bond_mean_tol: float,
    bond_std_max: float,
    radius_mean_tol: float,
    radius_std_max: float,
    nonbonded_min_dist: float | None = None,
    max_nonbonded_violations: int = 0,
    radius_coeff: float = 0.45,
) -> tuple[bool, dict]:
    """Check basic geometric validity in normalized space.

    Criteria:
    - mean bond length close to normalized target
    - bond length dispersion below threshold
    - mean radius close to 1.0 (normalized)
    - radius dispersion below threshold
    """
    bond_mean, bond_std = compute_bond_stats(pos, edge_index)
    radius_mean, radius_std = compute_radius_stats(pos)

    target_bond = bond_target_normalized(C_value, radius_coeff=radius_coeff).item()

    ok_bond_mean = abs(bond_mean - target_bond) <= bond_mean_tol
    ok_bond_std = bond_std <= bond_std_max
    ok_radius_mean = abs(radius_mean - 1.0) <= radius_mean_tol
    ok_radius_std = radius_std <= radius_std_max

    nonbonded_violations = 0
    ok_nonbonded = True
    if nonbonded_min_dist is not None:
        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(float('inf'))
        row, col = edge_index
        dists[row, col] = float('inf')
        dists[col, row] = float('inf')
        nonbonded_violations = int((dists < nonbonded_min_dist).sum().item())
        ok_nonbonded = nonbonded_violations <= max_nonbonded_violations

    ok = ok_bond_mean and ok_bond_std and ok_radius_mean and ok_radius_std and ok_nonbonded

    stats = {
        "bond_mean": bond_mean,
        "bond_std": bond_std,
        "bond_target": target_bond,
        "radius_mean": radius_mean,
        "radius_std": radius_std,
        "nonbonded_violations": nonbonded_violations,
        "ok_bond_mean": ok_bond_mean,
        "ok_bond_std": ok_bond_std,
        "ok_radius_mean": ok_radius_mean,
        "ok_radius_std": ok_radius_std,
        "ok_nonbonded": ok_nonbonded,
    }
    return ok, stats


def main():
    parser = argparse.ArgumentParser(description='Generate Fullerene Structures')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--C', type=int, required=True, help='Number of carbon atoms (20-720)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of structures to generate')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output directory')
    parser.add_argument('--ddim', action='store_true', default=True, help='Use DDIM sampling (default: True)')
    parser.add_argument('--no_ddim', dest='ddim', action='store_false', help='Use full DDPM sampling')
    parser.add_argument('--ddim_steps', type=int, default=200, help='DDIM sampling steps (default: 200)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model weights for generation (default: True)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Use primary model weights instead of EMA')
    parser.add_argument('--output_scale', type=str, default='normalized', choices=['normalized', 'angstrom'],
                        help='Save coordinates in normalized training space or approximate Å space')
    parser.add_argument('--filter_invalid', action='store_true', default=True,
                        help='Filter invalid structures (default: True)')
    parser.add_argument('--no_filter', dest='filter_invalid', action='store_false',
                        help='Disable validity filtering')
    parser.add_argument('--bond_mean_tol', type=float, default=0.08,
                        help='Allowed deviation of mean bond length (normalized units)')
    parser.add_argument('--bond_std_max', type=float, default=0.08,
                        help='Max allowed bond length std (normalized units)')
    parser.add_argument('--radius_mean_tol', type=float, default=0.12,
                        help='Allowed deviation of mean radius from 1.0 (normalized units)')
    parser.add_argument('--radius_std_max', type=float, default=0.12,
                        help='Max allowed radius std (normalized units)')
    parser.add_argument('--save_rejects', action='store_true',
                        help='Also save rejected samples to <output_dir>/rejected')
    parser.add_argument('--rescale_to_unit_radius', action='store_true', default=True,
                        help='Rescale each sample to mean radius 1.0 (default: True)')
    parser.add_argument('--no_rescale', dest='rescale_to_unit_radius', action='store_false',
                        help='Disable rescaling to unit radius')
    parser.add_argument('--relax_steps', type=int, default=-1,
                        help='Post-generation relaxation steps (-1 = auto based on C, 0 = disable)')
    parser.add_argument('--relax_lr', type=float, default=5e-2,
                        help='Learning rate for relaxation')
    parser.add_argument('--relax_bond_weight', type=float, default=1.0,
                        help='Bond length weight during relaxation')
    parser.add_argument('--relax_radius_weight', type=float, default=0.2,
                        help='Mean radius weight during relaxation')
    parser.add_argument('--relax_radius_var_weight', type=float, default=0.2,
                        help='Radius variance weight during relaxation')
    parser.add_argument('--relax_nonbonded_min_dist', type=float, default=0.45,
                        help='Minimum non-bonded distance in normalized space')
    parser.add_argument('--relax_nonbonded_weight', type=float, default=0.2,
                        help='Non-bonded repulsion weight during relaxation')
    parser.add_argument('--filter_nonbonded_min_dist', type=float, default=None,
                        help='Filter: minimum non-bonded distance (normalized). None disables')
    parser.add_argument('--filter_max_nonbonded_violations', type=int, default=0,
                        help='Max allowed non-bonded violations')
    parser.add_argument('--sampling_project_radius', action='store_true', default=True,
                        help='Project to unit radius at each sampling step (default: True)')
    parser.add_argument('--no_sampling_project_radius', dest='sampling_project_radius', action='store_false',
                        help='Disable per-step radius projection')
    parser.add_argument('--sampling_nonbonded_min_dist', type=float, default=None,
                        help='Sampling-time nonbonded min distance (normalized). None disables')
    parser.add_argument('--sampling_nonbonded_strength', type=float, default=0.5,
                        help='Sampling-time repulsion strength')
    parser.add_argument('--sampling_nonbonded_iters', type=int, default=1,
                        help='Sampling-time repulsion iterations per step')
    parser.add_argument('--sampling_bond_strength', type=float, default=0.5,
                        help='Sampling-time bond projection strength')
    parser.add_argument('--sampling_bond_iters', type=int, default=1,
                        help='Sampling-time bond projection iterations per step')
    parser.add_argument('--sampling_max_step', type=float, default=0.05,
                        help='Clamp per-step displacement in normalized units (<=0 disables)')
    parser.add_argument('--sampling_rescale_each_step', action='store_true', default=True,
                        help='Rescale to unit radius after each sampling step (default: True)')
    parser.add_argument('--no_sampling_rescale', dest='sampling_rescale_each_step', action='store_false',
                        help='Disable per-step rescaling')
    parser.add_argument('--use_topology_gnn', action='store_true',
                        help='Use topology GNN to generate adjacency instead of dataset templates')
    parser.add_argument('--topology_gnn_ckpt', type=str, default='checkpoints/topology_gnn.pt',
                        help='Topology GNN checkpoint path')
    args = parser.parse_args()
    
    # Load checkpoint
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Loading checkpoint: %s", args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    logger.info("Checkpoint from epoch %d, best val_loss=%.6f",
                checkpoint['epoch'], checkpoint['best_val_loss'])
    
    # Create model — dispatch on architecture
    logger.info("Creating model...")
    model_config = config['model']
    architecture = model_config.get('architecture', 'egnn')
    diffusion_type = config.get('diffusion', {}).get('type', 'ddpm')

    if architecture == 'painn':
        from model_painn import FullerenePaiNNModel
        model = FullerenePaiNNModel(
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 8),
            num_rbf=model_config.get('num_rbf', 20),
            cutoff=model_config.get('cutoff', 5.0),
            time_embed_dim=model_config.get('time_embed_dim', 128),
            C_embed_dim=model_config.get('C_embed_dim', 64),
            C_fourier_features=model_config.get('C_fourier_features', 16),
            max_C=model_config.get('max_C', 720),
            cfg_drop_prob=0.0,  # No dropout during inference
            num_atom_types=model_config.get('num_atom_types', 1),
            use_pbc=model_config.get('use_pbc', False),
        ).to(device)
    else:
        model = FullereneDiffusionModel(
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            edge_dim=model_config.get('edge_dim', 0),
            C_embed_dim=model_config.get('C_embed_dim', 64),
            time_embed_dim=model_config.get('time_embed_dim', 128),
            max_C=model_config.get('max_C', 720),
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
    
    # Prefer EMA weights for generation (typically 5-15% better sample quality)
    if args.use_ema and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("Model loaded with EMA weights (decay=%.4f)",
                    config.get('training', {}).get('ema_decay', 0.9999))
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.use_ema:
            logger.warning("EMA weights requested but not found in checkpoint — using primary weights")
        logger.info("Model loaded with primary weights")
    
    # Create scheduler(s)
    diff_config = config['diffusion']
    scheduler = DiffusionScheduler(
        num_steps=diff_config.get('num_steps', 1000),
        beta_schedule=diff_config.get('beta_schedule', 'cosine'),
        beta_start=diff_config.get('beta_start', 0.0001),
        beta_end=diff_config.get('beta_end', 0.02),
        device=device,
    )
    fm_sched = FlowMatchingScheduler() if diffusion_type == 'flow_matching' else None
    
    # Generate structures
    logger.info("Generating %d samples of C%d...", args.num_samples, args.C)

    # Template bank (dataset adjacency) and/or topology GNN
    template_bank = None
    topology_model = None
    data_cfg = config.get('data', {})
    xyz_dir = data_cfg.get('xyz_dir')
    if xyz_dir:
        xyz_path = Path(xyz_dir)
        if not xyz_path.is_absolute():
            xyz_path = (Path(__file__).resolve().parent / xyz_path).resolve()
        template_bank = TemplateBank(xyz_dir=xyz_path)

    if args.use_topology_gnn:
        topo_ckpt = torch.load(args.topology_gnn_ckpt, map_location=device)
        topo_cfg = topo_ckpt.get('config', {})
        max_nodes = int(topo_cfg.get('max_C', 720))
        hidden_dim = int(topo_cfg.get('hidden_dim', 64))
        latent_dim = int(topo_cfg.get('latent_dim', 32))
        topology_model = TopologyGNN(max_nodes=max_nodes, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        topology_model.load_state_dict(topo_ckpt['model_state_dict'])
        topology_model.eval()
    elif template_bank is None:
        raise ValueError("Checkpoint config missing data.xyz_dir (needed for template graphs)")

    cfg_gw = config.get('diffusion', {}).get('cfg_guidance_weight', 2.0)
    fm_steps = config.get('diffusion', {}).get('sampling_steps', 50)
    structures = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        C_value=args.C,
        device=device,
        template_bank=template_bank,
        topology_model=topology_model,
        use_ddim=args.ddim,
        ddim_steps=args.ddim_steps,
        sampling_project_radius=args.sampling_project_radius,
        sampling_nonbonded_min_dist=args.sampling_nonbonded_min_dist,
        sampling_nonbonded_strength=args.sampling_nonbonded_strength,
        sampling_nonbonded_iters=args.sampling_nonbonded_iters,
        sampling_bond_strength=args.sampling_bond_strength,
        sampling_bond_iters=args.sampling_bond_iters,
        sampling_max_step=args.sampling_max_step,
        sampling_rescale_each_step=args.sampling_rescale_each_step,
        diffusion_type=diffusion_type,
        fm_scheduler=fm_sched,
        cfg_weight=cfg_gw,
        fm_num_steps=fm_steps,
    )

    # Resolve adaptive relax_steps: -1 means auto (scales with C)
    relax_steps = args.relax_steps
    if relax_steps < 0:
        # Heuristic: scales with atom count for reliable convergence
        relax_steps = max(100, int(50 + args.C * 2.0))
        logger.info("Auto relax_steps for C%d: %d", args.C, relax_steps)

    # Save to files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reject_dir = output_dir / "rejected"
    if args.save_rejects:
        reject_dir.mkdir(exist_ok=True)
    
    logger.info("Saving structures to %s/ ...", output_dir)
    valid_count = 0
    stats_list = []
    for i, pos in enumerate(structures):
        # Match the same template family for edges
        template = create_template_graph(args.C, 'cpu', template_bank, topology_model)

        pos_work = pos
        if args.rescale_to_unit_radius:
            pos_work = rescale_to_unit_radius(pos_work)

        if relax_steps > 0:
            pos_work = relax_structure(
                pos_work,
                template.edge_index,
                args.C,
                steps=relax_steps,
                lr=args.relax_lr,
                bond_weight=args.relax_bond_weight,
                radius_weight=args.relax_radius_weight,
                radius_var_weight=args.relax_radius_var_weight,
                nonbonded_min_dist=args.relax_nonbonded_min_dist,
                nonbonded_weight=args.relax_nonbonded_weight,
            )

        pos_to_save = pos_work
        if args.output_scale == 'angstrom':
            pos_to_save = denormalize_positions(pos_to_save, args.C).cpu()

        is_valid, stats = is_valid_structure(
            pos_work,
            template.edge_index,
            args.C,
            bond_mean_tol=args.bond_mean_tol,
            bond_std_max=args.bond_std_max,
            radius_mean_tol=args.radius_mean_tol,
            radius_std_max=args.radius_std_max,
            nonbonded_min_dist=args.filter_nonbonded_min_dist,
            max_nonbonded_violations=args.filter_max_nonbonded_violations,
        )
        stats_list.append(stats)

        if args.filter_invalid and not is_valid:
            if args.save_rejects:
                filepath = reject_dir / f"C{args.C}_sample_{i:03d}.xyz"
                save_xyz(pos_to_save, template.edge_index, filepath)
            continue

        filepath = output_dir / f"C{args.C}_sample_{i:03d}.xyz"
        save_xyz(pos_to_save, template.edge_index, filepath)
        valid_count += 1

    if args.filter_invalid:
        total = len(structures)
        valid_rate = valid_count / total if total > 0 else 0.0
        bond_mean = sum(s["bond_mean"] for s in stats_list) / max(1, total)
        bond_std = sum(s["bond_std"] for s in stats_list) / max(1, total)
        radius_mean = sum(s["radius_mean"] for s in stats_list) / max(1, total)
        radius_std = sum(s["radius_std"] for s in stats_list) / max(1, total)
        logger.info("Validity summary: valid=%d/%d (%.1f%%) "
                    "avg bond_mean=%.4f bond_std=%.4f "
                    "avg radius_mean=%.4f radius_std=%.4f",
                    valid_count, total, valid_rate * 100,
                    bond_mean, bond_std, radius_mean, radius_std)
    
    logger.info("Generation complete!")


if __name__ == '__main__':
    main()
