"""Synthetic topology generation for training.

Generates 3-regular planar graphs for C>=20 and embeds them into 3D with
simple geometric refinement. Intended as a fallback/augmentation when
no dataset templates exist for a given C.

Two generation strategies:
  1. ``generate_planar_3_regular_graph`` — random search (fast for small C,
     unreliable for C > 100).
  2. ``generate_fullerene_topology`` — convex-hull dual construction.
     Deterministic and works for *any* even C >= 20.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

from units import bond_target_normalized


# ---------------------------------------------------------------------------
# Strategy 1: random 3-regular + planarity check (small C)
# ---------------------------------------------------------------------------


def generate_planar_3_regular_graph(
    C: int,
    *,
    max_tries: int = 200,
    seed: Optional[int] = None,
) -> Optional[nx.Graph]:
    """Generate a connected 3-regular planar graph for C>=20.

    Returns None if no graph is found within max_tries.
    """
    if C < 20 or C % 2 != 0:
        return None

    rng = random.Random(seed)
    for _ in range(max_tries):
        G = nx.random_regular_graph(3, C, seed=rng.randint(0, 2**32 - 1))
        if not nx.is_connected(G):
            continue
        is_planar, _ = nx.check_planarity(G)
        if is_planar:
            return G
    return None


# ---------------------------------------------------------------------------
# Strategy 2: convex-hull dual (works for all C >= 20)
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int, jitter: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Place *n* points approximately uniformly on a unit sphere.

    Uses the Fibonacci spiral (golden-ratio method).  An optional *jitter*
    adds small random perturbations to break exact symmetry (useful to avoid
    degenerate convex-hull faces).
    """
    rng = np.random.RandomState(seed)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(n, dtype=np.float64)

    theta = 2.0 * np.pi * indices / golden_ratio  # azimuthal angle
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)  # polar angle

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.stack([x, y, z], axis=-1)  # (n, 3)

    if jitter > 0:
        pts += rng.randn(n, 3) * jitter
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= np.clip(norms, 1e-8, None)  # project back to sphere

    return pts.astype(np.float64)


def generate_fullerene_topology(
    C: int,
    *,
    seed: Optional[int] = None,
    max_retries: int = 10,
) -> Optional[nx.Graph]:
    """Generate a 3-regular planar graph with *C* vertices via convex-hull dual.

    Mathematical basis
    ------------------
    Place  N = C/2 + 2  points on a unit sphere.  Their convex hull is a
    triangulation with V = N, E = 3N − 6, F = 2N − 4.

    The *dual* graph (one vertex per triangle-face, edges between face-pairs
    sharing an original edge) has:
        V' = F = 2N − 4 = 2(C/2 + 2) − 4 = C   vertices,
        E' = E = 3N − 6 = 3(C/2 + 2) − 6 ≈ 3C/2  edges.
    Every dual vertex has degree 3 (each triangular face has 3 edges), so
    the dual is **3-regular and planar** by construction.

    Returns ``None`` only if SciPy/ConvexHull fails after *max_retries*.
    """
    if C < 20 or C % 2 != 0:
        return None

    from scipy.spatial import ConvexHull  # type: ignore

    N = C // 2 + 2  # points on sphere

    rng_seed = seed if seed is not None else 42
    for attempt in range(max_retries):
        pts = _fibonacci_sphere(N, jitter=0.02, seed=rng_seed + attempt)

        try:
            hull = ConvexHull(pts)
        except Exception:
            continue

        simplices = hull.simplices  # (F, 3) — triangle vertex indices

        # Verify face count matches expectation
        if simplices.shape[0] != C:
            # Degenerate hull (co-planar/co-spherical points) — retry with
            # more jitter.
            continue

        # Build dual graph: one node per face, edge between faces sharing an
        # original edge.
        edge_to_faces = {}  # type: ignore
        for fi, tri in enumerate(simplices):
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
                key = (min(a, b), max(a, b))
                edge_to_faces.setdefault(key, []).append(fi)

        G = nx.Graph()
        G.add_nodes_from(range(C))
        for faces in edge_to_faces.values():
            if len(faces) == 2:
                G.add_edge(faces[0], faces[1])

        # Verify 3-regularity and connectivity
        degrees = [d for _, d in G.degree()]
        if all(d == 3 for d in degrees) and nx.is_connected(G):
            return G

    return None


def _graph_to_edge_index(G: nx.Graph) -> torch.Tensor:
    edges: List[List[int]] = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _spring_layout_3d(G: nx.Graph, seed: Optional[int] = None) -> np.ndarray:
    pos = nx.spring_layout(G, seed=seed, dim=3)
    coords = np.array([pos[i] for i in range(G.number_of_nodes())], dtype=np.float32)
    coords -= coords.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(coords, axis=1)
    mean_r = float(np.mean(radii)) if radii.size else 1.0
    if mean_r > 1e-6:
        coords /= mean_r
    return coords


def _clamp_step(delta: torch.Tensor, max_step: Optional[float]) -> torch.Tensor:
    if max_step is None or max_step <= 0:
        return delta
    norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
    scale = torch.clamp(max_step / norm, max=1.0)
    return delta * scale


def _apply_bond_projection(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    target_length: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: Optional[float] = None,
) -> torch.Tensor:
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


def _apply_nonbonded_repulsion(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    min_dist: float,
    strength: float = 0.5,
    iters: int = 1,
    max_step: Optional[float] = None,
) -> torch.Tensor:
    out = pos
    row, col = edge_index
    for _ in range(iters):
        dists = torch.cdist(out, out)
        n = dists.size(0)
        inf = torch.tensor(float("inf"), device=dists.device, dtype=dists.dtype)
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


def refine_positions(
    coords: np.ndarray,
    edge_index: torch.Tensor,
    C: int,
    *,
    bond_strength: float = 0.6,
    bond_iters: int = 5,
    nonbonded_min_dist: float = 0.45,
    nonbonded_strength: float = 0.6,
    nonbonded_iters: int = 3,
    max_step: Optional[float] = 0.05,
) -> np.ndarray:
    pos = torch.tensor(coords, dtype=torch.float32)
    target_bond = bond_target_normalized(C).to(pos.device)
    pos = _apply_bond_projection(
        pos,
        edge_index,
        target_length=float(target_bond.item()),
        strength=bond_strength,
        iters=bond_iters,
        max_step=max_step,
    )
    pos = _apply_nonbonded_repulsion(
        pos,
        edge_index,
        min_dist=nonbonded_min_dist,
        strength=nonbonded_strength,
        iters=nonbonded_iters,
        max_step=max_step,
    )
    centered = pos - pos.mean(dim=0, keepdim=True)
    radii = centered.norm(dim=1)
    mean_r = radii.mean().clamp(min=1e-6)
    pos = centered / mean_r
    return pos.detach().cpu().numpy()


@dataclass
class SyntheticConfig:
    C_values: Iterable[int]
    per_C: int = 1
    max_tries: int = 200
    seed: int = 20260131
    refine: bool = True


def build_synthetic_dataset(cfg: SyntheticConfig) -> List[Data]:
    rng = random.Random(cfg.seed)
    out: List[Data] = []
    for C in cfg.C_values:
        for _ in range(cfg.per_C):
            G = generate_planar_3_regular_graph(C, max_tries=cfg.max_tries, seed=rng.randint(0, 2**32 - 1))
            if G is None:
                continue
            coords = _spring_layout_3d(G, seed=rng.randint(0, 2**32 - 1))
            edge_index = _graph_to_edge_index(G)
            if cfg.refine:
                coords = refine_positions(coords, edge_index, C)
            data = Data(
                pos=torch.tensor(coords, dtype=torch.float32),
                edge_index=edge_index,
                C=torch.tensor([C], dtype=torch.long),
                num_nodes=C,
                synthetic=True,
            )
            out.append(data)
    return out
