"""Lightweight validator for extended-XYZ fullerene files.

Designed to work without PyTorch.

Checks:
- 3-valency (degree==3) from neighbor lists
- connectivity (single connected component)
- bond length statistics (using provided edges)
- sphericity-ish metrics (gyration tensor eigenvalues)

Usage:
  python validate_xyz.py path/to/file.xyz
  python validate_xyz.py path/to/folder --glob "*.xyz"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def parse_extended_xyz(filepath: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    lines = filepath.read_text().splitlines()
    n = int(lines[0].strip())

    positions: list[list[float]] = []
    raw_indices: list[int] = []
    raw_neighbors: list[list[int]] = []

    for i in range(2, 2 + n):
        parts = lines[i].split()
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(parts) >= 8:
            raw_indices.append(int(parts[4]))
            raw_neighbors.append([int(parts[5]), int(parts[6]), int(parts[7])])

    # Detect 1-based indexing
    one_based = False
    if raw_indices:
        if min(raw_indices) == 1 and max(raw_indices) == n:
            one_based = True

    edges: list[Tuple[int, int]] = []
    for idx, nbs in zip(raw_indices, raw_neighbors):
        idx0 = idx - 1 if one_based else idx
        for nb in nbs:
            nb0 = nb - 1 if one_based else nb
            if 0 <= nb0 < n and nb0 != idx0:
                edges.append((idx0, nb0))

    return np.asarray(positions, dtype=np.float64), edges


def degree_stats(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
    return deg


def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if n == 0:
        return True
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    seen = set([0])
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def bond_lengths(positions: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    seen = set()
    bl = []
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        bl.append(np.linalg.norm(positions[u] - positions[v]))
    return np.asarray(bl, dtype=np.float64)


def gyration_eigvals(positions: np.ndarray) -> np.ndarray:
    centered = positions - positions.mean(axis=0, keepdims=True)
    S = centered.T @ centered / len(positions)
    eigvals = np.linalg.eigvalsh(S)
    return np.sort(eigvals)[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str)
    ap.add_argument("--glob", type=str, default="*.xyz")
    args = ap.parse_args()

    path = Path(args.path)
    files = [path] if path.is_file() else sorted(path.glob(args.glob))
    if not files:
        raise SystemExit("No files matched")

    for fp in files:
        pos, edges = parse_extended_xyz(fp)
        n = len(pos)
        deg = degree_stats(n, edges)
        bl = bond_lengths(pos, edges)
        eig = gyration_eigvals(pos)
        r_mean = np.mean(np.linalg.norm(pos - pos.mean(axis=0, keepdims=True), axis=1))

        ok_deg = np.all(deg == 3)
        ok_conn = is_connected(n, edges)

        logger.info("%s: N=%d connected=%s deg3=%s "
              "bond_mean=%.4f bond_std=%.4f mean_radius=%.4f "
              "eig=%.4f,%.4f,%.4f",
              fp.name, n, ok_conn, ok_deg,
              bl.mean(), bl.std(), r_mean,
              eig[0], eig[1], eig[2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    main()
