#!/usr/bin/env python3
"""Analyze the fullerene_xyz dataset (extended XYZ with explicit 3-neighbor connectivity).

Outputs:
- Per-structure CSV with geometry + graph sanity checks
- Per-C aggregated CSV
- Plots (English labels, Arial font)
- HTML report (English, CSS uses Arial)

The dataset lives in: dataset/fullerenes/fullerene_xyz
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _ensure_matplotlib_arial() -> None:
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class ParsedXYZ:
    n: int
    name: str
    elements: List[str]
    xyz: np.ndarray  # (n, 3)
    neighbors: List[Tuple[int, int, int]]  # 0-based indices, length n


def parse_extended_xyz(path: Path) -> ParsedXYZ:
    """Parse the dataset's extended XYZ format.

    Expected per-atom line format (8 columns):
        Element  x  y  z  index  nb1  nb2  nb3

    Index and neighbor ids are 1-based in the file.
    """

    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"Empty file: {path}")
        n = int(first.strip())
        name = f.readline().strip()

        elements: List[str] = []
        coords = np.zeros((n, 3), dtype=float)
        neighbors: List[Tuple[int, int, int]] = []

        for i in range(n):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in {path} at atom line {i+1}/{n}")
            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"Bad atom line (expected >=8 columns) in {path}: {line!r}")

            el = parts[0]
            x, y, z = map(float, parts[1:4])
            idx = int(parts[4])
            nb1, nb2, nb3 = map(int, parts[5:8])

            # Consistency check: index should match line order (1-based)
            if idx != i + 1:
                # Some files could differ; keep parsing but signal via exception for now.
                raise ValueError(f"Index mismatch in {path}: line {i+1} has idx={idx}")

            elements.append(el)
            coords[i] = [x, y, z]
            neighbors.append((nb1 - 1, nb2 - 1, nb3 - 1))

    return ParsedXYZ(n=n, name=name, elements=elements, xyz=coords, neighbors=neighbors)


def build_edges(neighbors: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    """Return undirected edges as sorted pairs (u, v), unique."""

    edges = set()
    for u, (a, b, c) in enumerate(neighbors):
        for v in (a, b, c):
            if v == u:
                continue
            if v < 0:
                continue
            edges.add((u, v) if u < v else (v, u))
    return sorted(edges)


def connected_components(n: int, edges: List[Tuple[int, int]]) -> int:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    seen = [False] * n
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def degree_stats(n: int, edges: List[Tuple[int, int]]) -> Tuple[int, int, float]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return min(deg), max(deg), float(sum(deg) / n)


def bond_lengths(xyz: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.array([], dtype=float)
    arr = np.empty((len(edges),), dtype=float)
    for i, (u, v) in enumerate(edges):
        d = xyz[u] - xyz[v]
        arr[i] = float(np.linalg.norm(d))
    return arr


def gyration_metrics(xyz: np.ndarray) -> Dict[str, float]:
    """Compute radius of gyration and asphericity from the gyration tensor.

    Uses common polymer/cluster shape metrics.
    """

    centered = xyz - xyz.mean(axis=0, keepdims=True)
    s = (centered.T @ centered) / xyz.shape[0]
    evals = np.linalg.eigvalsh(s)
    evals = np.maximum(evals, 0.0)
    lam1, lam2, lam3 = map(float, sorted(evals, reverse=True))
    rg2 = float(evals.sum())
    rg = float(math.sqrt(rg2))

    # Asphericity b in [0, 1] for 3D: b = ( (λ1-λ2)^2 + (λ2-λ3)^2 + (λ3-λ1)^2 ) / (2 (λ1+λ2+λ3)^2 )
    denom = 2.0 * (rg2 ** 2) if rg2 > 0 else float("nan")
    b = (
        (lam1 - lam2) ** 2 + (lam2 - lam3) ** 2 + (lam3 - lam1) ** 2
    ) / denom if denom and not math.isnan(denom) else float("nan")

    return {
        "rg": rg,
        "rg2": rg2,
        "gyr_e1": lam1,
        "gyr_e2": lam2,
        "gyr_e3": lam3,
        "asphericity": float(b),
    }


def radial_metrics(xyz: np.ndarray) -> Dict[str, float]:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    r = np.linalg.norm(centered, axis=1)
    return {
        "r_mean": float(r.mean()),
        "r_std": float(r.std(ddof=0)),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
    }


_NAME_RE = re.compile(r"^C(?P<C>\d+)-(?P<tag>[^.]+)\.xyz$")


def parse_name_tokens(filename: str) -> Dict[str, Optional[str]]:
    m = _NAME_RE.match(filename)
    if not m:
        return {"symmetry": None, "isomer_id": None}

    tag = m.group("tag")
    parts = tag.split("-")
    symmetry = parts[0] if parts else None
    isomer_id = parts[-1] if len(parts) >= 2 else None
    if isomer_id is not None and not isomer_id.isdigit():
        # Some have no numeric suffix (e.g., C60-Ih)
        isomer_id = None
    return {"symmetry": symmetry, "isomer_id": isomer_id}


def expected_face_counts_for_fullerene(n: int) -> Dict[str, Optional[int]]:
    """For a classical fullerene (3-regular planar, only 5/6 faces), F5=12 and F6=n/2-10."""

    if n % 2 != 0:
        return {"pentagons_expected": None, "hexagons_expected": None}
    hexes = n // 2 - 10
    if hexes < 0:
        return {"pentagons_expected": None, "hexagons_expected": None}
    return {"pentagons_expected": 12, "hexagons_expected": int(hexes)}


def load_index(index_csv: Path) -> List[Dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_dataset(dataset_dir: Path, out_dir: Path) -> None:
    index_csv = dataset_dir / "index.csv"
    rows = load_index(index_csv)

    per_struct: List[Dict[str, object]] = []

    for r in rows:
        C = int(r["C"])
        filename = r["filename"]
        valid = int(r["valid"])
        if valid != 1:
            continue

        xyz_path = dataset_dir / f"C{C}" / filename
        parsed = parse_extended_xyz(xyz_path)
        edges = build_edges(parsed.neighbors)

        comps = connected_components(parsed.n, edges)
        deg_min, deg_max, deg_mean = degree_stats(parsed.n, edges)
        bonds = bond_lengths(parsed.xyz, edges)

        rm = radial_metrics(parsed.xyz)
        gm = gyration_metrics(parsed.xyz)
        faces = expected_face_counts_for_fullerene(parsed.n)
        tokens = parse_name_tokens(filename)

        entry: Dict[str, object] = {
            "C": C,
            "filename": filename,
            "name": parsed.name.strip(),
            "symmetry": tokens["symmetry"],
            "isomer_id": tokens["isomer_id"],
            "n_atoms": parsed.n,
            "n_edges": len(edges),
            "components": comps,
            "deg_min": deg_min,
            "deg_max": deg_max,
            "deg_mean": deg_mean,
            "bond_mean": float(bonds.mean()) if bonds.size else float("nan"),
            "bond_std": float(bonds.std(ddof=0)) if bonds.size else float("nan"),
            "bond_min": float(bonds.min()) if bonds.size else float("nan"),
            "bond_max": float(bonds.max()) if bonds.size else float("nan"),
            **rm,
            **gm,
            **faces,
            "bytes": int(r["bytes"]),
            "sha256": r["sha256"],
        }
        per_struct.append(entry)

    # Write per-structure CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    per_struct_path = out_dir / "structures.csv"
    if per_struct:
        fieldnames = list(per_struct[0].keys())
        with per_struct_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_struct)

    # Aggregate by C
    by_C: Dict[int, List[Dict[str, object]]] = {}
    for e in per_struct:
        by_C.setdefault(int(e["C"]), []).append(e)

    by_C_rows: List[Dict[str, object]] = []
    for C in sorted(by_C.keys()):
        items = by_C[C]
        def _mean(key: str) -> float:
            arr = np.array([float(x[key]) for x in items], dtype=float)
            return float(np.nanmean(arr))
        def _std(key: str) -> float:
            arr = np.array([float(x[key]) for x in items], dtype=float)
            return float(np.nanstd(arr))

        by_C_rows.append(
            {
                "C": C,
                "n_structures": len(items),
                "symmetry_mode": max(
                    (str(x["symmetry"]) for x in items),
                    key=lambda s: sum(1 for y in items if str(y["symmetry"]) == s),
                ),
                "bond_mean_mean": _mean("bond_mean"),
                "bond_mean_std": _std("bond_mean"),
                "r_mean_mean": _mean("r_mean"),
                "r_mean_std": _std("r_mean"),
                "asphericity_mean": _mean("asphericity"),
                "asphericity_std": _std("asphericity"),
                "rg_mean": _mean("rg"),
                "rg_std": _std("rg"),
            }
        )

    by_C_path = out_dir / "by_C.csv"
    if by_C_rows:
        with by_C_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(by_C_rows[0].keys()))
            w.writeheader()
            w.writerows(by_C_rows)

    # Plots + report
    _ensure_matplotlib_arial()
    import matplotlib.pyplot as plt

    Cs = [row["C"] for row in by_C_rows]
    counts = [row["n_structures"] for row in by_C_rows]

    def _savefig(name: str) -> str:
        p = out_dir / name
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close()
        return p.name

    # 1) Isomer count by C
    plt.figure(figsize=(10, 4))
    plt.bar(Cs, counts, color="#2E86AB")
    plt.xlabel("Number of carbon atoms (C)")
    plt.ylabel("Number of structures")
    plt.title("Fullerene isomer count by size")
    plot_isomers = _savefig("isomer_count_by_C.png")

    # 2) Mean radius vs C
    r_mean = [row["r_mean_mean"] for row in by_C_rows]
    r_std = [row["r_mean_std"] for row in by_C_rows]
    plt.figure(figsize=(8, 4))
    plt.errorbar(Cs, r_mean, yerr=r_std, fmt="o-", capsize=3, color="#F18F01")
    plt.xlabel("Number of carbon atoms (C)")
    plt.ylabel("Mean radial distance to centroid (Å)")
    plt.title("Size scaling: mean radius vs C")
    plot_radius = _savefig("radius_vs_C.png")

    # 3) Asphericity vs C
    asp = [row["asphericity_mean"] for row in by_C_rows]
    asp_std = [row["asphericity_std"] for row in by_C_rows]
    plt.figure(figsize=(8, 4))
    plt.errorbar(Cs, asp, yerr=asp_std, fmt="o-", capsize=3, color="#4CAF50")
    plt.xlabel("Number of carbon atoms (C)")
    plt.ylabel("Asphericity (0 = sphere-like)")
    plt.title("Shape descriptor: asphericity vs C")
    plot_asphericity = _savefig("asphericity_vs_C.png")

    # 4) Bond length distribution (global)
    all_bonds = np.array([float(x["bond_mean"]) for x in per_struct], dtype=float)
    plt.figure(figsize=(7, 4))
    plt.hist(all_bonds[~np.isnan(all_bonds)], bins=40, color="#6D597A", alpha=0.9)
    plt.xlabel("Mean bonded C–C distance per structure (Å)")
    plt.ylabel("Count")
    plt.title("Distribution of mean bond length across structures")
    plot_bonds = _savefig("bond_mean_hist.png")

    # HTML report
    now = _dt.datetime.now().strftime("%Y-%m-%d")
    total_struct = len(per_struct)
    distinct_C = len(by_C_rows)

    # Top-10 by isomer count
    top10 = sorted(by_C_rows, key=lambda x: int(x["n_structures"]), reverse=True)[:10]

    report_html = out_dir / "report.html"
    report_html.write_text(
        """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Fullerene XYZ Dataset Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 0.2em 0; }
    .meta { color: #444; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
    img { max-width: 100%; border: 1px solid #ddd; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f5f5f5; text-align: left; }
    .note { font-size: 13px; color: #333; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
  </style>
</head>
<body>
  <h1>Fullerene XYZ Dataset Analysis</h1>
  <div class=\"meta\">Generated on: """
        + now
        + """</div>

  <h2>Dataset overview</h2>
  <ul>
    <li>Total structures analyzed: <b>"""
        + str(total_struct)
        + """</b></li>
    <li>Distinct sizes (C values): <b>"""
        + str(distinct_C)
        + """</b></li>
    <li>File format: extended XYZ with explicit 3-neighbor connectivity (8 columns per atom line)</li>
    <li>Ring counts: the report lists <i>expected</i> pentagon/hexagon counts for classical fullerenes (F5=12, F6=C/2-10). Face enumeration is not performed here.</li>
  </ul>

  <h2>Most populated sizes</h2>
  <table>
    <thead><tr><th>C</th><th># structures</th><th>Mean radius (Å)</th><th>Mean asphericity</th></tr></thead>
    <tbody>
"""
        + "\n".join(
            f"<tr><td>C{r['C']}</td><td>{r['n_structures']}</td><td>{r['r_mean_mean']:.3f}</td><td>{r['asphericity_mean']:.4f}</td></tr>"
            for r in top10
        )
        + """
    </tbody>
  </table>

  <h2>Figures</h2>
  <div class=\"grid\">
    <div>
      <div class=\"note\"><b>Figure 1.</b> Isomer count by size.</div>
      <img src=\""""
        + plot_isomers
        + """\" alt=\"Isomer count by C\" />
    </div>
    <div>
      <div class=\"note\"><b>Figure 2.</b> Mean radius (with std error bars) vs size.</div>
      <img src=\""""
        + plot_radius
        + """\" alt=\"Radius vs C\" />
    </div>
    <div>
      <div class=\"note\"><b>Figure 3.</b> Asphericity (0 means sphere-like) vs size.</div>
      <img src=\""""
        + plot_asphericity
        + """\" alt=\"Asphericity vs C\" />
    </div>
    <div>
      <div class=\"note\"><b>Figure 4.</b> Distribution of mean bond length per structure.</div>
      <img src=\""""
        + plot_bonds
        + """\" alt=\"Bond length distribution\" />
    </div>
  </div>

  <h2>Outputs</h2>
  <ul>
    <li><code>structures.csv</code>: per-structure metrics and sanity checks</li>
    <li><code>by_C.csv</code>: per-C aggregated statistics</li>
  </ul>

  <div class=\"note\">
    Tip: For AI structure exploration, you can use <code>structures.csv</code> to select diverse seeds by size, radius, and asphericity; then add expensive labels (DFT/ML energies) via active learning.
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset/fullerenes/fullerene_xyz")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/fullerenes/fullerene_xyz"),
        help="Path to fullerene_xyz dataset directory (contains index.csv and C*/ folders)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset-dir>/analysis_YYYYMMDD",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if args.out_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d")
        out_dir = dataset_dir / f"analysis_{stamp}"
    else:
        out_dir = args.out_dir

    analyze_dataset(dataset_dir=dataset_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
