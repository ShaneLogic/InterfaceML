#!/usr/bin/env python3
"""Create ML-ready assets from the fullerene_xyz analysis outputs.

Reads:
  dataset/fullerenes/fullerene_xyz/<analysis_dir>/structures.csv

Writes into:
  dataset/fullerenes/fullerene_xyz/<analysis_dir>/ml_assets/

Outputs (English-only, Arial font where applicable):
  - split.csv (train/val/test assignment per structure)
  - split_summary_by_C.csv
  - weights_by_C.csv (for reweighting / balanced sampling)
  - split_report.html + plots

Design goals:
  - Deterministic split (seeded)
  - Stratified by size C (each C split independently)
  - Sensible behavior for tiny groups (n < 3)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _ensure_matplotlib_arial() -> None:
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 20260122


def _read_structures(structures_csv: Path) -> List[Dict[str, str]]:
    with structures_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _counts_for_group(n: int, cfg: SplitConfig) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a group of size n.

    Rules:
      - If n < 3: put everything in train (val/test would be meaningless)
      - Otherwise: round by fractions, then enforce at least 1 val and 1 test
        and ensure all counts are non-negative and sum to n.
    """

    if n < 3:
        return n, 0, 0

    n_val = int(round(n * cfg.val_frac))
    n_test = int(round(n * cfg.test_frac))
    n_train = n - n_val - n_test

    # Ensure at least 1 for val/test
    if n_val == 0:
        n_val = 1
        n_train -= 1
    if n_test == 0:
        n_test = 1
        n_train -= 1

    # If train got negative due to tiny n, fix by borrowing back.
    if n_train < 1:
        # Keep at least 1 train; reduce val/test as needed (but keep >=1 each if possible)
        deficit = 1 - n_train
        n_train = 1
        while deficit > 0 and n_val > 1:
            n_val -= 1
            deficit -= 1
        while deficit > 0 and n_test > 1:
            n_test -= 1
            deficit -= 1
        # Final safety: if still deficit (e.g., n=3), allow 1/1/1

    # Final correction to ensure sum exactly n
    total = n_train + n_val + n_test
    if total != n:
        n_train += (n - total)

    # Sanity
    if n_train < 0 or n_val < 0 or n_test < 0:
        raise ValueError(f"Invalid split counts for n={n}: {(n_train, n_val, n_test)}")
    if n_train + n_val + n_test != n:
        raise ValueError(f"Split counts do not sum to n={n}: {(n_train, n_val, n_test)}")

    return n_train, n_val, n_test


def _assign_splits(rows: List[Dict[str, str]], cfg: SplitConfig) -> List[Dict[str, str]]:
    """Assign train/val/test splits stratified by C."""

    by_C: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_C[int(r["C"])].append(r)

    rng = np.random.default_rng(cfg.seed)

    out: List[Dict[str, str]] = []
    for C in sorted(by_C.keys()):
        group = list(by_C[C])
        idx = np.arange(len(group))
        rng.shuffle(idx)

        n_train, n_val, n_test = _counts_for_group(len(group), cfg)
        train_idx = set(idx[:n_train])
        val_idx = set(idx[n_train : n_train + n_val])
        test_idx = set(idx[n_train + n_val :])

        for i, r in enumerate(group):
            split = "train" if i in train_idx else "val" if i in val_idx else "test"
            out.append(
                {
                    "C": r["C"],
                    "filename": r["filename"],
                    "split": split,
                    "symmetry": r.get("symmetry", ""),
                    "isomer_id": r.get("isomer_id", ""),
                    "r_mean": r.get("r_mean", ""),
                    "asphericity": r.get("asphericity", ""),
                    "bond_mean": r.get("bond_mean", ""),
                }
            )

    return out


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _weights_by_C(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return per-C weights normalized to mean 1.

    - weight_inv: proportional to 1 / count
    - weight_sqrt_inv: proportional to 1 / sqrt(count)
    """

    counts = Counter(int(r["C"]) for r in rows)

    inv = {C: 1.0 / n for C, n in counts.items()}
    sqrt_inv = {C: 1.0 / math.sqrt(n) for C, n in counts.items()}

    inv_mean = sum(inv.values()) / len(inv)
    sqrt_mean = sum(sqrt_inv.values()) / len(sqrt_inv)

    out = []
    for C in sorted(counts.keys()):
        out.append(
            {
                "C": str(C),
                "n_structures": str(counts[C]),
                "weight_inv": f"{inv[C] / inv_mean:.6f}",
                "weight_sqrt_inv": f"{sqrt_inv[C] / sqrt_mean:.6f}",
            }
        )
    return out


def _split_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_C = defaultdict(lambda: Counter())
    for r in rows:
        by_C[int(r["C"])][r["split"]] += 1

    out = []
    for C in sorted(by_C.keys()):
        c = by_C[C]
        out.append(
            {
                "C": str(C),
                "train": str(c["train"]),
                "val": str(c["val"]),
                "test": str(c["test"]),
                "total": str(c["train"] + c["val"] + c["test"]),
            }
        )
    return out


def _make_report(out_dir: Path, summary: List[Dict[str, str]], weights: List[Dict[str, str]], rows: List[Dict[str, str]]) -> None:
    _ensure_matplotlib_arial()
    import matplotlib.pyplot as plt

    # Stacked bar: train/val/test counts by C
    Cs = [int(r["C"]) for r in summary]
    train = np.array([int(r["train"]) for r in summary])
    val = np.array([int(r["val"]) for r in summary])
    test = np.array([int(r["test"]) for r in summary])

    plt.figure(figsize=(10, 4))
    plt.bar(Cs, train, label="train", color="#2E86AB")
    plt.bar(Cs, val, bottom=train, label="val", color="#F18F01")
    plt.bar(Cs, test, bottom=train + val, label="test", color="#4CAF50")
    plt.xlabel("Number of carbon atoms (C)")
    plt.ylabel("Structures")
    plt.title("Train/Val/Test split distribution by size")
    plt.legend(frameon=False)
    split_plot = out_dir / "split_stacked_by_C.png"
    plt.savefig(split_plot, dpi=200, bbox_inches="tight")
    plt.close()

    # Weight curves
    wC = [int(r["C"]) for r in weights]
    w_inv = [float(r["weight_inv"]) for r in weights]
    w_sqrt = [float(r["weight_sqrt_inv"]) for r in weights]
    plt.figure(figsize=(9, 4))
    plt.plot(wC, w_inv, "o-", label="Inverse-count weight", color="#6D597A")
    plt.plot(wC, w_sqrt, "o-", label="Inverse-sqrt-count weight", color="#355070")
    plt.xlabel("Number of carbon atoms (C)")
    plt.ylabel("Weight (normalized mean = 1)")
    plt.title("Recommended class weights for imbalanced sizes")
    plt.legend(frameon=False)
    weights_plot = out_dir / "weights_by_C.png"
    plt.savefig(weights_plot, dpi=200, bbox_inches="tight")
    plt.close()

    # Report HTML
    now = _dt.datetime.now().strftime("%Y-%m-%d")
    total = len(rows)
    counts = Counter(r["split"] for r in rows)

    top_imbalanced = sorted(weights, key=lambda r: float(r["weight_inv"]), reverse=True)[:10]

    html = out_dir / "split_report.html"
    html.write_text(
        """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Fullerene XYZ ML Assets Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 0.2em 0; }
    .meta { color: #444; margin-bottom: 16px; }
    img { max-width: 100%; border: 1px solid #ddd; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f5f5f5; text-align: left; }
    .note { font-size: 13px; color: #333; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
  </style>
</head>
<body>
  <h1>ML Assets Report (Fullerene XYZ)</h1>
  <div class=\"meta\">Generated on: """
        + now
        + """</div>

  <h2>Split overview</h2>
  <ul>
    <li>Total structures: <b>"""
        + str(total)
        + """</b></li>
    <li>train/val/test: <b>"""
        + f"{counts.get('train',0)}/{counts.get('val',0)}/{counts.get('test',0)}"
        + """</b></li>
    <li>Stratification: split is created independently within each size C</li>
    <li>Determinism: seeded shuffling (seed="""
        + str(SplitConfig().seed)
        + """)</li>
  </ul>

  <h2>Figures</h2>
  <div class=\"note\"><b>Figure 1.</b> Split distribution by C (stacked).</div>
  <img src=\"split_stacked_by_C.png\" alt=\"Split distribution\" />

  <div class=\"note\" style=\"margin-top:12px\"><b>Figure 2.</b> Suggested class weights for imbalanced C sizes.</div>
  <img src=\"weights_by_C.png\" alt=\"Weights\" />

  <h2>Most upweighted sizes (inverse-count)</h2>
  <table>
    <thead><tr><th>C</th><th># structures</th><th>weight_inv</th><th>weight_sqrt_inv</th></tr></thead>
    <tbody>
"""
        + "\n".join(
            f"<tr><td>C{r['C']}</td><td>{r['n_structures']}</td><td>{r['weight_inv']}</td><td>{r['weight_sqrt_inv']}</td></tr>"
            for r in top_imbalanced
        )
        + """
    </tbody>
  </table>

  <h2>Files</h2>
  <ul>
    <li><code>split.csv</code>: per-structure split assignment</li>
    <li><code>split_summary_by_C.csv</code>: split counts per C</li>
    <li><code>weights_by_C.csv</code>: recommended weights per C</li>
  </ul>

  <div class=\"note\">
    Practical training tip: If you train a conditional model on C, consider sampling C uniformly and sampling structures uniformly within that C, or use <code>weight_sqrt_inv</code> as a loss weight to avoid over-amplifying tiny groups.
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ML assets for fullerene_xyz")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("dataset/fullerenes/fullerene_xyz/analysis_20260122"),
        help="Path to the analysis directory containing structures.csv",
    )
    parser.add_argument("--seed", type=int, default=SplitConfig.seed)
    parser.add_argument("--train-frac", type=float, default=SplitConfig.train_frac)
    parser.add_argument("--val-frac", type=float, default=SplitConfig.val_frac)
    parser.add_argument("--test-frac", type=float, default=SplitConfig.test_frac)
    args = parser.parse_args()

    cfg = SplitConfig(train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    structures_csv = args.analysis_dir / "structures.csv"
    out_dir = args.analysis_dir / "ml_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_structures(structures_csv)
    splits = _assign_splits(rows, cfg)

    # Validate uniqueness
    keyset = set()
    for r in splits:
        k = (r["C"], r["filename"])
        if k in keyset:
            raise RuntimeError(f"Duplicate structure key in split: {k}")
        keyset.add(k)
    if len(keyset) != len(rows):
        raise RuntimeError("Split rows do not cover all structures")

    _write_csv(out_dir / "split.csv", list(splits[0].keys()), splits)

    summary = _split_summary(splits)
    _write_csv(out_dir / "split_summary_by_C.csv", list(summary[0].keys()), summary)

    weights = _weights_by_C(splits)
    _write_csv(out_dir / "weights_by_C.csv", list(weights[0].keys()), weights)

    _make_report(out_dir=out_dir, summary=summary, weights=weights, rows=splits)


if __name__ == "__main__":
    main()
