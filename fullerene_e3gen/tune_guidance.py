"""Guidance scale tuning script for Fullerene Diffusion Model.

Sweeps guidance_scale values and recommends the optimal setting based
on a composite quality score of validity, bond distribution, and sphericity.

Usage:
    python tune_guidance.py --checkpoint checkpoints/best_model.pt --C 60 --num_samples 10

Author: InterfaceML Project
Date: 2026-02
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

from diffusion_utils import DiffusionScheduler
from evaluate import compute_sphericity
from generate import (
    compute_bond_stats,
    generate_samples,
    is_valid_structure,
)
from model import FullereneDiffusionModel
from template_bank import TemplateBank
from units import denormalize_positions


def _create_model_from_checkpoint(
    checkpoint: dict, device: torch.device
) -> tuple[FullereneDiffusionModel, DiffusionScheduler, dict]:
    """Instantiate model + scheduler from a saved checkpoint."""
    config = checkpoint["config"]
    model_config = config["model"]

    model = FullereneDiffusionModel(
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        edge_dim=model_config["edge_dim"],
        C_embed_dim=model_config["C_embed_dim"],
        time_embed_dim=model_config["time_embed_dim"],
        max_C=model_config.get("max_C", 720),
        continuous_C_embed=model_config.get("continuous_C_embed", False),
        C_fourier_features=model_config.get("C_fourier_features", 16),
        use_hierarchical=model_config.get("use_hierarchical", False),
        hierarchical_threshold=model_config.get("hierarchical_threshold", 80),
        hierarchical_layers=model_config.get("hierarchical_layers", 2),
        use_global_attention=model_config.get("use_global_attention", False),
        global_attention_heads=model_config.get("global_attention_heads", 4),
    ).to(device)

    # Prefer EMA weights
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded primary weights (no EMA available)")

    diff_config = config["diffusion"]
    scheduler = DiffusionScheduler(
        num_steps=diff_config["num_steps"],
        beta_schedule=diff_config["beta_schedule"],
        beta_start=diff_config.get("beta_start", 0.0001),
        beta_end=diff_config.get("beta_end", 0.02),
        device=device,
    )

    return model, scheduler, config


def _evaluate_scale(
    model: FullereneDiffusionModel,
    scheduler: DiffusionScheduler,
    config: dict,
    template_bank: TemplateBank,
    C_value: int,
    num_samples: int,
    guidance_scale: float,
    device: torch.device,
) -> dict:
    """Generate samples at a given guidance_scale and return quality metrics."""

    structures = generate_samples(
        model=model,
        scheduler=scheduler,
        num_samples=num_samples,
        C_value=C_value,
        device=device,
        template_bank=template_bank,
        topology_model=None,
        guidance_scale=guidance_scale,
        sampling_project_radius=True,
        sampling_rescale_each_step=True,
        sampling_bond_iters=1,
        sampling_bond_strength=0.3,
    )

    if not structures:
        return {
            "guidance_scale": guidance_scale,
            "num_generated": 0,
            "validity": 0.0,
            "bond_mean": 0.0,
            "bond_std": 0.0,
            "sphericity_anisotropy": 1.0,
            "degree3_frac": 0.0,
        }

    # Get a template for edge_index
    template = template_bank.make_template_data(C_value, device=torch.device("cpu"))
    edge_index = template.edge_index

    valid_count = 0
    bond_means = []
    bond_stds = []
    anisotropies = []
    degree3_count = 0

    for pos in structures:
        bm, bs = compute_bond_stats(pos, edge_index)
        bond_means.append(bm)
        bond_stds.append(bs)

        ok, _ = is_valid_structure(
            pos,
            edge_index,
            C_value,
            bond_mean_tol=0.15,
            bond_std_max=0.15,
            radius_mean_tol=0.2,
            radius_std_max=0.2,
        )
        if ok:
            valid_count += 1

        # Sphericity (in Angstrom space for meaningful anisotropy)
        pos_ang = denormalize_positions(pos, C_value).cpu().numpy()
        sph = compute_sphericity(pos_ang)
        anisotropies.append(sph["shape_anisotropy"])

        # Degree-3 check
        row = edge_index[0]
        degrees = torch.zeros(C_value)
        degrees.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
        if (degrees == 3).all():
            degree3_count += 1

    n = len(structures)
    import numpy as np

    return {
        "guidance_scale": guidance_scale,
        "num_generated": n,
        "validity": valid_count / n,
        "bond_mean": float(np.mean(bond_means)),
        "bond_std": float(np.mean(bond_stds)),
        "sphericity_anisotropy": float(np.mean(anisotropies)),
        "degree3_frac": degree3_count / n,
    }


def _composite_score(metrics: dict) -> float:
    """Compute composite quality score (higher is better).

    score = validity - 0.5 * bond_std - 0.3 * anisotropy
    """
    return (
        metrics["validity"]
        - 0.5 * metrics["bond_std"]
        - 0.3 * metrics["sphericity_anisotropy"]
    )


def main():
    parser = argparse.ArgumentParser(description="Tune guidance_scale for fullerene generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--C", type=int, default=60, help="Carbon count to evaluate")
    parser.add_argument("--num_samples", type=int, default=10, help="Samples per scale")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
        help="Guidance scales to sweep",
    )
    parser.add_argument("--output", type=str, default="guidance_tuning.json", help="Output JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, scheduler, config = _create_model_from_checkpoint(checkpoint, device)

    # Template bank
    data_cfg = config.get("data", {})
    xyz_dir = data_cfg.get("xyz_dir")
    if not xyz_dir:
        raise ValueError("Config missing data.xyz_dir")
    xyz_path = Path(xyz_dir)
    if not xyz_path.is_absolute():
        xyz_path = (Path(__file__).resolve().parent / xyz_path).resolve()
    template_bank = TemplateBank(xyz_dir=xyz_path)

    # Sweep
    results = []
    for scale in args.scales:
        logger.info("--- guidance_scale=%.2f ---", scale)
        metrics = _evaluate_scale(
            model, scheduler, config, template_bank,
            C_value=args.C,
            num_samples=args.num_samples,
            guidance_scale=scale,
            device=device,
        )
        metrics["composite_score"] = _composite_score(metrics)
        results.append(metrics)

        logger.info(
            "  valid=%.0f%%  bond=%.4f±%.4f  aniso=%.4f  score=%.4f",
            metrics["validity"] * 100,
            metrics["bond_mean"],
            metrics["bond_std"],
            metrics["sphericity_anisotropy"],
            metrics["composite_score"],
        )

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Scale':>8} {'Valid%':>8} {'BondMean':>10} {'BondStd':>10} {'Aniso':>8} {'Score':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['guidance_scale']:8.2f} {r['validity']*100:7.1f}% "
            f"{r['bond_mean']:10.4f} {r['bond_std']:10.4f} "
            f"{r['sphericity_anisotropy']:8.4f} {r['composite_score']:8.4f}"
        )
    print("=" * 80)

    # Recommend best
    best = max(results, key=lambda r: r["composite_score"])
    print(f"\nRecommended guidance_scale: {best['guidance_scale']:.2f} (score={best['composite_score']:.4f})")

    # Save JSON
    output = {
        "C_value": args.C,
        "num_samples": args.num_samples,
        "results": results,
        "recommended_scale": best["guidance_scale"],
        "recommended_score": best["composite_score"],
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
