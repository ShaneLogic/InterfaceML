"""Step 1: Generate fullerene + perovskite structures using AI models."""

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_round(config: dict, round_dir: Path) -> dict:
    """Generate fullerene + perovskite structures for one AL round.

    Args:
        config: Full pipeline configuration dict.
        round_dir: Directory for this round (e.g. active_learning_runs/round_000/).

    Returns:
        dict with keys: fullerene_dir, perovskite_dir, fullerene_count, perovskite_count.
    """
    seed = config["pipeline"].get("seed", 42)
    gen_cfg = config["generation"]

    fullerene_dir = round_dir / "generated" / "fullerenes"
    perovskite_dir = round_dir / "generated" / "perovskites"
    fullerene_dir.mkdir(parents=True, exist_ok=True)
    perovskite_dir.mkdir(parents=True, exist_ok=True)

    fullerene_count = _generate_fullerenes(gen_cfg["fullerene"], fullerene_dir, seed)
    perovskite_count = _generate_perovskites(gen_cfg["perovskite"], perovskite_dir, seed)

    logger.info(
        "Generated %d fullerenes and %d perovskites",
        fullerene_count,
        perovskite_count,
    )

    return {
        "fullerene_dir": fullerene_dir,
        "perovskite_dir": perovskite_dir,
        "fullerene_count": fullerene_count,
        "perovskite_count": perovskite_count,
    }


def _generate_fullerenes(cfg: dict, output_dir: Path, seed: int) -> int:
    """Generate fullerene XYZ files using FullereneAPI."""
    # Import here to avoid hard dependency at module level
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fullerene_e3gen"))
    from api import FullereneAPI

    checkpoint = cfg["checkpoint"]
    c_values = cfg["C_values"]
    num_samples = cfg["num_samples_per_C"]

    logger.info("Loading fullerene model from %s", checkpoint)
    api = FullereneAPI(checkpoint_path=checkpoint)

    count = 0
    for c_val in c_values:
        logger.info("Generating %d C%d fullerenes", num_samples, c_val)
        np.random.seed(seed + c_val)

        samples = api.generate(num_carbon=c_val, num_samples=num_samples)

        for i, sample in enumerate(samples):
            filename = f"C{c_val}_sample_{i:03d}.xyz"
            filepath = output_dir / filename
            api.save_structure(
                positions=sample["positions"],
                edges=sample.get("edges", []),
                output_path=str(filepath),
                format="xyz",
            )
            count += 1
            logger.debug("Saved %s", filepath)

    return count


def _generate_perovskites(cfg: dict, output_dir: Path, seed: int) -> int:
    """Generate perovskite CIF files using PerovskiteAPI."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perovskite_e3gen"))
    from api import PerovskiteAPI

    checkpoint = cfg["checkpoint"]
    compositions = cfg["compositions"]
    num_atoms = cfg.get("num_atoms", 5)
    num_samples = cfg["num_samples_per_comp"]

    logger.info("Loading perovskite model from %s", checkpoint)
    api = PerovskiteAPI(checkpoint_path=checkpoint)

    count = 0
    for comp in compositions:
        a, b, x = comp["A"], comp["B"], comp["X"]
        formula = f"{a}{b}{x}3"
        logger.info("Generating %d %s perovskites", num_samples, formula)

        samples = api.generate(
            A=a,
            B=b,
            X=x,
            num_atoms=num_atoms,
            num_samples=num_samples,
            seed=seed,
        )

        for i, sample in enumerate(samples):
            filename = f"{formula}_sample_{i:03d}.cif"
            filepath = output_dir / filename
            api.save_structure(sample, str(filepath), fmt="cif")
            count += 1
            logger.debug("Saved %s", filepath)

    return count
