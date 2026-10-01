"""Main orchestrator for the active learning pipeline.

Usage:
    # Run phases 1-3 (generate → build → dft_prep), then stop for manual DFT
    python -m active_learning.pipeline --config active_learning/config.yaml --round 0

    # After DFT completes, parse results
    python -m active_learning.pipeline --config active_learning/config.yaml --round 0 --phase parse

    # Prepare retraining data
    python -m active_learning.pipeline --config active_learning/config.yaml --round 0 --phase retrain

    # Run a specific phase
    python -m active_learning.pipeline --phase generate --round 1
    python -m active_learning.pipeline --phase build --round 1
    python -m active_learning.pipeline --phase dft_prep --round 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from .build_interfaces import build_interfaces
from .dft_inputs import prepare_dft_calculations
from .generate_components import generate_round
from .parse_results import parse_dft_results
from .prepare_retraining import prepare_retraining_data

logger = logging.getLogger(__name__)

PHASES = ["generate", "build", "dft_prep", "parse", "retrain"]
# Phases that run before DFT (automated)
PRE_DFT_PHASES = ["generate", "build", "dft_prep"]


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open() as f:
        return yaml.safe_load(f)


def get_round_dir(config: dict, round_num: int) -> Path:
    """Get the directory for a specific round."""
    base_dir = Path(config["pipeline"].get("base_dir", "active_learning_runs"))
    return base_dir / f"round_{round_num:03d}"


def run_phase(phase: str, config: dict, round_dir: Path) -> dict:
    """Run a single phase of the pipeline.

    Args:
        phase: One of PHASES.
        config: Full pipeline configuration.
        round_dir: Directory for this round.

    Returns:
        Phase result dict.
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    state_path = round_dir / "pipeline_state.json"
    state = _load_state(state_path)

    logger.info("=" * 60)
    logger.info("Running phase: %s", phase)
    logger.info("Round directory: %s", round_dir)
    logger.info("=" * 60)

    if phase == "generate":
        result = generate_round(config, round_dir)
        state["generate"] = {
            "status": "completed",
            "fullerene_dir": str(result["fullerene_dir"]),
            "perovskite_dir": str(result["perovskite_dir"]),
            "fullerene_count": result["fullerene_count"],
            "perovskite_count": result["perovskite_count"],
        }

    elif phase == "build":
        gen_state = state.get("generate", {})
        fullerene_dir = Path(
            gen_state.get("fullerene_dir", round_dir / "generated" / "fullerenes")
        )
        perovskite_dir = Path(
            gen_state.get("perovskite_dir", round_dir / "generated" / "perovskites")
        )
        interfaces_dir = round_dir / "interfaces"

        records = build_interfaces(
            fullerene_dir=fullerene_dir,
            perovskite_dir=perovskite_dir,
            output_dir=interfaces_dir,
            config=config,
        )
        successful = [r for r in records if r.get("status") == "success"]
        state["build"] = {
            "status": "completed",
            "interfaces_dir": str(interfaces_dir),
            "total_records": len(records),
            "successful": len(successful),
        }
        result = {"records": records, "interfaces_dir": interfaces_dir}

    elif phase == "dft_prep":
        build_state = state.get("build", {})
        interfaces_dir = Path(
            build_state.get("interfaces_dir", round_dir / "interfaces")
        )

        # Load metadata
        metadata_path = interfaces_dir / "metadata.jsonl"
        records = []
        if metadata_path.exists():
            with metadata_path.open() as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

        dft_jobs_dir = round_dir / "dft_jobs"
        jobs = prepare_dft_calculations(records, dft_jobs_dir, config)
        prepared = [j for j in jobs if j.get("status") == "prepared"]

        state["dft_prep"] = {
            "status": "completed",
            "dft_jobs_dir": str(dft_jobs_dir),
            "total_jobs": len(jobs),
            "prepared": len(prepared),
        }
        result = {"jobs": jobs, "dft_jobs_dir": dft_jobs_dir}

        logger.info("")
        logger.info("=" * 60)
        logger.info("DFT preparation complete!")
        logger.info("  Jobs directory: %s", dft_jobs_dir)
        logger.info("  Prepared jobs:  %d", len(prepared))
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review CP2K inputs in %s", dft_jobs_dir)
        logger.info("  2. Submit jobs:  bash %s/submit_all.sh", dft_jobs_dir)
        logger.info("  3. After DFT completes, run:")
        logger.info(
            "     python -m active_learning.pipeline --round %d --phase parse",
            config["pipeline"].get("round", 0),
        )
        logger.info("=" * 60)

    elif phase == "parse":
        dft_state = state.get("dft_prep", {})
        dft_jobs_dir = Path(
            dft_state.get("dft_jobs_dir", round_dir / "dft_jobs")
        )
        results_dir = round_dir / "results"
        results_path = results_dir / "results.json"

        results = parse_dft_results(dft_jobs_dir, results_path)
        converged = sum(1 for r in results if r.get("converged"))

        state["parse"] = {
            "status": "completed",
            "results_path": str(results_path),
            "total_results": len(results),
            "converged": converged,
        }
        result = {"results": results, "results_path": results_path}

    elif phase == "retrain":
        parse_state = state.get("parse", {})
        results_path = Path(
            parse_state.get("results_path", round_dir / "results" / "results.json")
        )
        retrain_dir = round_dir / "retraining"

        # Existing data directory (project-level)
        existing_data = Path(
            config.get("retraining", {}).get(
                "existing_data_dir", "dataset"
            )
        )

        summary = prepare_retraining_data(
            results_path=results_path,
            existing_data_dir=existing_data,
            output_dir=retrain_dir,
            config=config,
        )

        state["retrain"] = {
            "status": "completed",
            "retrain_dir": str(retrain_dir),
            "num_accepted": summary["num_accepted"],
            "num_rejected": summary["num_rejected"],
        }
        result = summary

    else:
        raise ValueError(f"Unknown phase: {phase}. Must be one of {PHASES}")

    _save_state(state, state_path)
    return result


def run_round(config_path: str, round_num: int = None, phase: str = None):
    """Execute one active learning round (or a specific phase).

    Args:
        config_path: Path to config.yaml.
        round_num: Override round number (default: from config).
        phase: Run only this phase (default: run pre-DFT phases).
    """
    config = load_config(config_path)

    if round_num is None:
        round_num = config["pipeline"].get("round", 0)
    config["pipeline"]["round"] = round_num

    round_dir = get_round_dir(config, round_num)

    if phase:
        return run_phase(phase, config, round_dir)

    # Run all pre-DFT phases sequentially
    results = {}
    for p in PRE_DFT_PHASES:
        results[p] = run_phase(p, config, round_dir)

    return results


def _load_state(state_path: Path) -> dict:
    """Load pipeline state from JSON."""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def _save_state(state: dict, state_path: Path) -> None:
    """Save pipeline state to JSON."""
    state_path.write_text(json.dumps(state, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Active learning pipeline: AI → Interface → DFT → Retrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full round (stops after dft_prep for manual DFT submission)
  python -m active_learning.pipeline --round 0

  # After DFT completes, parse results
  python -m active_learning.pipeline --round 0 --phase parse

  # Prepare retraining data
  python -m active_learning.pipeline --round 0 --phase retrain

  # Run specific phases
  python -m active_learning.pipeline --phase generate --round 1
  python -m active_learning.pipeline --phase build --round 1
        """,
    )
    parser.add_argument(
        "--config",
        default="active_learning/config.yaml",
        help="Path to pipeline config (default: active_learning/config.yaml)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round number (default: from config)",
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default=None,
        help="Run only this phase (default: run generate→build→dft_prep)",
    )

    args = parser.parse_args()

    run_round(
        config_path=args.config,
        round_num=args.round,
        phase=args.phase,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
