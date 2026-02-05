"""
AI integration helpers for the InterfaceML web app.

This module centralizes discovery and loading of the fullerene diffusion model,
including optional environment variable overrides for custom deployments.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class AiStatus:
    """Status container for AI module availability."""

    available: bool
    api: Optional[Any]
    error: Optional[str]
    base_path: Optional[Path]
    checkpoint_path: Optional[Path]
    local_refiner: Optional[Any] = None
    local_error: Optional[str] = None


def _resolve_fullerene_path() -> Path:
    env_path = os.getenv("INTERFACEML_FULLERENE_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent.parent.parent / "fullerene_diffusion_poc"


def _resolve_checkpoint(base_path: Path) -> Path:
    env_ckpt = os.getenv("INTERFACEML_FULLERENE_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    return base_path / "checkpoints" / "best_model.pt"


def _resolve_local_refiner_checkpoint() -> Optional[Path]:
    env_ckpt = os.getenv("INTERFACEML_LOCAL_GNN_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    return None


def load_fullerene_api() -> AiStatus:
    """Attempt to load the fullerene diffusion API and return a status object."""
    base_path = _resolve_fullerene_path()
    if not base_path.exists():
        return AiStatus(
            available=False,
            api=None,
            error=f"Fullerene module path not found: {base_path}",
            base_path=base_path,
            checkpoint_path=None,
            local_refiner=None,
            local_error=None,
        )

    # Ensure the module path is importable
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))

    try:
        from api import FullereneAPI  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return AiStatus(
            available=False,
            api=None,
            error=f"Failed to import FullereneAPI: {exc}",
            base_path=base_path,
            checkpoint_path=None,
            local_refiner=None,
            local_error=None,
        )

    checkpoint_path = _resolve_checkpoint(base_path)
    if not checkpoint_path.exists():
        return AiStatus(
            available=False,
            api=None,
            error=f"Fullerene checkpoint not found: {checkpoint_path}",
            base_path=base_path,
            checkpoint_path=checkpoint_path,
            local_refiner=None,
            local_error=None,
        )

    try:
        api = FullereneAPI(str(checkpoint_path))
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return AiStatus(
            available=False,
            api=None,
            error=f"Failed to initialize FullereneAPI: {exc}",
            base_path=base_path,
            checkpoint_path=checkpoint_path,
            local_refiner=None,
            local_error=None,
        )

    local_refiner = None
    local_error = None
    local_ckpt = _resolve_local_refiner_checkpoint()
    if local_ckpt is not None:
        try:
            from interfaceml.ai.local_gnn import LocalRefiner

            local_refiner = LocalRefiner(checkpoint_path=str(local_ckpt))
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            local_error = f"Failed to initialize LocalRefiner: {exc}"

    return AiStatus(
        available=True,
        api=api,
        error=None,
        base_path=base_path,
        checkpoint_path=checkpoint_path,
        local_refiner=local_refiner,
        local_error=local_error,
    )
