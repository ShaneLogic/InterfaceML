"""
AI integration helpers for the InterfaceML web app.

This module centralizes discovery and loading of the fullerene diffusion models,
including optional environment variable overrides for custom deployments.

Supports multiple model backends:
  - ``egnn``: Enhanced EGNN + DDPM (default, original model)
  - ``painn_fm``: PaiNN + Flow Matching (angular-aware, faster sampling)

Both models are loaded at startup (when checkpoints exist) and stored in
``AiStatus.models``.  The web routes accept a ``model`` parameter so
users can choose which backend to use for generation.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Describes a single loaded model backend."""

    key: str                     # e.g. "egnn", "painn_fm"
    label: str                   # Human-readable display name
    api: Any                     # FullereneAPI instance (or None)
    checkpoint_path: Optional[Path]
    architecture: str            # "egnn" or "painn"
    diffusion_type: str          # "ddpm" or "flow_matching"
    error: Optional[str] = None
    available: bool = False


@dataclass
class AiStatus:
    """Status container for AI module availability (multi-model)."""

    available: bool
    api: Optional[Any]                          # Default model API (backward compat)
    error: Optional[str]
    base_path: Optional[Path]
    checkpoint_path: Optional[Path]
    local_refiner: Optional[Any] = None
    local_error: Optional[str] = None

    # --- Multi-model registry ---
    models: Dict[str, ModelEntry] = field(default_factory=dict)
    default_model: str = "egnn"

    def get_api(self, model_key: Optional[str] = None) -> Optional[Any]:
        """Return the FullereneAPI for *model_key*, falling back to default."""
        key = model_key or self.default_model
        entry = self.models.get(key)
        if entry and entry.available and entry.api is not None:
            return entry.api
        # Fallback to legacy .api
        return self.api

    def get_model_entry(self, model_key: Optional[str] = None) -> Optional[ModelEntry]:
        key = model_key or self.default_model
        return self.models.get(key)

    @property
    def available_model_keys(self):
        return [k for k, v in self.models.items() if v.available]


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_fullerene_path() -> Path:
    env_path = os.getenv("INTERFACEML_FULLERENE_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent.parent.parent / "fullerene_e3gen"


def _resolve_checkpoint(base_path: Path, subdir: Optional[str] = None) -> Path:
    env_ckpt = os.getenv("INTERFACEML_FULLERENE_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    if subdir:
        return base_path / "checkpoints" / subdir / "best_model.pt"
    return base_path / "checkpoints" / "best_model.pt"


def _resolve_painn_checkpoint(base_path: Path) -> Path:
    env_ckpt = os.getenv("INTERFACEML_PAINN_FM_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    return base_path / "checkpoints" / "painn_fm" / "best_model.pt"


def _resolve_local_refiner_checkpoint() -> Optional[Path]:
    env_ckpt = os.getenv("INTERFACEML_LOCAL_GNN_CHECKPOINT")
    if env_ckpt:
        return Path(env_ckpt)
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _try_load_model(
    FullereneAPI: type,
    checkpoint_path: Path,
    key: str,
    label: str,
    architecture: str,
    diffusion_type: str,
) -> ModelEntry:
    """Try to instantiate a FullereneAPI from *checkpoint_path*."""
    if not checkpoint_path.exists():
        return ModelEntry(
            key=key, label=label, api=None,
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            diffusion_type=diffusion_type,
            error=f"Checkpoint not found: {checkpoint_path}",
            available=False,
        )
    try:
        api = FullereneAPI(str(checkpoint_path))
        return ModelEntry(
            key=key, label=label, api=api,
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            diffusion_type=diffusion_type,
            error=None,
            available=True,
        )
    except Exception as exc:
        return ModelEntry(
            key=key, label=label, api=None,
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            diffusion_type=diffusion_type,
            error=f"Failed to load: {exc}",
            available=False,
        )


def load_fullerene_api() -> AiStatus:
    """Attempt to load all available fullerene model backends and return a status object."""
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

    # ------------------------------------------------------------------
    # 1. Load EGNN (default / original model)
    # ------------------------------------------------------------------
    egnn_ckpt = _resolve_checkpoint(base_path)
    egnn_entry = _try_load_model(
        FullereneAPI, egnn_ckpt,
        key="egnn",
        label="EGNN + DDPM",
        architecture="egnn",
        diffusion_type="ddpm",
    )
    if egnn_entry.available:
        logger.info("Loaded EGNN model from %s", egnn_ckpt)
    else:
        logger.warning("EGNN model unavailable: %s", egnn_entry.error)

    # ------------------------------------------------------------------
    # 2. Load PaiNN + Flow Matching
    # ------------------------------------------------------------------
    painn_ckpt = _resolve_painn_checkpoint(base_path)
    painn_entry = _try_load_model(
        FullereneAPI, painn_ckpt,
        key="painn_fm",
        label="PaiNN + Flow Matching",
        architecture="painn",
        diffusion_type="flow_matching",
    )
    if painn_entry.available:
        logger.info("Loaded PaiNN+FM model from %s", painn_ckpt)
    else:
        logger.warning("PaiNN+FM model unavailable: %s", painn_entry.error)

    # ------------------------------------------------------------------
    # Determine default / primary API
    # ------------------------------------------------------------------
    models: Dict[str, ModelEntry] = {}
    if egnn_entry.available:
        models["egnn"] = egnn_entry
    if painn_entry.available:
        models["painn_fm"] = painn_entry

    # Even if a model only partially available, store it so the frontend
    # can show its status.
    if not egnn_entry.available:
        models["egnn"] = egnn_entry
    if not painn_entry.available:
        models["painn_fm"] = painn_entry

    primary_api = None
    primary_ckpt = None
    default_model = "egnn"
    if egnn_entry.available:
        primary_api = egnn_entry.api
        primary_ckpt = egnn_entry.checkpoint_path
    elif painn_entry.available:
        primary_api = painn_entry.api
        primary_ckpt = painn_entry.checkpoint_path
        default_model = "painn_fm"

    any_available = egnn_entry.available or painn_entry.available
    primary_error = None if any_available else (egnn_entry.error or painn_entry.error)

    # ------------------------------------------------------------------
    # 3. Load local GNN refiner (shared across models)
    # ------------------------------------------------------------------
    local_refiner = None
    local_error = None
    local_ckpt = _resolve_local_refiner_checkpoint()
    if local_ckpt is not None:
        try:
            from interfaceml.ai.local_gnn import LocalRefiner

            local_refiner = LocalRefiner(checkpoint_path=str(local_ckpt))
        except Exception as exc:  # pragma: no cover
            local_error = f"Failed to initialize LocalRefiner: {exc}"

    return AiStatus(
        available=any_available,
        api=primary_api,
        error=primary_error,
        base_path=base_path,
        checkpoint_path=primary_ckpt,
        local_refiner=local_refiner,
        local_error=local_error,
        models=models,
        default_model=default_model,
    )
