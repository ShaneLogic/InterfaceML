"""
Celery task definitions for long-running AI generation jobs.

This module provides async task wrappers around the interface generation
pipeline.  When Celery + Redis are available, tasks are dispatched to a
background worker.  When they are *not* available, the module exposes a
lightweight in-process fallback using ``concurrent.futures`` so the
rest of the app can use the same API surface without requiring Redis.

Usage
-----
Start the worker:

    celery -A interfaceml.web.tasks.celery_app worker --loglevel=info

Or use the in-process fallback (no Redis needed):

    from interfaceml.web.tasks import get_task_backend
    backend = get_task_backend()
    task_id = backend.submit_interface_generation(payload, ai_status, core_status, upload_folder)
    result  = backend.get_result(task_id)
"""

from __future__ import annotations

import logging
import os
import secrets
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Celery (optional dependency)
# ---------------------------------------------------------------------------
try:
    from celery import Celery
    from celery.result import AsyncResult

    CELERY_AVAILABLE = True
except Exception:
    Celery = None  # type: ignore
    AsyncResult = None  # type: ignore
    CELERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Celery app (created lazily)
# ---------------------------------------------------------------------------
_celery_app: Optional[Any] = None


def get_celery_app() -> Any:
    """Return (and lazily create) the Celery application."""
    global _celery_app
    if _celery_app is not None:
        return _celery_app
    if not CELERY_AVAILABLE:
        return None

    broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    _celery_app = Celery(
        "interfaceml",
        broker=broker,
        backend=backend,
    )
    _celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        result_expires=3600,  # 1 hour
        task_time_limit=600,  # 10 min hard limit
        task_soft_time_limit=540,  # 9 min soft limit
    )
    return _celery_app


# Expose for ``celery -A interfaceml.web.tasks.celery_app``
celery_app = get_celery_app()


# ---------------------------------------------------------------------------
# Core generation logic (shared between Celery & thread fallback)
# ---------------------------------------------------------------------------
def _run_interface_generation(
    payload: Dict[str, Any],
    ai_status: Any,
    core_status: Any,
    upload_folder: str,
) -> Dict[str, Any]:
    """Run the full interface generation pipeline synchronously.

    Returns a JSON-serialisable result dict matching the API schema.
    """
    from pathlib import Path

    from werkzeug.utils import secure_filename

    from interfaceml.ai.interface_generator import InterfaceAIGenerator

    base_filename = payload["base_filename"]
    safe_name = secure_filename(str(base_filename))
    upload_path = Path(upload_folder)
    base_path = (upload_path / safe_name).resolve()

    try:
        base_path.relative_to(upload_path.resolve())
    except ValueError:
        return {"status": "error", "error": "Invalid base_filename path"}

    if not base_path.exists():
        return {"status": "error", "error": "Base structure file not found"}

    try:
        base_structure = core_status.io.load_structure(base_path)
    except Exception as exc:
        return {"status": "error", "error": f"Failed to load base structure: {exc}"}

    local_refine = payload.get("local_refine", True)
    model_key = payload.get("model") or getattr(ai_status, 'default_model', 'egnn')
    selected_api = ai_status.get_api(model_key) if hasattr(ai_status, 'get_api') else ai_status.api
    generator = InterfaceAIGenerator(
        fullerene_api=selected_api,
        local_refiner=ai_status.local_refiner if local_refine else None,
    )

    start_time = time.time()
    try:
        results = generator.generate_interface(
            base_structure=base_structure,
            num_atoms=int(payload.get("num_atoms", 60)),
            num_samples=int(payload.get("num_samples", 1)),
            ddim=bool(payload.get("ddim", False)),
            miller=tuple(payload.get("miller", [0, 0, 1])),
            slab_thickness=float(payload.get("slab_thickness", 18.0)),
            vacuum=float(payload.get("vacuum", 20.0)),
            separation=float(payload.get("separation", 3.2)),
            supercell_xy=tuple(payload["supercell_xy"]) if payload.get("supercell_xy") else None,
            buffer=float(payload.get("buffer", 10.0)),
            xy_frac=tuple(payload.get("xy_frac", [0.5, 0.5])),
            termination=payload.get("termination"),
            layer_tol=float(payload.get("layer_tol", 1.5)),
            refine_scope=str(payload.get("refine_scope", "adsorbate")),
        )
    except Exception as exc:
        return {"status": "error", "error": f"Interface generation failed: {exc}"}

    generation_time = time.time() - start_time
    run_id = secrets.token_hex(6)

    structures_out = []
    for idx, result in enumerate(results):
        suffix = f"_{idx:03d}" if len(results) > 1 else ""
        output_filename = f"ai_interface_{run_id}{suffix}.vasp"
        output_path = upload_path / output_filename
        try:
            core_status.io.write_poscar(result.combined, output_path)
        except Exception as exc:
            logger.error("Failed to write sample %d: %s", idx, exc)
            continue

        structures_out.append({
            "sample_index": idx,
            "output_file": str(output_path),
            "download_url": f"/api/download/{output_filename}",
            "n_atoms": len(result.combined),
            "metadata": result.metadata,
        })

    if not structures_out:
        return {"status": "error", "error": "All samples failed to write"}

    first = structures_out[0]
    model_entry = ai_status.get_model_entry(model_key) if hasattr(ai_status, 'get_model_entry') else None
    return {
        "status": "success",
        "output_file": first["output_file"],
        "download_url": first["download_url"],
        "generation_time": round(generation_time, 2),
        "metadata": first["metadata"],
        "n_atoms": first["n_atoms"],
        "num_samples": len(structures_out),
        "samples": structures_out,
        "base_filename": safe_name,
        "local_refiner_error": getattr(ai_status, "local_error", None),
        "model": model_key,
        "model_label": model_entry.label if model_entry else model_key,
    }


# ---------------------------------------------------------------------------
# In-process thread-pool fallback (no Redis required)
# ---------------------------------------------------------------------------
@dataclass
class _TaskResult:
    status: str = "PENDING"  # PENDING | STARTED | SUCCESS | FAILURE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InProcessTaskBackend:
    """Thread-pool based task backend — no external dependencies."""

    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, _TaskResult] = {}
        self._lock = Lock()

    def submit_interface_generation(
        self,
        payload: Dict[str, Any],
        ai_status: Any,
        core_status: Any,
        upload_folder: str,
    ) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = _TaskResult(status="PENDING")

        def _worker():
            with self._lock:
                self._tasks[task_id].status = "STARTED"
            try:
                result = _run_interface_generation(payload, ai_status, core_status, upload_folder)
                with self._lock:
                    self._tasks[task_id].status = "SUCCESS"
                    self._tasks[task_id].result = result
            except Exception as exc:
                with self._lock:
                    self._tasks[task_id].status = "FAILURE"
                    self._tasks[task_id].error = traceback.format_exc()

        self._pool.submit(_worker)
        return task_id

    def get_result(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            tr = self._tasks.get(task_id)
        if tr is None:
            return {"task_id": task_id, "state": "NOT_FOUND"}
        out: Dict[str, Any] = {"task_id": task_id, "state": tr.status}
        if tr.status == "SUCCESS":
            out["result"] = tr.result
        elif tr.status == "FAILURE":
            out["error"] = tr.error
        return out


# ---------------------------------------------------------------------------
# Celery task backend
# ---------------------------------------------------------------------------
class CeleryTaskBackend:
    """Uses Celery + Redis for distributed task processing."""

    def submit_interface_generation(
        self,
        payload: Dict[str, Any],
        ai_status: Any,
        core_status: Any,
        upload_folder: str,
    ) -> str:
        # We cannot pickle ai_status / core_status, so the Celery task
        # must re-initialize them.  Pass only JSON-safe payload + paths.
        result = celery_generate_interface.delay(payload, upload_folder)
        return result.id

    def get_result(self, task_id: str) -> Dict[str, Any]:
        res = AsyncResult(task_id, app=get_celery_app())
        out: Dict[str, Any] = {"task_id": task_id, "state": res.state}
        if res.state == "SUCCESS":
            out["result"] = res.result
        elif res.state == "FAILURE":
            out["error"] = str(res.result)
        return out


# ---------------------------------------------------------------------------
# Celery task function (only registered if Celery is available)
# ---------------------------------------------------------------------------
if CELERY_AVAILABLE and celery_app is not None:

    @celery_app.task(bind=True, name="interfaceml.generate_interface")
    def celery_generate_interface(self, payload: Dict[str, Any], upload_folder: str):
        """Celery task: re-initialise AI modules and run generation."""
        from interfaceml.web.ai import load_fullerene_api
        from interfaceml.web.core import load_core_modules

        ai_status = load_fullerene_api()
        core_status = load_core_modules()
        if not ai_status.available:
            return {"status": "error", "error": ai_status.error or "AI unavailable in worker"}
        if not core_status.available:
            return {"status": "error", "error": "Core modules unavailable in worker"}

        return _run_interface_generation(payload, ai_status, core_status, upload_folder)
else:
    celery_generate_interface = None  # type: ignore


# ---------------------------------------------------------------------------
# Factory: choose the best available backend
# ---------------------------------------------------------------------------
_backend_instance: Optional[Any] = None


def get_task_backend() -> Any:
    """Return the task backend (Celery if available, else in-process threads)."""
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    if CELERY_AVAILABLE:
        # Probe Redis connectivity
        try:
            app = get_celery_app()
            conn = app.connection()
            conn.ensure_connection(max_retries=1, timeout=2)
            conn.close()
            _backend_instance = CeleryTaskBackend()
            logger.info("Task backend: Celery + Redis")
            return _backend_instance
        except Exception:
            logger.info("Redis not reachable — falling back to in-process thread pool")

    _backend_instance = InProcessTaskBackend()
    logger.info("Task backend: in-process thread pool (install celery + redis for distributed)")
    return _backend_instance
