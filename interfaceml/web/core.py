"""
Core module loading helpers for the InterfaceML web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CoreStatus:
    """Status container for core module availability."""

    available: bool
    splitting_available: bool
    io: Optional[Any]
    layering: Optional[Any]
    splitting: Optional[Any]
    error: Optional[str]


def load_core_modules() -> CoreStatus:
    """Attempt to load core modules and return a status object."""
    try:
        from interfaceml.core import io, layering
        try:
            from interfaceml.core import splitting
            splitting_available = True
        except ImportError:
            splitting = None
            splitting_available = False
        return CoreStatus(
            available=True,
            splitting_available=splitting_available,
            io=io,
            layering=layering,
            splitting=splitting,
            error=None,
        )
    except ImportError as exc:
        return CoreStatus(
            available=False,
            splitting_available=False,
            io=None,
            layering=None,
            splitting=None,
            error=str(exc),
        )
