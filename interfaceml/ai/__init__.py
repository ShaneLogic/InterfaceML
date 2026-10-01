"""
AI utilities for InterfaceML.

This package hosts optional ML-based generators and refiners that are
used by the web and CLI interfaces when available.
"""

from interfaceml.ai.interface_generator import InterfaceAIGenerator, InterfaceGenerationResult
from interfaceml.ai.local_gnn import LocalRefiner, LocalRefinerConfig

__all__ = [
    "InterfaceAIGenerator",
    "InterfaceGenerationResult",
    "LocalRefiner",
    "LocalRefinerConfig",
]
