"""Tests for interfaceml.utils.performance module."""

import time

from interfaceml.utils.performance import (
    timer,
    estimate_memory_usage,
    suggest_batch_size,
)


class TestTimer:
    """Test timer context manager."""

    def test_returns_elapsed(self):
        with timer("test", verbose=False) as result:
            time.sleep(0.05)
        assert result['elapsed'] >= 0.04

    def test_zero_elapsed(self):
        with timer("test", verbose=False) as result:
            pass
        assert result['elapsed'] >= 0.0


class TestSuggestBatchSize:
    """Test suggest_batch_size function."""

    def test_small_structures(self):
        batch = suggest_batch_size(100, avg_atoms_per_structure=10)
        assert batch >= 1
        assert batch <= 100

    def test_large_structures(self):
        batch = suggest_batch_size(100, avg_atoms_per_structure=10000, available_memory_mb=1.0)
        assert batch >= 1

    def test_single_structure(self):
        batch = suggest_batch_size(1, avg_atoms_per_structure=100)
        assert batch == 1
