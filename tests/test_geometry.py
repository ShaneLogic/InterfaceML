"""Tests for interfaceml.utils.geometry module."""

import numpy as np
import pytest

from interfaceml.utils.geometry import (
    angle_between,
    normalize_vector,
    compute_distance,
)


class TestAngleBetween:
    """Test angle_between function."""

    def test_parallel_vectors(self):
        assert angle_between(np.array([1, 0, 0]), np.array([2, 0, 0])) == pytest.approx(0.0)

    def test_antiparallel_vectors(self):
        assert angle_between(np.array([1, 0, 0]), np.array([-1, 0, 0])) == pytest.approx(180.0)

    def test_perpendicular_vectors(self):
        assert angle_between(np.array([1, 0, 0]), np.array([0, 1, 0])) == pytest.approx(90.0)

    def test_45_degrees(self):
        assert angle_between(np.array([1, 0, 0]), np.array([1, 1, 0])) == pytest.approx(45.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        assert angle_between(np.array([0, 0, 0]), np.array([1, 0, 0])) == 0.0


class TestNormalizeVector:
    """Test normalize_vector function."""

    def test_unit_vector(self):
        result = normalize_vector(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_non_unit_vector(self):
        result = normalize_vector(np.array([3.0, 4.0, 0.0]))
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0])
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        result = normalize_vector(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_large_vector(self):
        result = normalize_vector(np.array([1e6, 0.0, 0.0]))
        assert np.linalg.norm(result) == pytest.approx(1.0)


class TestComputeDistance:
    """Test compute_distance function."""

    def test_same_point(self):
        assert compute_distance(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_unit_distance(self):
        assert compute_distance(np.array([0, 0, 0]), np.array([1, 0, 0])) == pytest.approx(1.0)

    def test_3d_distance(self):
        # Distance along space diagonal of unit cube = sqrt(3)
        d = compute_distance(np.array([0, 0, 0]), np.array([1, 1, 1]))
        assert d == pytest.approx(np.sqrt(3))
