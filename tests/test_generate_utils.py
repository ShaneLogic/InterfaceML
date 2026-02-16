"""Tests for fullerene_e3gen/generate.py utility functions.

Tests the geometric helpers (rescale, bond projection, nonbonded repulsion,
validity checker) without needing a trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

_POC_DIR = Path(__file__).resolve().parents[1] / "fullerene_e3gen"
if str(_POC_DIR) not in sys.path:
    sys.path.insert(0, str(_POC_DIR))

from generate import (
    rescale_to_unit_radius,
    compute_bond_stats,
    compute_radius_stats,
    apply_bond_projection,
    apply_nonbonded_repulsion,
    is_valid_structure,
    save_xyz,
)


class TestRescaleToUnitRadius:
    """Test rescale_to_unit_radius function."""

    def test_already_unit(self):
        torch.manual_seed(0)
        pos = torch.randn(20, 3)
        pos = pos / pos.norm(dim=1, keepdim=True)  # on unit sphere
        result = rescale_to_unit_radius(pos)
        centered = result - result.mean(dim=0)
        radii = centered.norm(dim=1)
        assert radii.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_scaled_up(self):
        torch.manual_seed(0)
        pos = torch.randn(30, 3) * 5.0  # large scale
        result = rescale_to_unit_radius(pos)
        centered = result - result.mean(dim=0)
        radii = centered.norm(dim=1)
        assert radii.mean().item() == pytest.approx(1.0, abs=1e-5)


class TestComputeBondStats:
    """Test compute_bond_stats function."""

    def test_regular_triangle(self):
        # Equilateral triangle with side = 1.0
        pos = torch.tensor([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.5, 0.866, 0.0]])
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                   [1, 0, 2, 1, 2, 0]])
        mean, std = compute_bond_stats(pos, edge_index)
        assert mean == pytest.approx(1.0, abs=0.01)
        assert std < 0.05

    def test_empty_edges(self):
        pos = torch.randn(5, 3)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        mean, std = compute_bond_stats(pos, edge_index)
        assert mean == 0.0
        assert std == 0.0


class TestComputeRadiusStats:
    """Test compute_radius_stats function."""

    def test_unit_sphere(self):
        torch.manual_seed(42)
        pos = torch.randn(100, 3)
        pos = pos / pos.norm(dim=1, keepdim=True)
        mean, std = compute_radius_stats(pos)
        assert mean == pytest.approx(1.0, abs=0.05)
        assert std < 0.1


class TestApplyBondProjection:
    """Test apply_bond_projection function."""

    def test_shrinks_long_bonds(self):
        # Two atoms far apart
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        result = apply_bond_projection(pos, edge_index, target_length=1.0, strength=0.5, iters=5)
        new_dist = (result[0] - result[1]).norm().item()
        assert new_dist < 3.0  # Should have moved closer


class TestApplyNonbondedRepulsion:
    """Test apply_nonbonded_repulsion function."""

    def test_pushes_overlapping_atoms(self):
        # Two non-bonded atoms very close
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # no bonds
        result = apply_nonbonded_repulsion(pos, edge_index, min_dist=1.0, strength=0.5, iters=3)
        new_dist = (result[0] - result[1]).norm().item()
        assert new_dist > 0.1  # Should have been pushed apart


class TestIsValidStructure:
    """Test is_valid_structure function."""

    def test_valid_sphere(self):
        torch.manual_seed(42)
        N = 60
        pos = torch.randn(N, 3)
        pos = pos / pos.norm(dim=1, keepdim=True)  # unit sphere

        # Build some plausible edges (nearest neighbors)
        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(float('inf'))
        _, idx = dists.topk(3, dim=1, largest=False)
        src = torch.arange(N).unsqueeze(1).expand_as(idx).reshape(-1)
        dst = idx.reshape(-1)
        edge_index = torch.stack([src, dst])

        ok, stats = is_valid_structure(
            pos, edge_index, N,
            bond_mean_tol=0.5,
            bond_std_max=0.5,
            radius_mean_tol=0.5,
            radius_std_max=0.5,
        )
        assert isinstance(ok, bool)
        assert isinstance(stats, dict)
        assert 'bond_mean' in stats
        assert 'radius_mean' in stats


class TestSaveXyz:
    """Test save_xyz function."""

    def test_writes_valid_xyz(self, tmp_path):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [0.71, 1.23, 0.0]])
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                   [1, 0, 2, 1, 2, 0]])
        filepath = tmp_path / "test.xyz"
        save_xyz(pos, edge_index, filepath)

        content = filepath.read_text()
        lines = content.strip().split('\n')
        assert lines[0] == '3'  # num atoms
        assert len(lines) == 5  # header + comment + 3 atoms
        assert lines[2].startswith('C ')  # carbon atoms
