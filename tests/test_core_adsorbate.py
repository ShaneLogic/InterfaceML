"""Tests for interfaceml.core.adsorbate module.

Covers rotation_matrix_from_axis_angle, auto_supercell_xy,
prepare_adsorbate_layer, stack_structures, and build_adsorbate_interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from interfaceml.core.adsorbate import (
    auto_supercell_xy,
    build_adsorbate_interface,
    prepare_adsorbate_layer,
    rotation_matrix_from_axis_angle,
    stack_structures,
    SupercellChoice,
)

# ---------------------------------------------------------------------------
# Paths to real structures (skip gracefully if not present)
# ---------------------------------------------------------------------------
_STRUCTURES = Path(__file__).resolve().parents[1] / "structures"
_CIF_PEROVSKITE = _STRUCTURES / "perovskites" / "fapbi3-1.cif"
_CIF_FULLERENE = _STRUCTURES / "etl" / "c70-1.cif"

_has_perovskite = _CIF_PEROVSKITE.exists()
_has_fullerene = _CIF_FULLERENE.exists()


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def cubic_slab():
    """Simple 10x10x5 slab with 4 atoms."""
    lattice = Lattice.from_parameters(10, 10, 5, 90, 90, 90)
    return Structure(lattice, ["Si"] * 4,
                     [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]])


@pytest.fixture()
def small_molecule():
    """Small 'adsorbate' with 3 atoms in a big box."""
    lattice = Lattice.cubic(20.0)
    coords = [[10, 10, 10], [11.5, 10, 10], [10, 11.5, 10]]
    return Structure(lattice, ["C"] * 3, coords, coords_are_cartesian=True)


# ===================================================================
# TestRotationMatrix
# ===================================================================


class TestRotationMatrix:
    """rotation_matrix_from_axis_angle correctness."""

    def test_identity_zero_angle(self):
        R = rotation_matrix_from_axis_angle(np.array([0, 0, 1.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_degree_z(self):
        R = rotation_matrix_from_axis_angle(np.array([0, 0, 1.0]), np.pi / 2)
        # x-axis should map to y-axis
        result = R @ np.array([1.0, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

    def test_180_degree(self):
        R = rotation_matrix_from_axis_angle(np.array([0, 0, 1.0]), np.pi)
        # x should map to -x, y to -y
        result = R @ np.array([1.0, 0, 0])
        np.testing.assert_allclose(result, [-1, 0, 0], atol=1e-12)

    def test_orthogonality(self):
        """R^T R should be identity."""
        R = rotation_matrix_from_axis_angle(np.array([1.0, 1.0, 1.0]), 1.23)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_zero_axis_returns_identity(self):
        R = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 0.0]), 1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


# ===================================================================
# TestAutoSupercellXY
# ===================================================================


class TestAutoSupercellXY:
    """auto_supercell_xy returns valid supercell choices."""

    def test_return_type(self, cubic_slab, small_molecule):
        choice = auto_supercell_xy(cubic_slab, small_molecule)
        assert isinstance(choice, SupercellChoice)
        assert isinstance(choice.nx, int)
        assert isinstance(choice.ny, int)

    def test_min_supercell(self, cubic_slab, small_molecule):
        choice = auto_supercell_xy(cubic_slab, small_molecule, min_supercell=2)
        assert choice.nx >= 2
        assert choice.ny >= 2

    def test_max_supercell(self, cubic_slab, small_molecule):
        choice = auto_supercell_xy(cubic_slab, small_molecule, max_supercell=3)
        assert choice.nx <= 3
        assert choice.ny <= 3


# ===================================================================
# TestPrepareAdsorbateLayer
# ===================================================================


class TestPrepareAdsorbateLayer:
    """prepare_adsorbate_layer placement and rotation."""

    def test_center_placement(self, small_molecule, cubic_slab):
        result = prepare_adsorbate_layer(small_molecule, cubic_slab.lattice)
        # Centroid should be near the center of the a,b plane
        cart = np.array(result.cart_coords)
        centroid_xy = cart[:, :2].mean(axis=0)
        target_xy = 0.5 * cubic_slab.lattice.matrix[0][:2] + 0.5 * cubic_slab.lattice.matrix[1][:2]
        np.testing.assert_allclose(centroid_xy, target_xy, atol=1.0)

    def test_rotation_application(self, small_molecule, cubic_slab):
        R = rotation_matrix_from_axis_angle(np.array([0, 0, 1.0]), np.pi / 2)
        result = prepare_adsorbate_layer(small_molecule, cubic_slab.lattice, rotation=R)
        assert isinstance(result, Structure)
        assert len(result) == len(small_molecule)


# ===================================================================
# TestStackStructures
# ===================================================================


class TestStackStructures:
    """stack_structures returns correct types and atom counts."""

    def test_return_types(self, cubic_slab, small_molecule):
        bottom, top, combined = stack_structures(cubic_slab, small_molecule)
        assert isinstance(bottom, Structure)
        assert isinstance(top, Structure)
        assert isinstance(combined, Structure)

    def test_atom_count(self, cubic_slab, small_molecule):
        bottom, top, combined = stack_structures(cubic_slab, small_molecule)
        assert len(combined) == len(cubic_slab) + len(small_molecule)

    def test_z_ordering(self, cubic_slab, small_molecule):
        """Top layer should be above bottom layer after stacking."""
        bottom, top, combined = stack_structures(cubic_slab, small_molecule, separation=5.0)
        n = np.array([0, 0, 1.0])  # Normal for orthorhombic
        bottom_z = np.dot(np.array(bottom.cart_coords), n).max()
        top_z = np.dot(np.array(top.cart_coords), n).min()
        assert top_z > bottom_z - 1.0  # Top should be above bottom (with some tolerance)


# ===================================================================
# TestBuildAdsorbateInterface
# ===================================================================


class TestBuildAdsorbateInterface:
    """End-to-end build_adsorbate_interface with real files."""

    @pytest.mark.skipif(
        not (_has_perovskite and _has_fullerene),
        reason="Real structure files not found",
    )
    def test_end_to_end(self):
        base = Structure.from_file(str(_CIF_PEROVSKITE))
        adsorbate = Structure.from_file(str(_CIF_FULLERENE))
        bottom, top, combined, choice = build_adsorbate_interface(
            base, adsorbate,
            miller=(0, 0, 1),
            slab_thickness=10.0,
            vacuum=15.0,
            supercell_xy=(2, 2),
        )
        assert isinstance(combined, Structure)
        assert len(combined) > 0
        assert len(combined) == len(bottom) + len(top)
        assert isinstance(choice, SupercellChoice)
