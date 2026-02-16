"""Tests for interfaceml.core.io module.

Covers load_structure (CIF/VASP/XYZ auto-detection), write_poscar,
read_cp2k_xyz_last_frame, and get_element_symbols.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from interfaceml.core.io import (
    get_element_symbols,
    load_structure,
    read_cp2k_xyz_last_frame,
    write_poscar,
)

# ---------------------------------------------------------------------------
# Paths to real test structures (skip gracefully if not present)
# ---------------------------------------------------------------------------
_STRUCTURES = Path(__file__).resolve().parents[1] / "structures"
_CIF_PEROVSKITE = _STRUCTURES / "perovskites" / "fapbi3-1.cif"
_CIF_FULLERENE = _STRUCTURES / "etl" / "c70-1.cif"
_XYZ_CP2K = _STRUCTURES / "heterojunctions" / "cp2k_opt.xyz"

_has_perovskite = _CIF_PEROVSKITE.exists()
_has_fullerene = _CIF_FULLERENE.exists()
_has_cp2k_xyz = _XYZ_CP2K.exists()


# ===================================================================
# TestLoadStructure
# ===================================================================


class TestLoadStructure:
    """Auto-detection of CIF, VASP, and XYZ formats."""

    @pytest.mark.skipif(not _has_perovskite, reason="fapbi3-1.cif not found")
    def test_load_cif(self):
        s = load_structure(_CIF_PEROVSKITE)
        assert isinstance(s, Structure)
        assert len(s) > 0

    @pytest.mark.skipif(not _has_perovskite, reason="fapbi3-1.cif not found")
    def test_cif_has_lattice(self):
        s = load_structure(_CIF_PEROVSKITE)
        assert s.lattice.a > 0
        assert s.lattice.b > 0
        assert s.lattice.c > 0

    @pytest.mark.skipif(not _has_cp2k_xyz, reason="cp2k_opt.xyz not found")
    def test_load_xyz(self):
        s = load_structure(_XYZ_CP2K)
        assert isinstance(s, Structure)
        assert len(s) > 0

    def test_load_xyz_without_cell(self, tmp_path):
        """XYZ file with no Tv_ vectors gets a fallback box."""
        xyz = tmp_path / "mol.xyz"
        xyz.write_text("3\ncomment\nC 0.0 0.0 0.0\nC 1.5 0.0 0.0\nC 0.0 1.5 0.0\n")
        s = load_structure(xyz)
        assert isinstance(s, Structure)
        assert s.lattice.a >= 10.0  # fallback padding

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "struct.pdb"
        bad.write_text("dummy")
        with pytest.raises(Exception):
            load_structure(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_structure(tmp_path / "nonexistent.cif")


# ===================================================================
# TestWritePoscar
# ===================================================================


class TestWritePoscar:
    """write_poscar round-trip and formatting tests."""

    @pytest.fixture()
    def simple_structure(self):
        lattice = Lattice.cubic(5.0)
        return Structure(lattice, ["Si", "O", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])

    def test_round_trip(self, simple_structure, tmp_path):
        out = tmp_path / "POSCAR"
        write_poscar(simple_structure, out)
        reloaded = Structure.from_file(str(out))
        assert len(reloaded) == len(simple_structure)

    def test_selective_dynamics(self, simple_structure, tmp_path):
        out = tmp_path / "POSCAR"
        sd = [(True, True, False)] * len(simple_structure)
        write_poscar(simple_structure, out, selective_dynamics=sd)
        text = out.read_text()
        assert "Selective" in text or "selective" in text.lower()

    def test_custom_comment(self, simple_structure, tmp_path):
        out = tmp_path / "POSCAR"
        write_poscar(simple_structure, out, comment="test_comment_123")
        first_line = out.read_text().splitlines()[0]
        assert "test_comment_123" in first_line

    def test_element_grouping(self, tmp_path):
        """Elements are grouped by species in the output."""
        lattice = Lattice.cubic(5.0)
        # Interleaved species: Si, O, Si, O
        s = Structure(lattice, ["Si", "O", "Si", "O"],
                      [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
        out = tmp_path / "POSCAR"
        write_poscar(s, out)
        text = out.read_text()
        # The elements line should list each element once
        lines = text.splitlines()
        # In VASP POSCAR, line 6 (0-indexed 5) has element symbols
        elem_line = lines[5].split()
        assert elem_line == ["Si", "O"]


# ===================================================================
# TestReadCp2kXyz
# ===================================================================


class TestReadCp2kXyz:
    """read_cp2k_xyz_last_frame for CP2K trajectory files."""

    @pytest.mark.skipif(not _has_cp2k_xyz, reason="cp2k_opt.xyz not found")
    def test_last_frame_reading(self):
        symbols, coords, cell = read_cp2k_xyz_last_frame(_XYZ_CP2K)
        assert len(symbols) > 0
        assert coords.shape[1] == 3
        assert len(symbols) == coords.shape[0]

    @pytest.mark.skipif(not _has_cp2k_xyz, reason="cp2k_opt.xyz not found")
    def test_coords_finite(self):
        _, coords, _ = read_cp2k_xyz_last_frame(_XYZ_CP2K)
        assert np.all(np.isfinite(coords))

    def test_malformed_file_raises(self, tmp_path):
        bad = tmp_path / "bad.xyz"
        bad.write_text("not an xyz file\n")
        with pytest.raises(ValueError):
            read_cp2k_xyz_last_frame(bad)

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.xyz"
        empty.write_text("")
        with pytest.raises(ValueError):
            read_cp2k_xyz_last_frame(empty)


# ===================================================================
# TestGetElementSymbols
# ===================================================================


class TestGetElementSymbols:
    """get_element_symbols returns correct list of symbols."""

    def test_returns_list(self):
        s = Structure(Lattice.cubic(5.0), ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        result = get_element_symbols(s)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_string_types(self):
        s = Structure(Lattice.cubic(5.0), ["Pb", "I", "C"], [[0, 0, 0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])
        result = get_element_symbols(s)
        assert all(isinstance(sym, str) for sym in result)

    def test_known_elements(self):
        s = Structure(Lattice.cubic(5.0), ["Pb", "I", "C"], [[0, 0, 0], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])
        result = get_element_symbols(s)
        assert result == ["Pb", "I", "C"]
