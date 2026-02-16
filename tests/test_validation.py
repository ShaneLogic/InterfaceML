"""Tests for interfaceml.utils.validation module."""

import tempfile
from pathlib import Path

import pytest

from interfaceml.utils.validation import (
    ValidationError,
    validate_filepath,
    validate_miller_indices,
    validate_positive_number,
)


class TestValidateFilepath:
    """Test validate_filepath function."""

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.cif"
        f.write_text("data")
        result = validate_filepath(f, must_exist=True)
        assert result == f

    def test_nonexistent_file_raises(self):
        with pytest.raises(ValidationError, match="does not exist"):
            validate_filepath("/nonexistent/file.cif", must_exist=True)

    def test_nonexistent_file_allowed(self):
        result = validate_filepath("/nonexistent/file.cif", must_exist=False)
        assert isinstance(result, Path)

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        with pytest.raises(ValidationError, match="extension"):
            validate_filepath(f, must_exist=True, allowed_extensions=[".cif", ".vasp"])

    def test_correct_extension(self, tmp_path):
        f = tmp_path / "test.cif"
        f.write_text("data")
        result = validate_filepath(f, must_exist=True, allowed_extensions=[".cif", ".vasp"])
        assert result == f

    def test_string_path_converted(self, tmp_path):
        f = tmp_path / "test.vasp"
        f.write_text("data")
        result = validate_filepath(str(f), must_exist=True)
        assert isinstance(result, Path)


class TestValidateMillerIndices:
    """Test validate_miller_indices function."""

    def test_valid_indices(self):
        validate_miller_indices([1, 0, 0])
        validate_miller_indices([1, 1, 1])
        validate_miller_indices((0, 0, 1))

    def test_invalid_length(self):
        with pytest.raises(ValidationError):
            validate_miller_indices([1, 0])

    def test_all_zeros(self):
        with pytest.raises(ValidationError):
            validate_miller_indices([0, 0, 0])


class TestValidatePositiveNumber:
    """Test validate_positive_number function."""

    def test_valid_positive(self):
        assert validate_positive_number(5.0, "param") == 5.0

    def test_zero_not_allowed(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(0.0, "param")

    def test_zero_allowed(self):
        assert validate_positive_number(0.0, "param", allow_zero=True) == 0.0

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_positive_number(-1.0, "param")

    def test_nan_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            validate_positive_number(float('nan'), "param")
