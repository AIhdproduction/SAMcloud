"""Tests for coordinate system validation."""

import pytest

from samcloud.pointcloud.coordinates import LOCAL_CRS, validate_crs


def test_validate_local_coordinate_system() -> None:
    assert validate_crs("local") == LOCAL_CRS


def test_invalid_coordinate_system_raises_error() -> None:
    with pytest.raises(Exception):
        validate_crs("not-a-coordinate-system")
