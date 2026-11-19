"""Tolerance profiles (MASTERPROMPT section 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from samcloud.bim.core.model import ElementKind
from samcloud.bim.core.qa import (
    TOLERANCES_PATH,
    CeilingSagReference,
    ToleranceProfile,
    load_tolerances,
)


def test_tls_defaults_match_section_7() -> None:
    tls = load_tolerances(profile="tls")
    assert tls.name == "tls"
    assert not tls.provisional
    assert tls.planar_snap_tol == 0.02
    assert tls.ceiling_sag_tol == 0.04
    assert tls.opening_dimension_target == 0.02
    assert tls.coverage_radius == 0.05
    assert tls.coverage_min == 0.3
    assert tls.snap_quantile == 0.99
    assert tls.noise_warning_factor == 2.0
    assert tls.ceiling_sag_reference is CeilingSagReference.FITTED
    assert tls.wall_default_thickness.interior == 0.12
    assert tls.wall_default_thickness.exterior == 0.30


def test_element_tolerances_start_from_snap_and_opening_values() -> None:
    tls = load_tolerances(profile="tls")
    for kind in (ElementKind.WALL, ElementKind.SLAB, ElementKind.CEILING):
        assert tls.tolerance_for(kind) == tls.planar_snap_tol
    for kind in (ElementKind.OPENING, ElementKind.WINDOW, ElementKind.DOOR):
        assert tls.tolerance_for(kind) == tls.opening_dimension_target
    assert tls.tolerance_for("floor") == tls.tolerance_for(ElementKind.SLAB)
    assert tls.tolerance_for("wall") == 0.02


def test_unknown_kind_and_profile_raise() -> None:
    tls = load_tolerances()
    with pytest.raises(ValueError, match="no deviation tolerance"):
        tls.tolerance_for(ElementKind.STAIR)
    with pytest.raises(ValueError, match="no deviation tolerance"):
        tls.tolerance_for("nonsense")
    with pytest.raises(KeyError, match="nonsense"):
        load_tolerances(profile="nonsense")


@pytest.mark.parametrize("name", ["slam", "photogrammetry"])
def test_relaxed_profiles_are_provisional_and_looser(name: str) -> None:
    tls = load_tolerances(profile="tls")
    profile = load_tolerances(profile=name)
    assert profile.provisional
    assert "to be confirmed" in profile.note.lower()
    assert profile.planar_snap_tol > tls.planar_snap_tol
    assert profile.ceiling_sag_tol > tls.ceiling_sag_tol
    assert profile.opening_dimension_target > tls.opening_dimension_target
    assert profile.coverage_radius > tls.coverage_radius
    assert profile.coverage_min <= tls.coverage_min
    slam = load_tolerances(profile="slam")
    photo = load_tolerances(profile="photogrammetry")
    assert photo.planar_snap_tol >= slam.planar_snap_tol


def test_colors_are_green_red_blue_and_configurable(tmp_path: Path) -> None:
    tls = load_tolerances()
    palette = tls.colors.palette()
    assert palette.dtype == np.uint8
    assert palette.shape == (3, 3)
    assert palette[0, 1] > palette[0, 0] and palette[0, 1] > palette[0, 2]  # green
    assert palette[1, 0] > palette[1, 1] and palette[1, 0] > palette[1, 2]  # red
    assert palette[2, 2] > palette[2, 0] and palette[2, 2] > palette[2, 1]  # blue

    source = TOLERANCES_PATH
    text = source.read_text(encoding="utf-8")
    assert "within: [46, 160, 67]" in text
    custom = tmp_path / "t.yaml"
    custom.write_text(text.replace("within: [46, 160, 67]", "within: [1, 2, 3]", 1), "utf-8")
    assert load_tolerances(custom).colors.within == (1, 2, 3)


def test_invalid_values_are_rejected() -> None:
    base = load_tolerances().model_dump(mode="json")
    base["planar_snap_tol"] = -0.01
    with pytest.raises(ValidationError):
        ToleranceProfile.model_validate(base)
    base = load_tolerances().model_dump(mode="json")
    base["unknown_key"] = 1
    with pytest.raises(ValidationError):
        ToleranceProfile.model_validate(base)
