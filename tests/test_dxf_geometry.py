"""Synthetic geometry coverage for the DXF plan pipeline."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pytest

from dxf_synthetic import write_indoor_building, write_outdoor_scene
from samcloud.dxf.cancel import CancelToken, Cancelled
from samcloud.dxf.geometry.raster import polygons_from_points
from samcloud.dxf.layers import load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import generate_plan


def _eg() -> Level:
    return Level("level_1", "EG", 0.0, 2.6, 0, 1.0, False)


def test_layer_configuration_validates_codes_and_geometry_mode(tmp_path: Path) -> None:
    configured = load_layer_config()
    assert configured.by_name("A-WALL").codes == (100,)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("layers:\n  - {mode: indoor, codes: [999], name: X, color: 1, linetype: CONTINUOUS, geometry: polygon, min_area: 0, simplify_tolerance: 0, hatch: false}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown Cyclone"):
        load_layer_config(unknown)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("layers:\n  - {mode: indoor, codes: [100], name: X, color: 1, linetype: CONTINUOUS, geometry: impossible, min_area: 0, simplify_tolerance: 0, hatch: false}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid geometry"):
        load_layer_config(invalid)


def test_indoor_wall_openings_and_furniture(tmp_path: Path) -> None:
    result = generate_plan(write_indoor_building(tmp_path / "indoor.las"), "indoor", _eg())
    walls = result.layers["A-WALL"]
    endpoints = np.asarray([point for wall in walls for point in (wall["start"], wall["end"])])
    assert len(walls) == 6
    assert endpoints[:, 0].min() == pytest.approx(0.0, abs=0.05)
    assert endpoints[:, 0].max() == pytest.approx(10.0, abs=0.05)
    assert endpoints[:, 1].min() == pytest.approx(0.0, abs=0.05)
    assert endpoints[:, 1].max() == pytest.approx(6.0, abs=0.05)
    assert len(result.layers["A-DOOR"]) == 2
    assert len(result.layers["A-WIND"]) == 4
    furniture = result.layers["A-FURN"]
    assert len(furniture) == 1
    assert furniture[0]["attributes"]["class"] == "119"
    assert float(furniture[0]["xscale"]) * float(furniture[0]["yscale"]) == pytest.approx(1.28, rel=0.25)


def test_ceiling_code_does_not_create_wall_geometry(tmp_path: Path) -> None:
    source = write_indoor_building(tmp_path / "ceiling.las")
    cloud = laspy.read(source)
    cloud.classification[np.asarray(cloud.classification) == 100] = 101
    cloud.write(source)
    result = generate_plan(source, "indoor", _eg())
    assert result.statuses["A-WALL"].state == "empty"
    assert result.layers["A-WALL"] == []


def test_polygons_preserve_holes() -> None:
    x, y = np.meshgrid(np.linspace(0.0, 4.0, 81), np.linspace(0.0, 4.0, 81))
    points = np.column_stack((x.ravel(), y.ravel()))
    keep = ~((points[:, 0] > 1.0) & (points[:, 0] < 3.0) & (points[:, 1] > 1.0) & (points[:, 1] < 3.0))
    polygons = polygons_from_points(points[keep], 0.05, simplify_tolerance=0.01)
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1


def test_outdoor_layers_and_profile_empty_reasons(tmp_path: Path) -> None:
    result = generate_plan(write_outdoor_scene(tmp_path / "sam3.las"), "outdoor", None)
    for name in ("S-GRND", "S-PAVE", "S-BLDG", "S-TREE", "S-VEH", "S-CURB", "S-MANH", "S-POLE", "S-SIGN", "S-FENC"):
        assert result.statuses[name].entity_count > 0
    flurseg = generate_plan(write_outdoor_scene(tmp_path / "flurseg.las", profile="flurseg"), "outdoor", None)
    for name in ("S-POLE", "S-SIGN", "S-FENC"):
        assert flurseg.statuses[name].reason == "model profile flurseg_v1 does not separate this class"


def test_cancelled_generation_raises(tmp_path: Path) -> None:
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        generate_plan(write_indoor_building(tmp_path / "cancel.las"), "indoor", _eg(), cancel=token)
