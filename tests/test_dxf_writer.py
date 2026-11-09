"""Result cache and audited DXF writer behaviour."""

from __future__ import annotations

from pathlib import Path

import ezdxf

from dxf_synthetic import write_indoor_building
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import PlanSettings, cache_directory, generate_plan, is_stale, load_cached_result, save_cached_result
from samcloud.dxf.writer import export_plan


def _result(tmp_path: Path):
    level = Level("level_1", "EG", 0.0, 2.6, 0, 1.0, False)
    return generate_plan(write_indoor_building(tmp_path / "building.las", origin_offset=(2_600_000.0, 1_200_000.0, 500.0)), "indoor", level)


def test_cache_round_trip_and_fingerprint_comparison(tmp_path: Path) -> None:
    result = _result(tmp_path)
    directory = cache_directory(tmp_path / "building.las", Level("level_1", "EG", 0.0, 2.6, 0, 1.0, False))
    save_cached_result(result, directory)
    loaded = load_cached_result(directory)
    assert loaded.layers == result.layers
    assert loaded.settings_fingerprint == result.settings_fingerprint
    assert not is_stale(loaded, result.settings_fingerprint)
    assert not is_stale(loaded, PlanSettings())
    assert is_stale(loaded, PlanSettings(cell_size=0.05))


def test_dxf_is_r2018_metres_and_coordinate_option_is_applied(tmp_path: Path) -> None:
    result = _result(tmp_path)
    local_path = export_plan(result, tmp_path / "local", "local")
    original_path = export_plan(result, tmp_path / "original", "original")
    local = ezdxf.readfile(local_path)
    original = ezdxf.readfile(original_path)
    assert local.dxfversion == "AC1032"
    assert local.header["$INSUNITS"] == 6
    assert not local.audit().has_errors
    assert "A-WALL" in local.layers
    local_line = next(entity for entity in local.modelspace() if entity.dxftype() == "LINE")
    original_line = next(entity for entity in original.modelspace() if entity.dxftype() == "LINE")
    assert original_line.dxf.start.x - local_line.dxf.start.x > 2_000_000
    assert not list((tmp_path / "local").glob("*.tmp"))
