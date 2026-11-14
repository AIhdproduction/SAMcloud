"""Sanitary blocks from point-cloud instances and instances.json in the clean-plan fingerprint."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from samcloud.ortho import plan
from samcloud.ortho.vector.raster_grid import RasterGrid


def _grid() -> RasterGrid:
    return RasterGrid(0.0, 5.0, 0.05, 100, 100)


def test_bathtub_instance_gets_bathtub_block_at_footprint() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[40:56, 20:55] = 7  # 0.8 m x 1.75 m
    results = plan._sanitary_instance_inserts(raster, {7: "bathtub"}, _grid(), [], 1.0)
    assert len(results) == 1
    insert, _mask, _slice, dominant = results[0]
    assert dominant == 7
    assert insert["name"].startswith("BATHTUB")
    assert insert["attributes"]["LABEL"] == "Badewanne"
    x, y = insert["point"]
    assert 0.9 < x < 2.8 and 2.0 < y < 3.1


def test_touching_bathtub_instances_merge_into_one_block() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[40:56, 20:38] = 47
    raster[40:56, 38:55] = 50
    results = plan._sanitary_instance_inserts(raster, {47: "bathtub", 50: "bathtub"}, _grid(), [], 1.0)
    assert len(results) == 1


def test_window_instance_never_gets_sanitary_block() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[40:56, 20:55] = 3
    assert plan._sanitary_instance_inserts(raster, {3: "window"}, _grid(), [], 1.0) == []
    assert plan._sanitary_instance_kind("window") is None
    assert plan._sanitary_instance_kind("Shower Tray") == "SHOWER"


def test_dominant_instance_nearby_sees_adjacent_window() -> None:
    raster = np.zeros((100, 100), dtype=np.uint32)
    raster[40:60, 30:34] = 3
    component = np.ones((10, 10), dtype=bool)
    component_slice = (slice(45, 55), slice(35, 45))
    assert plan._dominant_instance_nearby(raster, component_slice, component, 0.05, 0.15) == 3
    assert plan._dominant_instance_nearby(np.zeros_like(raster), component_slice, component, 0.05, 0.15) == 0


def test_fingerprint_tracks_instances_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(plan, "_available_clean_plan_settings", lambda project, level, settings: plan.CleanPlanSettings())
    monkeypatch.setattr(plan, "load_segment_result", lambda project, level, source: SimpleNamespace(fingerprint="seg"))
    without = plan.clean_plan_fingerprint(tmp_path, None)
    assert plan.clean_plan_fingerprint(tmp_path, None) == without
    target = tmp_path / "products" / "classified"
    target.mkdir(parents=True)
    (target / "instances.json").write_text('{"instances": []}', encoding="utf-8")
    first = plan.clean_plan_fingerprint(tmp_path, None)
    assert first != without
    (target / "instances.json").write_text('{"instances": [{"id": 1, "label": "bathtub"}]}', encoding="utf-8")
    assert plan.clean_plan_fingerprint(tmp_path, None) != first
