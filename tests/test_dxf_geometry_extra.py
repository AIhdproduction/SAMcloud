"""Mandatory regression coverage for DXF geometry, cache, and writer behaviour."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import ezdxf
import laspy
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from dxf_synthetic import write_classified_las, write_indoor_building, write_outdoor_scene
from samcloud.dxf.cancel import CancelToken, Cancelled
from samcloud.dxf.config import load_dxf_config
from samcloud.dxf.layers import LayerConfig, LayerDefinition, load_layer_config
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import PlanSettings, _hash_fingerprint_inputs, _mode_points, cache_directory, generate_plan, is_stale, load_cached_result, save_cached_result
from samcloud.dxf.writer import export_plan


def _level() -> Level:
    return Level("level_1", "1", 0.0, 2.6, 0, 1.0, False)


def _line_length(entity: dict) -> float:
    return float(np.linalg.norm(np.asarray(entity["end"], dtype=float)[:2] - np.asarray(entity["start"], dtype=float)[:2]))


def _line_angle(entity: dict) -> float:
    direction = np.asarray(entity["end"], dtype=float)[:2] - np.asarray(entity["start"], dtype=float)[:2]
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)


def _wall_polygons(result: object) -> list[Polygon]:
    return [Polygon(entity["points"], entity.get("holes", [])) for entity in result.layers["A-WALL"]]


def _write_ring_ground(path: Path, *, fill_tree: bool = True) -> Path:
    grid_x, grid_y = np.meshgrid(np.arange(0.0, 12.01, 0.08), np.arange(0.0, 10.01, 0.08))
    ground = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)))
    hole = np.linalg.norm(ground[:, :2] - (6.0, 5.0), axis=1) < 1.7
    ground = ground[~hole]
    if not fill_tree:
        return write_classified_las(path, ground, np.full(len(ground), 2, dtype=np.uint8))
    angles = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    crown = np.column_stack((6.0 + 1.45 * np.cos(angles), 5.0 + 1.45 * np.sin(angles), np.full(len(angles), 4.5)))
    trunk = np.column_stack((np.full(60, 6.0), np.full(60, 5.0), np.linspace(0.0, 4.0, 60)))
    return write_classified_las(path, np.vstack((ground, crown, trunk)), np.concatenate((np.full(len(ground), 2, dtype=np.uint8), np.full(len(crown), 5, dtype=np.uint8), np.full(len(trunk), 95, dtype=np.uint8))))


def test_t3_centerline_and_polygon_walls_cover_outline_and_cuts(tmp_path: Path) -> None:
    source = write_indoor_building(tmp_path / "building.las")
    centerlines = generate_plan(source, "indoor", _level())
    walls = centerlines.layers["A-WALL"]
    assert len(walls) == 6
    horizontal = [item for item in walls if min(abs(_line_angle(item)), abs(_line_angle(item) - 180.0)) < 1.0]
    vertical = [item for item in walls if abs(_line_angle(item) - 90.0) < 1.0]
    assert sum(_line_length(item) for item in horizontal) == pytest.approx(20.0 - 2 * 1.2, abs=0.1)
    assert sum(_line_length(item) for item in vertical) == pytest.approx(12.0, abs=0.1)
    endpoints = np.asarray([point[:2] for item in walls for point in (item["start"], item["end"])])
    for corner in ((0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)):
        assert np.linalg.norm(endpoints - corner, axis=1).min() <= 0.02
    polygons = generate_plan(source, "indoor", _level(), PlanSettings(wall_centerlines=False))
    assert polygons.layers["A-WALL"]
    assert all(Polygon(item["points"]).area > 0.03 for item in polygons.layers["A-WALL"])
    assert not any(polygon.contains(Point(4.2, 0.0)) or polygon.contains(Point(6.2, 6.0)) for polygon in _wall_polygons(polygons))


def test_t3b_rotated_walls_follow_dominant_axes(tmp_path: Path) -> None:
    result = generate_plan(write_indoor_building(tmp_path / "rotated.las", z_rotation_degrees=30.0), "indoor", _level())
    angles = [_line_angle(item) for item in result.layers["A-WALL"]]
    assert angles
    assert all(min(abs(angle - 30.0), abs(angle - 120.0)) <= 1.0 for angle in angles)


def test_t4_opening_locations_symbols_and_two_windows(tmp_path: Path) -> None:
    result = generate_plan(write_indoor_building(tmp_path / "openings.las", two_windows=True), "indoor", _level())
    doors, windows = result.layers["A-DOOR"], result.layers["A-WIND"]
    assert len(doors) == 2
    assert len(windows) == 8
    jamb_x = sorted((line["start"][0] + line["end"][0]) / 2.0 for line in doors)
    assert np.mean(jamb_x) == pytest.approx(4.2, abs=0.1)
    assert jamb_x[1] - jamb_x[0] == pytest.approx(1.2, abs=0.1)
    assert all(abs(line["start"][1] - line["end"][1]) >= 0.15 for line in doors)
    assert all(abs((line["start"][1] + line["end"][1]) / 2.0) <= 0.05 for line in doors)
    window_centres = [np.mean(np.asarray([line["start"][:2], line["end"][:2]]), axis=0) for line in windows[2::4]]
    assert sorted(round(float(point[0]), 1) for point in window_centres) == [2.8, 6.2]
    wall_segments = result.layers["A-WALL"]
    north = [item for item in wall_segments if np.mean((item["start"][1], item["end"][1])) > 5.9]
    assert len(north) == 3


def test_t5_table_insert_has_class_height_and_footprint(tmp_path: Path) -> None:
    result = generate_plan(write_indoor_building(tmp_path / "table.las"), "indoor", _level())
    furniture = result.layers["A-FURN"]
    assert len(furniture) == 1
    insert = furniture[0]
    assert insert["attributes"]["class"] == "119"
    assert float(insert["attributes"]["height"]) == pytest.approx(0.75, abs=0.05)
    assert sorted((float(insert["xscale"]), float(insert["yscale"]))) == pytest.approx([0.8, 1.6], abs=0.1)


def test_t7_outdoor_layers_and_ground_are_independent_of_trees(tmp_path: Path) -> None:
    source = write_outdoor_scene(tmp_path / "scene.las")
    result = generate_plan(source, "outdoor", None)
    expected = ("S-GRND", "S-PAVE", "S-BLDG", "S-TREE", "S-VEH", "S-CURB", "S-MANH", "S-POLE", "S-SIGN", "S-FENC")
    assert all(result.statuses[name].entity_count > 0 for name in expected)
    comparison = generate_plan(write_outdoor_scene(tmp_path / "without_trees.las", include_tree=False), "outdoor", None)
    ground_area = sum(Polygon(item["points"], item.get("holes", [])).area for item in result.layers["S-GRND"])
    comparison_area = sum(Polygon(item["points"], item.get("holes", [])).area for item in comparison.layers["S-GRND"])
    assert ground_area == pytest.approx(comparison_area, rel=0.01)


def test_t7b_canopy_hole_is_retained_or_filled_by_setting(tmp_path: Path) -> None:
    source = _write_ring_ground(tmp_path / "canopy.las")
    empty = generate_plan(source, "outdoor", None, PlanSettings(fill_under_canopy=False))
    filled = generate_plan(source, "outdoor", None, PlanSettings(fill_under_canopy=True))
    assert any(len(entity.get("holes", [])) for entity in empty.layers["S-GRND"])
    assert all(not entity.get("holes", []) for entity in filled.layers["S-GRND"])


def test_t7c_l_shaped_marking_tracks_both_legs(tmp_path: Path) -> None:
    x1 = np.arange(0.0, 4.01, 0.04)
    y2 = np.arange(0.0, 3.01, 0.04)
    leg_one = np.column_stack((x1, np.zeros_like(x1), np.zeros_like(x1)))
    leg_two = np.column_stack((np.full_like(y2, 4.0), y2, np.zeros_like(y2)))
    result = generate_plan(write_classified_las(tmp_path / "mark.las", np.vstack((leg_one, leg_two)), np.full(len(leg_one) + len(leg_two), 82, dtype=np.uint8)), "outdoor", None)
    lines = result.layers["S-MARK"]
    assert lines
    vertices = np.asarray([vertex[:2] for entity in lines for vertex in entity["points"]])
    deviations = np.minimum(np.abs(vertices[:, 1]), np.abs(vertices[:, 0] - 4.0))
    assert deviations.max() <= 0.1
    assert sum(np.linalg.norm(np.diff(np.asarray(entity["points"]), axis=0), axis=1).sum() for entity in lines) == pytest.approx(7.0, abs=0.3)


def test_t7d_two_signs_and_trees_create_separate_instances(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    centres = np.asarray(((2.0, 2.0), (8.0, 6.0)))
    crowns = [np.column_stack((rng.normal(x, 0.5, 100), rng.normal(y, 0.5, 100), rng.normal(4.5, 0.3, 100))) for x, y in centres]
    trunks = [np.column_stack((rng.normal(x, 0.05, 40), rng.normal(y, 0.05, 40), np.linspace(0.0, 4.0, 40))) for x, y in centres]
    signs = [np.column_stack((rng.normal(x, 0.03, 20), rng.normal(y, 0.03, 20), rng.normal(2.0, 0.03, 20))) for x, y in ((1.0, 8.0), (10.0, 1.0))]
    xyz = np.vstack((*crowns, *trunks, *signs))
    codes = np.concatenate((np.full(200, 5, dtype=np.uint8), np.full(80, 95, dtype=np.uint8), np.full(40, 81, dtype=np.uint8)))
    result = generate_plan(write_classified_las(tmp_path / "instances.las", xyz, codes), "outdoor", None)
    trees = result.layers["S-TREE"]
    assert len([entity for entity in trees if entity["type"] == "circle"]) == 2
    assert len([entity for entity in trees if entity["type"] == "point"]) == 2
    assert len(result.layers["S-SIGN"]) == 2
    radii = [float(entity["radius"]) for entity in trees if entity["type"] == "circle"]
    assert all(0.35 <= radius <= 1.3 for radius in radii)


def test_t8_flurseg_empty_reasons_and_markings(tmp_path: Path) -> None:
    result = generate_plan(write_outdoor_scene(tmp_path / "flurseg.las", profile="flurseg"), "outdoor", None)
    for name in ("S-POLE", "S-SIGN", "S-FENC"):
        assert result.statuses[name].state == "empty"
        assert result.statuses[name].reason == "model profile flurseg_v1 does not separate this class"
    assert result.statuses["S-MARK"].entity_count > 0


def test_t9_layer_failure_isolated_and_openings_without_walls_are_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.dxf.pipeline as pipeline

    original = pipeline._layer_entities

    def broken(layer: LayerDefinition, *args: object, **kwargs: object) -> list[dict[str, object]]:
        if layer.name == "S-VEH":
            raise RuntimeError("forced vehicle geometry failure")
        return original(layer, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_layer_entities", broken)
    failed = generate_plan(write_outdoor_scene(tmp_path / "failed.las"), "outdoor", None)
    assert failed.statuses["S-VEH"].state == "failed"
    assert "forced vehicle geometry failure" in str(failed.statuses["S-VEH"].reason)
    assert failed.statuses["S-GRND"].state == "ok"
    assert export_plan(failed, tmp_path / "exports").is_file()
    source = write_indoor_building(tmp_path / "doors_only.las")
    cloud = laspy.read(source)
    cloud.classification[np.asarray(cloud.classification) == 100] = 101
    cloud.write(source)
    no_walls = generate_plan(source, "indoor", _level())
    assert no_walls.statuses["A-DOOR"].state == "empty"
    assert no_walls.statuses["A-DOOR"].reason


def test_t10_dxf_readback_hatch_coordinates_and_note(tmp_path: Path) -> None:
    result = generate_plan(write_outdoor_scene(tmp_path / "coordinates.las"), "outdoor", None)
    local = ezdxf.readfile(export_plan(result, tmp_path / "local", "local"))
    original = ezdxf.readfile(export_plan(result, tmp_path / "original", "original"))
    for name, status in result.statuses.items():
        assert name in local.layers
        count = sum(1 for entity in local.modelspace() if entity.dxf.layer == name and entity.dxftype() != "HATCH")
        assert count == status.entity_count
    assert any(entity.dxftype() == "HATCH" and entity.dxf.layer == "S-GRND" for entity in local.modelspace())
    local_line = next(entity for entity in local.modelspace() if entity.dxftype() == "LWPOLYLINE")
    original_line = next(entity for entity in original.modelspace() if entity.dxftype() == "LWPOLYLINE")
    local_xy = np.asarray(local_line.get_points("xy")[0])
    original_xy = np.asarray(original_line.get_points("xy")[0])
    assert original_xy - local_xy == pytest.approx(result.frame.origin[:2], abs=0.001)
    note = next(entity for entity in local.modelspace() if entity.dxf.layer == "X-NOTE")
    assert (note.dxf.insert.x, note.dxf.insert.y) == pytest.approx(result.background.bounds[:2], abs=0.001)


def test_t11_cache_loading_and_every_fingerprint_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.dxf.pipeline as pipeline

    source = write_indoor_building(tmp_path / "cache.las")
    result = generate_plan(source, "indoor", _level())
    directory = cache_directory(source, _level())
    save_cached_result(result, directory)
    monkeypatch.setattr(pipeline, "generate_plan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache loading regenerated")))
    loaded = load_cached_result(directory)
    assert not is_stale(loaded, PlanSettings())
    base_layers = load_layer_config()
    recolored = LayerConfig(tuple(replace(item, color=item.color + 1) if item.name == "A-WALL" else item for item in base_layers.layers))
    assert is_stale(loaded, PlanSettings(layer_config=recolored))
    configuration = load_dxf_config()
    changed_geometry = replace(configuration, geometry=replace(configuration.geometry, wall_merge_gap=configuration.geometry.wall_merge_gap + 0.01))
    assert is_stale(loaded, PlanSettings(config=changed_geometry))
    other_inputs = {**loaded.fingerprint_inputs, "profile": "other_profile"}
    profile_changed = replace(loaded, fingerprint_inputs=other_inputs, settings_fingerprint=_hash_fingerprint_inputs(other_inputs))
    assert is_stale(profile_changed, PlanSettings())
    assert is_stale(loaded, PlanSettings(wall_centerlines=False))
    assert is_stale(loaded, PlanSettings(cell_size=0.03))


def test_t12_progress_callback_can_cancel_generation(tmp_path: Path) -> None:
    token = CancelToken()

    def cancel_on_progress(_fraction: float, _stage: str) -> None:
        token.cancel()

    with pytest.raises(Cancelled):
        generate_plan(write_indoor_building(tmp_path / "cancel_mid_run.las"), "indoor", _level(), progress=cancel_on_progress, cancel=token)


def test_t13_audit_failure_preserves_existing_target_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.dxf.writer as writer

    result = generate_plan(write_indoor_building(tmp_path / "audit.las"), "indoor", _level())
    directory = tmp_path / "output"
    directory.mkdir()
    target = directory / f"{result.project_name}_{result.level_name}.dxf"
    target.write_bytes(b"old dxf content")
    original_audit = writer._audit
    calls = 0

    def failing_second_audit(document: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("synthetic audit error")
        original_audit(document)

    monkeypatch.setattr(writer, "_audit", failing_second_audit)
    with pytest.raises(RuntimeError, match="DXF export audit failed: synthetic audit error"):
        export_plan(result, directory)
    assert target.read_bytes() == b"old dxf content"
    assert not list(directory.glob("*.tmp"))


def test_t15_streaming_memory_does_not_combine_all_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import samcloud.dxf.pipeline as pipeline

    rng = np.random.default_rng(21)
    count = 200_000
    xyz = np.column_stack((rng.uniform(0.0, 5.0, count), rng.uniform(0.0, 5.0, count), rng.normal(0.0, 0.004, count)))
    source = write_classified_las(tmp_path / "large.las", xyz, np.full(count, 2, dtype=np.uint8))
    config = load_dxf_config()
    config = replace(config, loading=replace(config.loading, chunk_size=10_000))
    original_concatenate = pipeline.np.concatenate

    def guarded_concatenate(parts: object, *args: object, **kwargs: object) -> np.ndarray:
        values = list(parts)
        if sum(len(value) for value in values if hasattr(value, "__len__")) > 100_000:
            raise AssertionError("full-cloud concatenation attempted")
        return original_concatenate(values, *args, **kwargs)

    monkeypatch.setattr(pipeline.np, "concatenate", guarded_concatenate)
    result = generate_plan(source, "outdoor", None, PlanSettings(config=config))
    assert result.statuses["S-GRND"].state == "ok"


def test_t16_layer_modes_select_expected_points() -> None:
    level = _level()
    indoor_points = np.asarray(((0.01, 0.01, 0.2), (0.01, 0.01, 1.0), (0.01, 0.01, 2.4)))
    indoor_codes = np.full(3, 100, dtype=np.uint8)
    top_points = np.asarray(((1.01, 1.01, 1.0), (1.01, 1.01, 3.0)))
    top_codes = np.full(2, 5, dtype=np.uint8)
    ground_points = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (4.0, 4.0, 4.0)))
    ground_codes = np.asarray((2, 3, 4, 5, 95), dtype=np.uint8)
    cut = LayerDefinition("indoor", "cut_band", (100,), "X-CUT", 1, "CONTINUOUS", "polygon", (0.9, 1.3), 0.0, 0.0, False)
    footprint = replace(cut, mode="footprint", name="X-FOOT", z_band=None)
    top = LayerDefinition("outdoor", "top_surface", (5,), "X-TOP", 1, "CONTINUOUS", "polygon", None, 0.0, 0.0, False)
    ground = LayerDefinition("outdoor", "ground_surface", (2, 3, 4, 5, 95), "X-GROUND", 1, "CONTINUOUS", "polygon", None, 0.0, 0.0, False)
    selected_cut, _ = _mode_points(indoor_points, indoor_codes, cut, "indoor", level, 0.1)
    selected_footprint, _ = _mode_points(indoor_points, indoor_codes, footprint, "indoor", level, 0.1)
    selected_top, _ = _mode_points(top_points, top_codes, top, "outdoor", None, 0.1)
    selected_ground, selected_codes = _mode_points(ground_points, ground_codes, ground, "outdoor", None, 0.1)
    assert len(selected_cut) == 1
    assert len(selected_footprint) == 3
    assert len(selected_top) == 1 and selected_top[0, 2] == pytest.approx(3.0)
    assert len(selected_ground) == 1 and selected_codes.tolist() == [2]
