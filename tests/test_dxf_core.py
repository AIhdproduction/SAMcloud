"""Behaviour tests for the non-GUI DXF export core."""

from __future__ import annotations

import time

import laspy
import numpy as np
import pytest

from dxf_synthetic import write_indoor_building
from samcloud.dxf.alignment import estimate_alignment
from samcloud.dxf.cancel import CancelToken, Cancelled
from samcloud.dxf.config import DxfExportConfig, PlausibilityConfig, load_dxf_config
from samcloud.dxf.las_source import Frame, plausibility_warnings, read_las_info
from samcloud.dxf.levels import assign_stairs, detect_levels
from samcloud.dxf.preview import create_preview
from samcloud.dxf.scale import two_point_scale


def test_chunked_reader_matches_full_las(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "indoor.las")
    info = read_las_info(path, chunk_size=97)
    full = laspy.read(path)
    expected = {int(code): int(count) for code, count in zip(*np.unique(full.classification, return_counts=True))}
    assert info.point_count == len(full.points)
    assert info.class_histogram == expected


def test_lv95_local_coordinates_round_trip(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "lv95.las", origin_offset=(2_600_000.0, 1_200_000.0, 500.0))
    info = read_las_info(path, chunk_size=101)
    original = np.array([[2_600_004.2, 1_200_003.4, 501.2]])
    local = info.frame.to_local(original)
    assert float(np.abs(local).max()) < 20.0
    assert np.allclose(info.frame.to_original(local), original, atol=0.001)


def test_preview_respects_limit_and_keeps_rgb_and_classes(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "preview.las")
    info = read_las_info(path)
    preview = create_preview(path, info.frame, info=info, max_points=150, chunk_size=111)
    assert len(preview.xyz) <= 150
    assert preview.rgb is not None and preview.rgb.dtype == np.uint8
    assert preview.cls.dtype == np.uint8
    assert 102 in preview.cls
    source = laspy.read(path)
    source_rgb = np.column_stack((source.red, source.green, source.blue)).astype(np.uint16) // 257
    source_pairs = set(zip(np.asarray(source.classification, dtype=np.uint8), map(tuple, source_rgb), strict=True))
    preview_pairs = set(zip(preview.cls, map(tuple, preview.rgb), strict=True))
    assert preview_pairs <= source_pairs


def test_preview_downsamples_one_million_points_within_time_limit(tmp_path) -> None:
    path = tmp_path / "million-points.las"
    rng = np.random.default_rng(20261109)
    point_count = 1_000_000
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.full(3, 0.001)
    cloud = laspy.LasData(header)
    cloud.xyz = rng.uniform(0.0, 100.0, (point_count, 3))
    cloud.classification = rng.integers(1, 120, point_count, dtype=np.uint8)
    cloud.red = rng.integers(0, 65536, point_count, dtype=np.uint16)
    cloud.green = rng.integers(0, 65536, point_count, dtype=np.uint16)
    cloud.blue = rng.integers(0, 65536, point_count, dtype=np.uint16)
    cloud.write(path)
    info = read_las_info(path, chunk_size=100_000)
    started = time.perf_counter()
    preview = create_preview(path, info.frame, info=info, max_points=10_000, chunk_size=100_000)
    assert time.perf_counter() - started < 10.0
    assert len(preview.xyz) <= 10_000


def test_plausibility_can_flag_scaled_building(tmp_path) -> None:
    normal = read_las_info(write_indoor_building(tmp_path / "normal.las"))
    scaled = read_las_info(write_indoor_building(tmp_path / "scaled.las", scale=0.37))
    config = DxfExportConfig(plausibility=PlausibilityConfig(indoor_min_extent=5.0, indoor_max_extent=500.0))
    assert not plausibility_warnings(normal, config)
    assert plausibility_warnings(scaled, config)


def test_two_point_scale_restores_metric_size() -> None:
    state = two_point_scale(np.array([0.0, 0.0, 0.0]), np.array([3.7, 0.0, 0.0]), 10.0)
    frame = Frame(np.zeros(3), state.factor)
    scaled_distance = np.linalg.norm(frame.to_local(np.array([[0.0, 0.0, 0.0], [3.7, 0.0, 0.0]]))[1])
    assert state.factor == pytest.approx(10.0 / 3.7)
    assert scaled_distance == pytest.approx(10.0, abs=0.01)


def test_gravity_alignment_corrects_three_degree_tilt(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "tilted.las", tilt_degrees=3.0)
    cloud = laspy.read(path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    result = estimate_alignment(xyz, cloud.classification)
    corrected = xyz @ result.rotation.T
    floor = corrected[np.asarray(cloud.classification) == 102]
    _, _, vectors = np.linalg.svd(floor - floor.mean(axis=0), full_matrices=False)
    normal = vectors[-1]
    angle = np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
    assert result.tilt_degrees == pytest.approx(3.0, abs=0.3)
    assert angle < 0.2


def test_level_detection_finds_two_storeys_and_names_them(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "levels.las")
    cloud = laspy.read(path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    levels = detect_levels(xyz, cloud.classification)
    assert len(levels) == 2
    assert [level.name for level in levels] == ["1", "2"]
    assert [level.floor_z for level in levels] == pytest.approx([0.0, 3.0], abs=0.05)
    assert [level.ceiling_z for level in levels] == pytest.approx([2.6, 5.6], abs=0.05)
    floor_points = xyz[np.asarray(cloud.classification) == 102, 2]
    assert [level.point_count for level in levels] == [int(np.count_nonzero(np.abs(floor_points - level.floor_z) <= 0.05)) for level in levels]

    upper_floor = (np.asarray(cloud.classification) == 102) & (xyz[:, 2] > 1.5)
    extra_points = xyz[upper_floor][: int(np.count_nonzero(upper_floor) // 2)]
    extra_classes = np.full(len(extra_points), 102, dtype=np.uint8)
    upper_heavy_levels = detect_levels(np.vstack((xyz, extra_points)), np.concatenate((np.asarray(cloud.classification), extra_classes)))
    assert [level.name for level in upper_heavy_levels] == ["1", "2"]


def test_stairs_are_assigned_to_eg(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "stairs.las")
    cloud = laspy.read(path)
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
    levels = detect_levels(xyz, cloud.classification)
    stair_points = xyz[np.asarray(cloud.classification) == 105]
    levels_by_id = {level.id: level for level in levels}
    assigned_names = {levels_by_id[level_id].name for level_id in assign_stairs(levels, stair_points).values()}
    assert assigned_names == {"1"}


def test_level_fallback_without_floor_classes_is_flagged(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "fallback.las")
    cloud = laspy.read(path)
    classes = np.asarray(cloud.classification).copy()
    classes[classes == 102] = 1
    levels = detect_levels(np.column_stack((cloud.x, cloud.y, cloud.z)), classes)
    assert levels
    assert all(level.low_confidence and level.reason for level in levels)


def test_cancelled_reader_and_preview_raise(tmp_path) -> None:
    path = write_indoor_building(tmp_path / "cancel.las")
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        read_las_info(path, cancel=token)
    with pytest.raises(Cancelled):
        create_preview(path, Frame(np.zeros(3)), cancel=token)


def test_config_defaults_and_validation(tmp_path) -> None:
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("mode_detection:\n  indoor_share: 0.2\n", encoding="utf-8")
    loaded = load_dxf_config(minimal)
    assert loaded.preview.max_points == 5_000_000
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("loading:\n  chunk_size: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_size"):
        load_dxf_config(invalid)
