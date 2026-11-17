"""Tests for clean DXF plan generation from true orthophoto and segmentation (Part A: outdoor)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import uuid

import cv2
import ezdxf
import numpy as np
import pytest
from shapely import affinity, coverage_is_valid
from shapely.geometry import LineString, Polygon
import yaml

from samcloud.dxf.cancel import CancelToken, Cancelled
from samcloud.dxf.las_source import Frame
from samcloud.dxf.levels import Level
from samcloud.dxf.pipeline import load_cached_result, save_cached_result
from samcloud.dxf.overlay import render_dxf_rgba
from samcloud.dxf.result import BackgroundImage, DxfPlanResult, LayerStatus
from samcloud.dxf.writer import export_plan
from samcloud.engine.ply_io import write_classified_ply
from samcloud.ortho.build import OrthoError, ortho_output_dir
from samcloud.ortho.vector.raster_grid import RasterGrid
from samcloud.ortho.plan import (
    CleanPlanSettings,
    clean_plan_cache_dir,
    clean_plan_fingerprint,
    generate_clean_plan,
    indoor_wall_metrics,
    load_clean_plan_settings,
    _filter_wall_stubs,
    _merge_parallel_wall_rectangles,
    _window_is_exterior,
)
from samcloud.ortho.vector.indoor import Opening


@pytest.fixture(scope="module")
def work_dir(request: pytest.FixtureRequest) -> Path:
    """Return an isolated directory for this test session."""
    configured_root = os.environ.get("SAMCLOUD_TEST_WORKDIR")
    if configured_root:
        path = Path(configured_root) / f"ortho_plan_{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    return request.getfixturevalue("tmp_path_factory").mktemp("ortho_plan")


def _build_synthetic_scene(project_dir: Path) -> dict[str, object]:
    """Create a fully populated outdoor orthophoto, segmentation and point cloud dataset."""
    project_dir.mkdir(parents=True, exist_ok=True)
    ortho_dir = ortho_output_dir(project_dir, None)
    ortho_dir.mkdir(parents=True, exist_ok=True)

    width = 200
    height = 200
    pixel_size = 0.1
    min_x, min_y, max_x, max_y = 0.0, 0.0, 20.0, 20.0

    frame = Frame(origin=np.array([1000.0, 2000.0, 0.0]), scale_factor=1.0, rotation=np.eye(3))

    # Rasters
    classes = np.zeros((height, width), dtype=np.uint8)
    dtm = np.zeros((height, width), dtype=np.float32)
    dsm = np.zeros((height, width), dtype=np.float32)

    # 2 m slope from y = 0 to y = 20 (z from 100.0 to 102.0)
    for row in range(height):
        y = max_y - (row + 0.5) * pixel_size
        z_val = 100.0 + 0.1 * y
        dtm[row, :] = z_val
        dsm[row, :] = z_val

    # Road band: y in [0, 5] -> rows 150 to 200
    # Sidewalk band: y in [5, 7.5] -> rows 125 to 150
    # Lawn: y in [7.5, 20] -> rows 0 to 125
    classes[150:200, :] = 11   # S-PAVE (road)
    classes[125:150, :] = 73   # S-WALK (sidewalk)
    classes[0:125, :] = 3      # S-VEG (lawn)

    # Rotated rectangular building inside the lawn (center x=10, y=14, size 6x4, angle 20 deg)
    rect_poly = affinity.rotate(
        Polygon([(7.0, 12.0), (13.0, 12.0), (13.0, 16.0), (7.0, 16.0)]),
        20.0,
        origin=(10.0, 14.0),
    )

    rng = np.random.default_rng(20261109)
    building_mask = np.zeros((height, width), dtype=bool)
    for row in range(height):
        for col in range(width):
            x = min_x + (col + 0.5) * pixel_size
            y = max_y - (row + 0.5) * pixel_size
            p = Polygon([(x - 0.04, y - 0.04), (x + 0.04, y - 0.04), (x + 0.04, y + 0.04), (x - 0.04, y + 0.04)])
            if rect_poly.intersects(p):
                building_mask[row, col] = True

    # Add noisy edge pixels
    noise_coords = rng.integers(0, 200, size=(40, 2))
    for r, c in noise_coords:
        if building_mask[r, c] and rng.random() < 0.2:
            building_mask[r, c] = False

    classes[building_mask] = 6  # S-BLDG
    dsm[building_mask] = dtm[building_mask] + 3.5

    # Two trees at (4.0, 16.0) and (16.0, 16.0) with Gaussian CHM bumps
    tree_centers = [(4.0, 16.0), (16.0, 16.0)]
    for tx, ty in tree_centers:
        for r in range(height):
            for c in range(width):
                x = min_x + (c + 0.5) * pixel_size
                y = max_y - (r + 0.5) * pixel_size
                dist = math.hypot(x - tx, y - ty)
                if dist <= 1.5:
                    classes[r, c] = 5  # S-TREE
                    bump = 4.0 * math.exp(-(dist**2) / (2 * 0.5**2))
                    dsm[r, c] = dtm[r, c] + bump

    # Manhole at (10.0, 2.5) on road
    for r in range(height):
        for c in range(width):
            x = min_x + (c + 0.5) * pixel_size
            y = max_y - (r + 0.5) * pixel_size
            if math.hypot(x - 10.0, y - 2.5) <= 0.4:
                classes[r, c] = 75  # S-MANH

    # Write ortho artifacts
    valid = np.ones((height, width), dtype=np.uint8) * 255
    view_count = np.ones((height, width), dtype=np.uint8) * 3
    rgba = np.full((height, width, 4), 200, dtype=np.uint8)
    rgba[..., 3] = 255

    cv2.imwrite(str(ortho_dir / "ortho.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(str(ortho_dir / "valid.png"), valid)
    cv2.imwrite(str(ortho_dir / "view_count.png"), view_count)
    np.save(ortho_dir / "dsm.npy", dsm)
    np.save(ortho_dir / "dtm.npy", dtm)
    np.save(ortho_dir / "classes_pc.npy", classes)

    ortho_metadata = {
        "bounds": [min_x, min_y, max_x, max_y],
        "pixel_size": pixel_size,
        "width": width,
        "height": height,
        "frame": {
            "origin": frame.origin.tolist(),
            "scale": frame.scale_factor,
            "rotation": frame.rotation.tolist(),
        },
        "mode": "outdoor",
        "level": None,
        "source_paths": {
            "point_source": str(project_dir / "products" / "classified" / "classified.ply"),
        },
    }
    (ortho_dir / "ortho.json").write_text(json.dumps(ortho_metadata, indent=2), encoding="utf-8")
    (ortho_dir / "fingerprint.txt").write_text("ortho_fp_test_12345\n", encoding="utf-8")

    # Write segment artifacts
    conf = np.full((height, width), 0.9, dtype=np.float16)
    np.save(ortho_dir / "class_combined.npy", classes)
    np.save(ortho_dir / "conf_combined.npy", conf)
    cv2.imwrite(str(ortho_dir / "class_combined.png"), classes)

    seg_metadata = {
        "source": "combined",
        "image_model": "sam3",
        "mode": "outdoor",
        "class_list": ["ground", "road", "sidewalk", "curb", "low vegetation", "tree", "building", "manhole"],
        "prompts": [],
        "model_profile": "sam3_outdoor",
        "fingerprint": "seg_fp_test_67890",
        "ortho_fingerprint": "ortho_fp_test_12345",
    }
    (ortho_dir / "segment_combined.json").write_text(json.dumps(seg_metadata, indent=2), encoding="utf-8")

    # Write classified PLY for streaming contour accumulator
    # Grid of points spaced 0.25 m
    px = np.arange(0.2, 19.8, 0.25)
    py = np.arange(0.2, 19.8, 0.25)
    grid_x, grid_y = np.meshgrid(px, py)
    pts_x = grid_x.ravel()
    pts_y = grid_y.ravel()
    pts_z = 100.0 + 0.1 * pts_y
    pts_xyz = np.column_stack((pts_x, pts_y, pts_z))

    # Original coordinates via frame
    pts_orig = frame.to_original(pts_xyz)
    pts_scalar = np.full(len(pts_orig), 1, dtype=np.uint8)  # default ground
    for i in range(len(pts_orig)):
        y = pts_y[i]
        if y < 5.0:
            pts_scalar[i] = 2  # road
        elif y < 7.5:
            pts_scalar[i] = 3  # sidewalk
        else:
            pts_scalar[i] = 5  # low vegetation

    ply_path = project_dir / "products" / "classified" / "classified.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    write_classified_ply(ply_path, pts_orig.astype(np.float32), None, pts_scalar)

    # project.samcloud.yml
    doc = {
        "project": {"name": project_dir.name},
        "settings": {
            "classification": {"class_set": "outdoor", "engine": "sam3"},
            "coordinate_systems": {},
        },
    }
    (project_dir / "project.samcloud.yml").write_text(yaml.safe_dump(doc), encoding="utf-8")

    return {
        "project_dir": project_dir,
        "ortho_dir": ortho_dir,
        "ply_path": ply_path,
        "building_rect": rect_poly,
    }


@pytest.fixture(scope="module")
def scene_data(work_dir: Path) -> dict[str, object]:
    project_dir = work_dir / "test_site"
    return _build_synthetic_scene(project_dir)


def test_outdoor_plan_layers_and_shared_borders(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    assert "S-PAVE" in result.layers and len(result.layers["S-PAVE"]) > 0
    assert "S-WALK" in result.layers and len(result.layers["S-WALK"]) > 0
    assert "S-VEG" in result.layers and len(result.layers["S-VEG"]) > 0
    assert "S-BLDG" in result.layers and len(result.layers["S-BLDG"]) > 0

    # Building regularization: 4 vertices and right angles
    bldg_ents = result.layers["S-BLDG"]
    assert len(bldg_ents) == 1
    bldg_poly = Polygon(bldg_ents[0]["points"])
    coords = list(bldg_poly.exterior.coords)[:-1]
    assert len(coords) == 4

    angles = []
    for i in range(4):
        p_prev = np.array(coords[i - 1])
        p_curr = np.array(coords[i])
        p_next = np.array(coords[(i + 1) % 4])
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        deg = math.degrees(math.acos(np.clip(cos_ang, -1.0, 1.0)))
        angles.append(deg)

    assert all(abs(ang - 90.0) < 0.5 for ang in angles)

    # Coverage validity across all area polygons
    area_polys = []
    for lyr_name in ("S-PAVE", "S-WALK", "S-VEG", "S-BLDG"):
        for ent in result.layers.get(lyr_name, []):
            if ent.get("type") == "polyline" and ent.get("closed"):
                holes = ent.get("holes", [])
                area_polys.append(Polygon(ent["points"], holes))

    assert len(area_polys) >= 4
    assert coverage_is_valid(area_polys)


def test_curb_from_road_sidewalk_boundary_is_long(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    curbs = result.layers.get("S-CURB", [])
    assert len(curbs) <= 2
    total_length = 0.0
    for ent in curbs:
        pts = ent["points"]
        for p1, p2 in zip(pts, pts[1:]):
            total_length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # Band length across scene is 20 m
    assert abs(total_length - 20.0) / 20.0 < 0.10


def test_tree_inserts_scaled_to_crowns(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    tree_ents = [ent for ent in result.layers.get("S-TREE", []) if ent.get("type") == "insert"]
    assert len(tree_ents) == 2
    for ent in tree_ents:
        assert ent["name"] == "TREE"
        assert abs(ent["xscale"] - ent["yscale"]) < 1e-4
        assert 1.5 <= ent["xscale"] <= 4.0
        assert "LABEL" in ent.get("attributes", {})


def test_manhole_block_insert(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    manh_ents = [ent for ent in result.layers.get("S-MANH", []) if ent.get("type") == "insert"]
    assert len(manh_ents) >= 1
    assert any(ent["name"] == "MANHOLE" for ent in manh_ents)


def test_contours_are_3d_on_topo_layers(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    major = result.layers.get("S-TOPO-MAJR", [])
    minor = result.layers.get("S-TOPO-MINR", [])
    assert len(major) > 0
    assert len(minor) > 0

    bldg_poly = Polygon(result.layers["S-BLDG"][0]["points"])

    for ent in major + minor:
        assert ent["type"] == "polyline3d"
        pts = ent["points"]
        for pt in pts:
            assert len(pt) == 3
        z_vals = [pt[2] for pt in pts]
        assert np.allclose(z_vals, z_vals[0], atol=1e-4)

        # No contour vertex inside the building footprint
        for pt in pts:
            assert not bldg_poly.contains(Polygon([(pt[0] - 0.05, pt[1] - 0.05), (pt[0] + 0.05, pt[1] - 0.05), (pt[0] + 0.05, pt[1] + 0.05), (pt[0] - 0.05, pt[1] + 0.05)]))


def test_labels_inside_their_areas_and_min_area(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    settings = CleanPlanSettings(label_min_area_outdoor=15.0)
    result = generate_clean_plan(project_dir, level=None, mode="outdoor", settings=settings)

    labels = result.layers.get("S-ANNO", [])
    assert len(labels) > 0

    all_polys = [
        Polygon(ent["points"])
        for lyr_name in ("S-PAVE", "S-WALK", "S-VEG", "S-BLDG")
        for ent in result.layers.get(lyr_name, [])
        if ent.get("type") == "polyline" and ent.get("closed")
    ]

    for lbl in labels:
        assert lbl["type"] == "text"
        pt = lbl["point"]
        # Must be inside at least one of the polygons
        assert any(p.buffer(0.01).contains(Polygon([(pt[0] - 0.01, pt[1] - 0.01), (pt[0] + 0.01, pt[1] - 0.01), (pt[0] + 0.01, pt[1] + 0.01), (pt[0] - 0.01, pt[1] + 0.01)])) for p in all_polys)


def test_result_roundtrip_and_dxf_export(scene_data: dict[str, object], work_dir: Path) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")

    cache_dir = clean_plan_cache_dir(project_dir, None)
    loaded = load_cached_result(cache_dir)
    assert loaded.variant == "ortho"
    assert loaded.settings_fingerprint == result.settings_fingerprint

    export_dir = work_dir / "exported_dxf"
    dxf_path = export_plan(loaded, export_dir, coordinates="local")
    assert dxf_path.is_file()

    doc = ezdxf.readfile(dxf_path)
    assert "S-TOPO-MAJR" in doc.layers
    layer_topo = doc.layers.get("S-TOPO-MAJR")
    assert layer_topo.rgb == (150, 60, 40)

    msp = doc.modelspace()
    polylines_3d = [e for e in msp if e.dxftype() == "POLYLINE" and e.is_3d_polyline]
    assert len(polylines_3d) > 0

    anno_texts = [e for e in msp if e.dxftype() == "TEXT" and e.dxf.layer == "S-ANNO"]
    assert len(anno_texts) > 0

    tree_inserts = [e for e in msp if e.dxftype() == "INSERT" and e.dxf.name == "TREE"]
    assert len(tree_inserts) == 2
    assert "TREE" in doc.blocks


def test_generate_requires_segmentation(work_dir: Path) -> None:
    empty_proj = work_dir / "empty_proj"
    empty_proj.mkdir(parents=True, exist_ok=True)

    with pytest.raises(OrthoError, match="Build orthophoto first"):
        generate_clean_plan(empty_proj, level=None, mode="outdoor")

    # Now make ortho without segment
    ortho_dir = ortho_output_dir(empty_proj, None)
    ortho_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ortho.png", "dsm.npy", "dtm.npy", "classes_pc.npy", "valid.png", "view_count.png"):
        (ortho_dir / name).touch()
    (ortho_dir / "ortho.json").write_text(json.dumps({"bounds": [0, 0, 10, 10], "pixel_size": 0.1, "width": 100, "height": 100}), encoding="utf-8")
    (ortho_dir / "fingerprint.txt").write_text("dummy\n", encoding="utf-8")

    with pytest.raises(OrthoError, match="Run segmentation first"):
        generate_clean_plan(empty_proj, level=None, mode="outdoor")


def test_cancel_raises_and_keeps_previous_cache(scene_data: dict[str, object]) -> None:
    project_dir = scene_data["project_dir"]
    result = generate_clean_plan(project_dir, level=None, mode="outdoor")
    cache_dir = clean_plan_cache_dir(project_dir, None)
    fingerprint_path = cache_dir / "fingerprint.txt"
    initial_fp = fingerprint_path.read_text(encoding="utf-8").strip()

    token = CancelToken()
    token.cancel()

    with pytest.raises(Cancelled):
        generate_clean_plan(project_dir, level=None, mode="outdoor", cancel=token)

    assert fingerprint_path.read_text(encoding="utf-8").strip() == initial_fp


def _build_indoor_wall_face_scene(project_dir: Path, *, units_per_metre: float) -> tuple[Level, Polygon]:
    """Create a scan-like room with thin wall faces, openings, furniture, and WC."""
    project_dir.mkdir(parents=True, exist_ok=True)
    level = Level("level_1", "1", 0.0, 2.5 * units_per_metre, 0, 1.0, False)
    width, height, pixel = 120, 100, 0.05 * units_per_metre
    ortho_dir = ortho_output_dir(project_dir, level)
    ortho_dir.mkdir(parents=True, exist_ok=True)
    room = Polygon([(0.2 * units_per_metre, 0.2 * units_per_metre), (5.8 * units_per_metre, 0.2 * units_per_metre), (5.8 * units_per_metre, 4.8 * units_per_metre), (0.2 * units_per_metre, 4.8 * units_per_metre)])
    classes = np.zeros((height, width), dtype=np.uint8)
    dsm = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            x, y = (col + 0.5) * pixel, (height - row - 0.5) * pixel
            if room.contains(Polygon([(x - pixel / 2, y - pixel / 2), (x + pixel / 2, y - pixel / 2), (x + pixel / 2, y + pixel / 2), (x - pixel / 2, y + pixel / 2)]).centroid):
                classes[row, col] = 102
    def rows_for(y_value: float) -> int:
        return int(round(height - y_value / pixel))
    def cols_for(x_value: float) -> int:
        return int(round(x_value / pixel))
    left, right, bottom, top = cols_for(0.2 * units_per_metre), cols_for(5.8 * units_per_metre), rows_for(0.2 * units_per_metre), rows_for(4.8 * units_per_metre)
    classes[bottom, left:right + 1] = 100
    classes[top, left:right + 1] = 100
    classes[top:bottom + 1, left] = 100
    classes[top:bottom + 1, right] = 100
    door_columns = slice(cols_for(2.35 * units_per_metre), cols_for(3.25 * units_per_metre))
    window_columns = slice(cols_for(1.15 * units_per_metre), cols_for(2.15 * units_per_metre))
    classes[bottom, door_columns] = 103
    classes[top, window_columns] = 104
    table_rows, table_cols = slice(rows_for(3.5 * units_per_metre), rows_for(2.5 * units_per_metre)), slice(cols_for(2.1 * units_per_metre), cols_for(3.6 * units_per_metre))
    classes[table_rows, table_cols] = 117
    dsm[table_rows, table_cols] = 0.75 * units_per_metre
    wc_rows, wc_cols = slice(rows_for(1.7 * units_per_metre), rows_for(1.1 * units_per_metre)), slice(cols_for(4.4 * units_per_metre), cols_for(4.8 * units_per_metre))
    classes[wc_rows, wc_cols] = 121
    dsm[wc_rows, wc_cols] = 0.55 * units_per_metre
    np.save(ortho_dir / "dsm.npy", dsm)
    np.save(ortho_dir / "dtm.npy", np.zeros_like(dsm))
    np.save(ortho_dir / "classes_pc.npy", classes)
    cv2.imwrite(str(ortho_dir / "ortho.png"), np.full((height, width, 3), 180, dtype=np.uint8))
    cv2.imwrite(str(ortho_dir / "valid.png"), np.full((height, width), 255, dtype=np.uint8))
    cv2.imwrite(str(ortho_dir / "view_count.png"), np.ones((height, width), dtype=np.uint16))
    (ortho_dir / "ortho.json").write_text(json.dumps({"bounds": [0.0, 0.0, width * pixel, height * pixel], "pixel_size": pixel, "width": width, "height": height, "frame": {"origin": [0.0, 0.0, 0.0], "scale": 1.0, "rotation": np.eye(3).tolist()}}), encoding="utf-8")
    (ortho_dir / "fingerprint.txt").write_text("indoor-wall-face-test\n", encoding="utf-8")
    np.save(ortho_dir / "class_combined.npy", classes)
    np.save(ortho_dir / "conf_combined.npy", np.full_like(classes, 0.9, dtype=np.float16))
    cv2.imwrite(str(ortho_dir / "class_combined.png"), classes)
    (ortho_dir / "segment_combined.json").write_text(json.dumps({"source": "combined", "fingerprint": "indoor-seg", "ortho_fingerprint": "indoor-wall-face-test", "model_profile": "sam3_indoor"}), encoding="utf-8")
    fine = np.zeros_like(classes)
    fine[wc_rows, wc_cols] = 1
    np.save(ortho_dir / "fine_class.npy", fine)
    np.save(ortho_dir / "fine_conf.npy", np.where(fine, 0.9, 0.0).astype(np.float16))
    (ortho_dir / "fine_labels.json").write_text(json.dumps(["toilet"]), encoding="utf-8")
    points: list[list[float]] = []
    scalar: list[int] = []
    def add_face(x: float | None, y: float | None, values: np.ndarray, missing_mid: np.ndarray | None = None, *, exclude_low_in_gap: bool = False, scalar_code: int = 2) -> None:
        for value in values:
            in_gap = missing_mid is not None and bool(missing_mid[int(round(value / pixel))])
            for z_factor in (0.16, 0.24, 0.39, 0.46, 0.75, 0.84, 0.92):
                if in_gap and ((0.36 <= z_factor <= 0.52) or (exclude_low_in_gap and z_factor <= 0.32)):
                    continue
                for _replicate in range(3):
                    points.append([x if x is not None else value, y if y is not None else value, z_factor * 2.5 * units_per_metre])
                    scalar.append(scalar_code)
    xs = np.arange(0.2 * units_per_metre, 5.8 * units_per_metre + pixel / 2, pixel)
    ys = np.arange(0.2 * units_per_metre, 4.8 * units_per_metre + pixel / 2, pixel)
    door_missing = np.zeros(width + 1, dtype=bool); door_missing[cols_for(2.35 * units_per_metre):cols_for(3.25 * units_per_metre)] = True
    window_missing = np.zeros(width + 1, dtype=bool); window_missing[cols_for(1.15 * units_per_metre):cols_for(2.15 * units_per_metre)] = True
    add_face(None, 0.2 * units_per_metre, xs, door_missing, exclude_low_in_gap=True)
    add_face(None, 4.8 * units_per_metre, xs, window_missing)
    add_face(0.2 * units_per_metre, None, ys)
    add_face(5.8 * units_per_metre, None, ys)
    point_path = project_dir / "products" / "classified" / "classified.ply"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    write_classified_ply(point_path, np.asarray(points, dtype=np.float32), np.full((len(points), 3), 160, dtype=np.uint8), np.asarray(scalar, dtype=np.uint8))
    (project_dir / "project.samcloud.yml").write_text(yaml.safe_dump({"project": {"name": project_dir.name}, "settings": {"classification": {"class_set": "indoor", "engine": "sam3"}, "coordinate_systems": {}}}), encoding="utf-8")
    return level, room


def _single_line_settings() -> CleanPlanSettings:
    from dataclasses import replace as dataclass_replace

    settings = load_clean_plan_settings()
    return dataclass_replace(settings, walls=dataclass_replace(settings.walls, double_line=False))


def _assert_indoor_single_face_walls(units_per_metre: float, work_dir: Path) -> None:
    level, room = _build_indoor_wall_face_scene(work_dir / f"indoor_faces_{units_per_metre}", units_per_metre=units_per_metre)
    # The single measured face lines (walls.double_line: false) stay available.
    result = generate_clean_plan(work_dir / f"indoor_faces_{units_per_metre}", level=level, mode="indoor", settings=_single_line_settings())
    metrics = indoor_wall_metrics(result)
    assert len(result.layers["A-WALL"]) >= 4
    assert metrics["wall_face_rms_max"] <= 0.01 * units_per_metre
    assert metrics["wall_axis_fraction"] >= 0.95
    assert metrics["faces_inside_free_space"] <= 0.02
    walls = [LineString(entity["points"]) for entity in result.layers["A-WALL"] if entity.get("type") == "polyline"]
    table = Polygon([(2.1 * units_per_metre, 2.5 * units_per_metre), (3.6 * units_per_metre, 2.5 * units_per_metre), (3.6 * units_per_metre, 3.5 * units_per_metre), (2.1 * units_per_metre, 3.5 * units_per_metre)])
    assert not any(wall.intersects(table.buffer(0.05 * units_per_metre)) for wall in walls)
    assert len(result.layers["A-DOOR"]) == 3
    assert not any(entity.get("type") == "insert" and entity.get("name") == "DOOR" for entity in result.layers["A-DOOR"])
    assert len(result.layers["A-WIND"]) == 1
    exported = export_plan(result, work_dir / f"indoor_faces_{units_per_metre}" / "dxf", coordinates="local")
    assert ezdxf.readfile(exported).blocks.get("DOOR") is None
    assert any(entity.get("name") == "WC" or str(entity.get("name")).startswith("WC_WALL") for entity in result.layers["P-SANI"])
    outline = next(entity for entity in result.layers["A-FURN"] if entity.get("type") == "polyline")
    assert len(outline["points"]) == 4
    assert Polygon(outline["points"]).intersection(table).area / Polygon(outline["points"]).union(table).area >= 0.85
    assert abs(float(result.fingerprint_inputs["units_per_metre"]) - units_per_metre) <= 0.05
    assert room.area > 0.0


def test_indoor_single_face_walls_metric(work_dir: Path) -> None:
    _assert_indoor_single_face_walls(1.0, work_dir)


def test_indoor_single_face_walls_scaled_0_8(work_dir: Path) -> None:
    _assert_indoor_single_face_walls(0.8, work_dir)


@pytest.mark.parametrize("units_per_metre", [1.0, 0.8])
def test_indoor_double_line_walls_default(units_per_metre: float, work_dir: Path) -> None:
    project_dir = work_dir / f"indoor_double_{units_per_metre}"
    level, _room = _build_indoor_wall_face_scene(project_dir, units_per_metre=units_per_metre)
    result = generate_clean_plan(project_dir, level=level, mode="indoor")
    walls = result.layers["A-WALL"]
    assert walls and all(entity.get("type") == "polyline" and entity.get("closed") and entity.get("_double_wall") for entity in walls)
    bodies = [Polygon(entity["points"], entity.get("holes", [])) for entity in walls]
    assert all(body.is_valid and body.area > 0.0 for body in bodies)
    # Every body is at least the minimum wall thickness wide and the hatch matches the outlines.
    assert len(result.layers["A-WALL-PATT"]) == len(walls)
    table = Polygon([(2.1 * units_per_metre, 2.5 * units_per_metre), (3.6 * units_per_metre, 2.5 * units_per_metre), (3.6 * units_per_metre, 3.5 * units_per_metre), (2.1 * units_per_metre, 3.5 * units_per_metre)])
    assert not any(body.intersects(table.buffer(0.05 * units_per_metre)) for body in bodies)
    assert result.layers["A-WIND"] and result.layers["A-DOOR"]
    metrics = indoor_wall_metrics(result)
    assert metrics["faces_inside_free_space"] <= 0.02
    assert result.fingerprint_inputs["double_wall_bodies"] >= 1
    exported = export_plan(result, project_dir / "dxf", coordinates="local")
    document = ezdxf.readfile(exported)
    assert not document.audit().has_errors
    assert len(document.modelspace().query('HATCH[layer=="A-WALL-PATT"]')) == len(walls)


def _set_table_fine_label(project_dir: Path, level: Level, label: str) -> None:
    ortho_dir = ortho_output_dir(project_dir, level)
    classes = np.load(ortho_dir / "class_combined.npy")
    fine = np.where(classes == 117, 1, 0).astype(np.uint8)
    np.save(ortho_dir / "fine_class.npy", fine)
    np.save(ortho_dir / "fine_conf.npy", np.where(fine, 0.95, 0.0).astype(np.float16))
    (ortho_dir / "fine_labels.json").write_text(json.dumps([label]), encoding="utf-8")


def test_cat_tree_labelled_kratzbaum(work_dir: Path) -> None:
    project_dir = work_dir / "fine_cat_tree"
    level, _room = _build_indoor_wall_face_scene(project_dir, units_per_metre=1.0)
    _set_table_fine_label(project_dir, level, "cat tree")
    result = generate_clean_plan(project_dir, level=level, mode="indoor")
    assert any(entity.get("type") == "polyline" and entity.get("_label") == "Kratzbaum" for entity in result.layers["A-FURN"])
    assert any(entity.get("text", "").startswith("Kratzbaum") for entity in result.layers["A-ANNO"])
    assert not any(entity.get("type") == "insert" and entity.get("name") == "BATHTUB" for entity in result.layers["A-FURN"])


def test_kitchen_counter_fine_label_to_einb(work_dir: Path) -> None:
    project_dir = work_dir / "fine_kitchen"
    level, _room = _build_indoor_wall_face_scene(project_dir, units_per_metre=1.0)
    _set_table_fine_label(project_dir, level, "kitchen counter")
    result = generate_clean_plan(project_dir, level=level, mode="indoor")
    assert any(entity.get("type") == "polyline" and entity.get("closed") and entity.get("_einb_kind") == "fine_kitchen" for entity in result.layers["A-EINB"])
    assert any(entity.get("type") == "text" and entity.get("text") == "Küche" for entity in result.layers["A-EINB"])
    assert result.fingerprint_inputs["kitchen_einb_from_fine"] == 1


def test_german_umlaut_label_in_dxf(work_dir: Path) -> None:
    result = DxfPlanResult(
        "umlaut", "site", "indoor", None, 1.0, Frame(np.zeros(3), 1.0, np.eye(3)), "umlaut-test",
        {"A-ANNO": [{"type": "text", "text": "Kühlschrank", "point": [0.5, 0.5], "height": 0.12, "rotation": 0.0}]},
        {"A-ANNO": LayerStatus("A-ANNO", "ok", None, 1)}, {"A-ANNO": 7},
        BackgroundImage(None, None, (0.0, 0.0, 1.0, 1.0)),
    )
    document = ezdxf.readfile(export_plan(result, work_dir / "umlaut_dxf", coordinates="local"))
    assert "Kühlschrank" in [entity.dxf.text for entity in document.modelspace().query("TEXT")]


def test_overlay_text_height_tracks_dxf_units() -> None:
    result = DxfPlanResult(
        "overlay", "site", "outdoor", None, 1.0, Frame(np.zeros(3), 1.0, np.eye(3)), "overlay-test",
        {"S-ANNO": [{"type": "text", "text": "H", "point": [0.2, 0.5], "height": 0.1, "rotation": 0.0}]},
        {"S-ANNO": LayerStatus("S-ANNO", "ok", None, 1)}, {"S-ANNO": 1}, BackgroundImage(None, None, (0.0, 0.0, 1.0, 1.0)),
    )
    rendered = render_dxf_rgba(result, size=(100, 100))
    ink = np.argwhere(rendered[..., 3] > 0)
    assert 6 <= int(ink[:, 0].max() - ink[:, 0].min() + 1) <= 14


def test_parallel_wall_duplicates_merge_to_one_rectangle() -> None:
    walls = [Polygon([(0.0, offset), (4.0, offset), (4.0, offset + 0.10), (0.0, offset + 0.10)]) for offset in (0.0, 0.15, 0.30)]
    merged = _merge_parallel_wall_rectangles(walls, 0.5)
    assert len(merged) == 1
    assert abs(merged[0].bounds[3] - merged[0].bounds[1] - 0.40) <= 1e-6


def test_free_standing_wall_stub_is_dropped() -> None:
    grid = RasterGrid(0.0, 2.0, 0.05, 40, 40)
    classes = np.zeros((40, 40), dtype=np.uint8)
    stub = Polygon([(0.5, 0.5), (0.9, 0.5), (0.9, 0.65), (0.5, 0.65)])
    assert _filter_wall_stubs([stub], classes, grid, 1.0) == []


def test_window_with_floor_on_both_sides_is_rejected() -> None:
    grid = RasterGrid(0.0, 4.0, 0.05, 80, 80)
    free = np.ones((80, 80), dtype=bool)
    opening = Opening("WINDOW", (2.0, 2.0), 1.0, 0.0, 0.15)
    assert not _window_is_exterior(opening, free, grid, 1.0)
