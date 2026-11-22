"""Synthetic bench: scan simulation and project writer produce a project in the real input format (no pipeline run)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from samcloud.bim.synthbench import geometry as geo
from samcloud.bim.synthbench.generate import generate_case
from samcloud.bim.synthbench.mesh import build_scene_mesh
from samcloud.bim.synthbench.scan import simulate_scan
from samcloud.bim.synthbench.scenes import make_scene


def _coarse(spec, **scan_overrides):
    spec.scan = replace(spec.scan, angular_step_deg=1.2, voxel_m=0.05, photos_per_station=2, photo_size=(160, 120), **scan_overrides)
    return spec


def test_scan_is_deterministic_and_points_lie_on_surfaces() -> None:
    spec = _coarse(make_scene("rect_basic", 101), noise_m=0.0)
    mesh = build_scene_mesh(spec)
    a = simulate_scan(spec, mesh)
    b = simulate_scan(spec, mesh)
    assert np.array_equal(a.xyz, b.xyz) and np.array_equal(a.cls, b.cls)
    assert a.stats["stations"] >= 2
    walls = a.xyz[a.true_cls == 2]
    rooms = geo.room_polygons(spec)
    from shapely import distance
    from shapely.geometry import Point

    boundary = rooms[0]["polygon"].boundary
    d = np.array([boundary.distance(Point(p[0], p[1])) for p in walls[:: max(1, len(walls) // 400)]])
    assert np.percentile(d, 90) < 0.01  # noise free wall returns lie on the room boundary (wall faces)
    floor = a.xyz[a.true_cls == 4]
    assert np.abs(floor[:, 2]).max() < 1e-6
    assert set(np.unique(a.cls)) >= {2, 3, 4, 17}
    del distance


def test_unscanned_region_label_noise_and_unvisited_room() -> None:
    spec = _coarse(make_scene("irregular", 106))
    scan = simulate_scan(spec, build_scene_mesh(spec))
    from shapely import contains_xy
    from shapely.geometry import Polygon

    region = Polygon(spec.scan.unscanned[0])
    assert not contains_xy(region, scan.xyz[:, 0], scan.xyz[:, 1]).any()
    assert 0.01 < scan.stats["label_flips"] / scan.stats["points"] < 0.05
    dense = _coarse(make_scene("dense", 107))
    stations = simulate_scan(dense, build_scene_mesh(dense)).stations
    storage = next(r for r in geo.room_polygons(dense) if r["name"] == "Abstellraum")["polygon"]
    assert not contains_xy(storage, stations[:, 0], stations[:, 1]).any()


def test_generated_project_is_readable_by_the_pipeline_readers(tmp_path: Path) -> None:
    import pycolmap
    from plyfile import PlyData

    from samcloud.dxf.las_source import read_las_info
    from samcloud.dxf.state import load_state

    spec = _coarse(make_scene("rect_basic", 101))
    info = generate_case("rect_basic", 101, tmp_path / "case", spec)
    project = Path(info["project"])
    ply = project / "products" / "classified" / "classified.ply"
    vertex = PlyData.read(str(ply))["vertex"]
    names = set(vertex.data.dtype.names)
    assert {"x", "y", "z", "red", "green", "blue", "scalar_class", "fine_term", "instance_id"} <= names
    las_info = read_las_info(ply)
    assert las_info.mode == "indoor"
    assert las_info.profile is not None and las_info.profile.profile_name == "sam3_indoor"
    instances = json.loads((project / "products" / "classified" / "instances.json").read_text(encoding="utf-8"))
    labels = {r["label"] for r in instances["instances"]}
    assert {"sofa", "chair", "shelf"} <= labels
    assert all(r["class_id"] in {5, 6, 9, 15, 17, 18, 19} for r in instances["instances"])
    fine = json.loads((project / "products" / "classified" / "fine_labels.json").read_text(encoding="utf-8"))
    assert fine["ids"]["sofa"] >= 1 and np.count_nonzero(np.asarray(vertex["fine_term"])) > 0
    reconstruction = pycolmap.Reconstruction(str(project / "products" / "sparse" / "0"))
    assert reconstruction.num_reg_images() == info["photos"] and len(reconstruction.points3D) > 100
    assert all((project / "inputs" / "images" / image.name).is_file() for image in reconstruction.images.values())
    state = load_state(project)
    assert state.segment_source == "pointcloud" and state.levels == []
    # the synthetic cloud is in metres and marked metric like a real metric project (plan units_per_metre = 1.0)
    from samcloud.core.metric_scale import metric_scale_decision

    decision = metric_scale_decision(project)
    assert decision.metric and decision.source == "project_setting"
    truth =json.loads((tmp_path / "case" / "gt" / "ground_truth.json").read_text(encoding="utf-8"))
    assert truth["rooms"][0]["area"] == pytest.approx(geo.room_polygons(spec)[0]["area"])
    # the cloud is delivered in the world frame (yaw + offset), not the scene frame
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(float)
    assert np.allclose(spec.to_scene(xyz)[:, 2].min(), 0.0, atol=0.05)
