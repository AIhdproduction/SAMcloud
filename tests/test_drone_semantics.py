"""Synthetic acceptance tests for drone post-classification semantics."""
from __future__ import annotations

import json

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from samcloud.drone.contracts import BUILDING, GROUND, ROOF_FLAT, ROOF_INCLINED, TREE, BuildingInstance, PointSet
from samcloud.drone.semantics import (
    assign_persistent_ids,
    load_instances,
    plausibility_rules,
    run_semantics,
    save_instances,
    segment_buildings,
    split_roof_facade,
    uncertainty_arrays,
)


def _sampler(xy):
    return np.zeros(len(np.asarray(xy).reshape(-1, 2)))


def _grid(x0, x1, y0, y1, z, step=0.2):
    x, y = np.meshgrid(np.arange(x0, x1 + step / 2, step), np.arange(y0, y1 + step / 2, step))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


def _pointset(xyz, classes, normals=None, confidence=None, margin=None):
    return PointSet(np.asarray(xyz, dtype=float), np.asarray(classes, dtype=np.uint16), normals=normals,
                    confidence=confidence, margin=margin)


def test_split_roof_facade_assigns_export_codes_and_facade_mask():
    flat = _grid(0, 2, 0, 2, 4)
    gable = _grid(3, 5, 0, 2, 4)
    facade = np.column_stack((np.zeros(25), np.linspace(0, 2, 25), np.linspace(0, 4, 25)))
    xyz = np.vstack((flat, gable, facade))
    normals = np.vstack((np.tile([0, 0, 1], (len(flat), 1)), np.tile([0.6, 0, 0.8], (len(gable), 1)),
                         np.tile([1, 0, 0], (len(facade), 1))))
    result, report = split_roof_facade(_pointset(xyz, np.full(len(xyz), BUILDING), normals), _sampler)
    assert np.all(result.classes[:len(flat)] == ROOF_FLAT)
    assert np.all(result.classes[len(flat):len(flat) + len(gable)] == ROOF_INCLINED)
    assert report["is_facade"][-len(facade):].all()
    assert report["roof_flat"] == len(flat) and report["roof_inclined"] == len(gable)


def test_segment_buildings_splits_two_adjacent_and_one_detached_house():
    houses = [_grid(0, 4, 0, 4, 4), _grid(4.1, 8.1, 0, 4, 7), _grid(11, 15, 0, 4, 5)]
    xyz = np.vstack(houses)
    ps = _pointset(xyz, np.full(len(xyz), ROOF_FLAT), np.tile([0, 0, 1], (len(xyz), 1)))
    instances = segment_buildings(ps, _sampler, min_points=100, min_height=0.1)
    assert len(instances) == 3
    expected = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), Polygon([(4.1, 0), (8.1, 0), (8.1, 4), (4.1, 4)]),
                Polygon([(11, 0), (15, 0), (15, 4), (11, 4)])]
    assert min(max(item.footprint.intersection(truth).area / item.footprint.union(truth).area for item in instances)
                   for truth in expected) > 0.9


def test_segment_buildings_handles_dense_class_6_facades():
    roofs = [_grid(0, 4, 0, 4, 6), _grid(10, 14, 0, 4, 6)]
    facades = []
    for x0, x1 in ((0, 4), (10, 14)):
        y, z = np.meshgrid(np.arange(0, 4.01, .25), np.arange(0, 8.01, .25))
        facades.extend((
            np.column_stack((np.full(y.size, x0), y.ravel(), z.ravel())),
            np.column_stack((np.full(y.size, x1), y.ravel(), z.ravel())),
        ))
    xyz = np.vstack((*roofs, *facades))
    ps = _pointset(xyz, np.full(len(xyz), BUILDING), np.tile([0, 0, 1], (len(xyz), 1)))
    instances = segment_buildings(ps, _sampler, min_points=100, min_height=.1)
    assert len(instances) == 2


def test_segment_buildings_reports_small_shed_as_dropped():
    shed = _grid(20, 22, 0, 3, 2.2)
    rejected = segment_buildings(_pointset(shed, np.full(len(shed), ROOF_FLAT), np.tile([0, 0, 1], (len(shed), 1))),
                                 _sampler, min_points=10, min_height=0.1)
    assert not rejected and segment_buildings.last_report["dropped"]


def test_persistent_ids_are_stable_and_increment_for_new_building(tmp_path):
    original = [BuildingInstance(1, np.arange(10), Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))]
    previous = save_instances(original, tmp_path / "buildings" / "instances.json")
    shifted = BuildingInstance(-1, np.arange(9), Polygon([(0.05, 0), (2.05, 0), (2.05, 2), (0.05, 2)]))
    new = BuildingInstance(-1, np.arange(8), Polygon([(5, 0), (7, 0), (7, 2), (5, 2)]))
    assign_persistent_ids([shifted, new], previous)
    assert shifted.building_id == 1 and new.building_id == 2
    assert load_instances(previous)[0].building_id == 1


def test_plausibility_reclassifies_planar_roof_but_keeps_nonplanar_tree():
    roof = _grid(0, 2, 0, 2, 5, 0.15)
    crown = np.column_stack((10 + np.random.default_rng(4).normal(0, .3, 100), np.random.default_rng(5).normal(0, .3, 100),
                             4 + np.random.default_rng(6).normal(0, .6, 100)))
    xyz = np.vstack((roof, crown))
    classes = np.full(len(xyz), TREE, dtype=np.uint16)
    confidence = np.full(len(xyz), .5, dtype=np.float32)
    margin = np.full(len(xyz), .1, dtype=np.float32)
    ps = _pointset(xyz, classes, np.tile([0, 0, 1], (len(xyz), 1)), confidence, margin)
    building = BuildingInstance(1, np.empty(0, dtype=int), Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    result, report = plausibility_rules(ps, _sampler, building_instances=[building])
    assert np.all(result.classes[:len(roof)] == ROOF_FLAT) and np.all(result.classes[len(roof):] == TREE)
    assert report["vegetation_to_roof"] == len(roof)
    values = uncertainty_arrays(ps)
    assert values["confidence"].dtype == np.float32 and values["view_count"] is None


def test_high_confidence_point_is_not_flipped_by_ambiguous_roof_rule():
    noisy = _grid(0, 2, 0, 2, 5, .15)
    noisy[:, 2] += .02 * ((np.arange(len(noisy)) % 2) * 2 - 1)
    ps = _pointset(noisy, np.full(len(noisy), TREE), np.tile([0, 0, 1], (len(noisy), 1)),
                   np.full(len(noisy), .95, dtype=np.float32), np.full(len(noisy), .5, dtype=np.float32))
    building = BuildingInstance(1, np.empty(0, dtype=int), Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    result, report = plausibility_rules(ps, _sampler, building_instances=[building])
    assert np.all(result.classes == TREE) and report["high_confidence_protected"] > 0


def test_plausibility_changes_vertical_ground_to_facade():
    y, z = np.meshgrid(np.linspace(0, 2, 20), np.linspace(.6, 4, 20))
    xyz = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    ps = _pointset(xyz, np.full(len(xyz), GROUND), np.tile([1, 0, 0], (len(xyz), 1)))
    building = BuildingInstance(1, np.empty(0, dtype=int), Polygon([(-.1, 0), (.1, 0), (.1, 2), (-.1, 2)]))
    result, report = plausibility_rules(ps, _sampler, building_instances=[building])
    assert np.all(result.classes == BUILDING) and report["ground_road_to_facade"] == len(ps)


def _write_ply(path, xyz, scalar_class):
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "u1")]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\nproperty uchar scalar_class\nend_header\n")
    data = np.zeros(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"], data["c"] = xyz.T[0], xyz.T[1], xyz.T[2], scalar_class
    path.write_bytes(header.encode("ascii") + data.tobytes())


def _georef():
    latitude, longitude, height = 47.134, 8.755, 450.0
    lat, lon = np.radians(latitude), np.radians(longitude)
    rotation = np.array(([-np.sin(lon), np.cos(lon), 0.0],
                         [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                         [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])).T
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    origin = np.array(to_ecef.transform(longitude, latitude, height))
    return {"scale": 1.0, "rotation": rotation.tolist(), "translation": origin.tolist(),
            "pairs": 5, "residual_m": {"mean": 0.0, "p95": 0.0, "max": 0.0}}


def test_run_semantics_writes_instances_json_for_tiny_products_directory(tmp_path):
    from samcloud.classification.custom import global_classes_path

    project, products = tmp_path / "project", tmp_path / "project" / "products"
    (project / "config").mkdir(parents=True)
    (products / "classified").mkdir(parents=True)
    (project / "config" / "classes.json").write_text(global_classes_path().read_text(encoding="utf-8"), encoding="utf-8")
    (products / "georeference.json").write_text(json.dumps(_georef()), encoding="utf-8")
    ground = _grid(-1, 2, -1, 2, 0, .5)
    roof = _grid(0, 1, 0, 1, 3, .25)
    _write_ply(products / "classified" / "classified.ply", np.vstack((ground, roof)),
               np.r_[np.full(len(ground), 2), np.full(len(roof), 6)].astype(np.uint8))
    result = run_semantics(products, tmp_path / "out")
    assert (tmp_path / "out" / "buildings" / "instances.json").is_file() and "report" in result
