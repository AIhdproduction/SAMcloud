"""Synthetic bench: scene generator determinism, validity and exact ground-truth geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from samcloud.bim.synthbench import geometry as geo
from samcloud.bim.synthbench.groundtruth import ground_truth
from samcloud.bim.synthbench.mesh import build_scene_mesh, element_area
from samcloud.bim.synthbench.scenes import DEV_SET, VAL_SET, Builder, make_scene
from samcloud.bim.synthbench.spec import ScanSpec, SceneSpec


@pytest.mark.parametrize("family,seed", DEV_SET + VAL_SET)
def test_scene_is_deterministic_and_valid(family: str, seed: int) -> None:
    first = make_scene(family, seed)
    second = make_scene(family, seed)
    assert first.to_dict() == second.to_dict()
    assert geo.validate_scene(first) == []
    assert geo.room_polygons(first)
    assert len(first.furniture) >= 4


def test_dev_and_validation_sets_are_disjoint_and_seeds_change_scenes() -> None:
    assert not set(DEV_SET) & set(VAL_SET)
    assert {f for f, _ in DEV_SET} == {f for f, _ in VAL_SET}
    for (family, dev_seed), (_f, val_seed) in zip(DEV_SET, VAL_SET):
        assert make_scene(family, dev_seed).to_dict()["walls"] != make_scene(family, val_seed).to_dict()["walls"]


def test_spec_json_round_trip() -> None:
    spec = make_scene("structure", 105)
    again = SceneSpec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert again.furniture[0].parts[0].size == spec.furniture[0].parts[0].size


def test_world_transform_round_trip() -> None:
    spec = make_scene("irregular", 106)
    pts = np.array([[1.0, 2.0, 0.5], [-3.0, 0.2, 2.0]])
    assert np.allclose(spec.to_scene(spec.to_world(pts)), pts)
    assert not np.allclose(spec.to_world(pts), pts)


def _box_room(width: float, depth: float, t: float) -> SceneSpec:
    b = Builder("box", "test", 1)
    b.spec.storey_height = 2.5
    s, _e, n, w = b.rect(0, 0, width, depth, t)
    b.room("R", (width / 2, depth / 2), 2.5)
    b.door(s, at=(1.0, 0), width=0.9)
    b.window(n, at=(width / 2, depth), width=1.2, sill=0.9, head=2.1)
    b.against("wardrobe", w, at=[(0, depth / 2)], toward=(width / 2, depth / 2), w=1.2)
    return b.finish(ScanSpec(), [])


def test_room_polygon_is_exact_interior_and_openings_cut_the_wall() -> None:
    spec = _box_room(4.0, 3.0, 0.3)
    rooms = geo.room_polygons(spec)
    assert len(rooms) == 1
    assert rooms[0]["area"] == pytest.approx(12.0, abs=1e-6)
    bands = geo.wall_bands(spec, 2.5)
    # the door band [0, 2.0..] and the window band remove area from the full wall solid
    full = geo.wall_solid(spec).area
    assert any(poly.area < full - 0.25 for _z0, _z1, poly in bands)
    truth = ground_truth(spec)
    assert {o["kind"] for o in truth["openings"]} == {"door", "window"}
    window = next(o for o in truth["openings"] if o["kind"] == "window")
    assert window["sill"] == pytest.approx(0.9) and window["width"] == pytest.approx(1.2)
    assert window["center"][1] == pytest.approx(3.15)  # centre line of the 0.3 m north wall
    assert len(truth["junctions"]) == 4


def test_against_places_back_to_wall_face() -> None:
    spec = _box_room(4.0, 3.0, 0.3)
    wardrobe = spec.furniture[0]
    fp = geo.item_footprint(wardrobe)
    minx, _miny, maxx, _maxy = fp.bounds
    assert minx == pytest.approx(0.02, abs=1e-3)  # 2 cm gap to the west wall face at x = 0
    assert maxx - minx == pytest.approx(0.6, abs=1e-3)
    assert geo.item_z(wardrobe)[0] == pytest.approx(0.0)


def test_mesh_elements_have_expected_areas_and_classes() -> None:
    spec = _box_room(4.0, 3.0, 0.3)
    mesh = build_scene_mesh(spec)
    floor = next(e for e in mesh.elements if e.kind == "floor")
    ceiling = next(e for e in mesh.elements if e.kind == "ceiling")
    assert element_area(ceiling) == pytest.approx(12.0, rel=1e-6)
    assert element_area(floor) == pytest.approx(geo.footprint_union(spec).area, rel=1e-6)
    kinds = {e.kind for e in mesh.elements}
    assert {"wall", "floor", "ceiling", "door", "window", "glass", "furniture"} <= kinds
    assert set(np.unique(mesh.face_class())) >= {2, 3, 4, 5, 6, 17}


def test_slanted_wall_angle_is_in_ground_truth() -> None:
    spec = make_scene("bedroom_slanted", 103)
    angles = []
    for wall in spec.walls:
        d = wall.direction()
        angles.append(math.degrees(math.atan2(d[1], d[0])) % 90.0)
    assert any(5.0 < a < 85.0 for a in angles)


@pytest.mark.parametrize("family,seed", DEV_SET + VAL_SET)
def test_passages_reach_the_ceiling_without_lintel(family: str, seed: int) -> None:
    spec = make_scene(family, seed)
    ceiling = max([spec.storey_height] + [r["ceiling_h"] for r in geo.room_polygons(spec)])
    truth = ground_truth(spec)
    for opening in spec.openings:
        if opening.kind == "passage":
            assert abs(opening.head - ceiling) <= 1e-3
    for gt in truth["openings"]:
        if gt["kind"] == "passage":
            assert abs(gt["head"] - ceiling) <= 1e-3
    # no wall band above a passage: the wall solid is cut up to the top
    for opening in spec.openings:
        if opening.kind != "passage":
            continue
        rect = geo.opening_rectangle(spec, opening)
        above = [b for b in geo.wall_bands(spec, ceiling) if b[0] >= opening.head - 1e-6 and not b[2].is_empty and b[2].intersection(rect).area > 1e-6]
        assert above == []
