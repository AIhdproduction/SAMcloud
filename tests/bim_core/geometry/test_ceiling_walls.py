"""Ceilings end at the inner wall faces (trim_to_walls), derived from the walls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from samcloud.bim.core.geometry import SceneGeometry, compute_scene_geometry
from samcloud.bim.core.model import (
    CeilingElement,
    Scene,
    WallElement,
    WallJoinElement,
)
from samcloud.bim.core.model.graph import DependencyGraph, recompute
from samcloud.bim.core.model.patches import PatchEngine
from samcloud.bim.core.synth.room import RoomLayout, RoomSpec, build_room_scene

SPEC = RoomSpec(layout=RoomLayout.RECTANGLE)
WIDTH, DEPTH = SPEC.size
HALF = SPEC.exterior_thickness / 2.0
TOL = 1e-3


def _ceilings(scene: Scene) -> list[CeilingElement]:
    return [e for e in scene.elements.values() if isinstance(e, CeilingElement)]


def _footprint_union(geometry: SceneGeometry) -> Polygon:
    return unary_union(
        [Polygon(w.footprint_relative_to(geometry.origin)) for w in geometry.walls.values()]
    )


def _outer_envelope_scene() -> Scene:
    """Room whose ceiling envelope reaches the OUTER wall faces (the defect of the first IFC)."""
    scene = build_room_scene(SPEC)
    (ceiling,) = _ceilings(scene)
    outer = [
        (-HALF, -HALF),
        (WIDTH + HALF, -HALF),
        (WIDTH + HALF, DEPTH + HALF),
        (-HALF, DEPTH + HALF),
    ]
    ceiling.params = ceiling.params.model_copy(update={"outline": outer})
    return scene


def _hausdorff(a: Polygon, b: Polygon) -> float:
    return float(a.hausdorff_distance(b))


def test_ceiling_outline_equals_the_inner_wall_face_polygon() -> None:
    scene = _outer_envelope_scene()
    geometry = compute_scene_geometry(scene)
    (ceiling,) = geometry.ceilings.values()
    assert ceiling.trimmed
    outline = Polygon(ceiling.outline)
    inner = Polygon(
        [(HALF, HALF), (WIDTH - HALF, HALF), (WIDTH - HALF, DEPTH - HALF), (HALF, DEPTH - HALF)]
    )
    assert _hausdorff(outline, inner) < TOL
    assert outline.area == pytest.approx(inner.area, abs=1e-3)
    # no vertex inside or outside a wall footprint: every vertex lies on the wall face
    walls = _footprint_union(geometry)
    for x, y in ceiling.outline:
        from shapely.geometry import Point

        point = Point(x - geometry.origin[0], y - geometry.origin[1])
        assert walls.distance(point) < TOL
        assert not walls.buffer(-TOL).covers(point)
    assert (
        Polygon([(x - geometry.origin[0], y - geometry.origin[1]) for x, y in ceiling.outline])
        .intersection(walls)
        .area
        < 1e-6
    )


def test_untrimmed_ceiling_keeps_the_envelope() -> None:
    scene = _outer_envelope_scene()
    (ceiling,) = _ceilings(scene)
    ceiling.params = ceiling.params.model_copy(update={"trim_to_walls": False})
    geometry = compute_scene_geometry(scene)
    (result,) = geometry.ceilings.values()
    assert not result.trimmed
    assert Polygon(result.outline).area == pytest.approx((WIDTH + 2 * HALF) * (DEPTH + 2 * HALF))


def test_ceiling_follows_the_walls_through_the_dependency_graph() -> None:
    scene = build_room_scene(SPEC)
    recompute(scene)
    (ceiling,) = _ceilings(scene)
    walls = [e for e in scene.elements.values() if isinstance(e, WallElement)]
    graph = DependencyGraph.from_scene(scene)
    assert all(graph.has_edge(w.id, ceiling.id) for w in walls)
    # the dependency is also reachable for a changed wall
    assert ceiling.id in graph.affected_by([walls[0].id])


def test_thickness_patch_moves_the_ceiling_and_undo_restores(tmp_path: Path) -> None:
    scene = build_room_scene(SPEC)
    recompute(scene)
    (ceiling,) = _ceilings(scene)
    west = next(
        e
        for e in scene.elements.values()
        if isinstance(e, WallElement) and e.tag == "exterior_west"
    )
    before = json.dumps(scene.derived["elements"][ceiling.id], sort_keys=True)
    area_before = scene.derived["elements"][ceiling.id]["quantities"]["GrossArea"]
    engine = PatchEngine(scene, tmp_path / "patches.jsonl")
    record = engine.apply(
        [{"op": "replace", "path": f"/elements/{west.id}/params/thickness", "value": 0.5}],
        "test",
        "user",
    )
    assert ceiling.id in record.recompute.changed
    outline = [tuple(p) for p in scene.derived["elements"][ceiling.id]["geometry"]["outline"]]
    xs = sorted({round(x, 6) for x, _ in outline})
    assert xs[0] == pytest.approx(0.25)  # west inner face moved from 0.15 to 0.25 m
    area_after = scene.derived["elements"][ceiling.id]["quantities"]["GrossArea"]
    assert area_after == pytest.approx(area_before - 0.1 * (DEPTH - 2 * HALF), abs=1e-6)
    engine.undo()
    assert json.dumps(scene.derived["elements"][ceiling.id], sort_keys=True) == before


def test_open_side_keeps_the_point_based_edge() -> None:
    scene = build_room_scene(SPEC)
    east = next(
        e
        for e in scene.elements.values()
        if isinstance(e, WallElement) and e.tag == "exterior_east"
    )
    for key in [
        k
        for k, e in scene.elements.items()
        if k == east.id
        or (isinstance(e, WallJoinElement) and east.id in (e.params.wall_a, e.params.wall_b))
    ]:
        del scene.elements[key]
    (ceiling,) = _ceilings(scene)
    envelope_east = WIDTH - 0.2  # the points end 0.2 m before the missing east wall
    ceiling.params = ceiling.params.model_copy(
        update={
            "outline": [
                (-HALF, -HALF),
                (envelope_east, -HALF),
                (envelope_east, DEPTH + HALF),
                (-HALF, DEPTH + HALF),
            ]
        }
    )
    geometry = compute_scene_geometry(scene)
    (result,) = geometry.ceilings.values()
    xs = [x for x, _ in result.outline]
    ys = [y for _, y in result.outline]
    assert max(xs) == pytest.approx(envelope_east, abs=TOL)  # open side: point-based edge
    assert min(xs) == pytest.approx(HALF, abs=TOL)  # walled sides: inner faces
    assert min(ys) == pytest.approx(HALF, abs=TOL)
    assert max(ys) == pytest.approx(DEPTH - HALF, abs=TOL)
